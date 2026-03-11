# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe

def _adaptive_threshold(base_threshold, adaptive_alpha, mask_tensor, mask_id):
    """Compute adaptive threshold: τ = τ_0 * (1 - α * (1 - r_mask))"""
    if adaptive_alpha is None or adaptive_alpha == 0 or base_threshold is None:
        return base_threshold
    total = mask_tensor.numel()
    if total == 0:
        return base_threshold
    r_mask = float((mask_tensor == mask_id).sum().item()) / total
    return base_threshold * (1.0 - adaptive_alpha * (1.0 - r_mask))


@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    suffix_mode=None, window_blocks=2, adaptive_alpha=None,
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    seq_len = Lp + gen_length
    _prune = (suffix_mode == 'window')

    # x: (B, Lp + gen_length)
    x = torch.full((B, seq_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        if _prune:
            # -- 1-forward pruned warm-up --
            keep = list(range(e))   # prefix + current block
            w_end = min(Lp + (nb + 1 + window_blocks) * block_length, seq_len)
            w_end = max(w_end, e)
            keep.extend(range(e, w_end))
            if keep[-1] != seq_len - 1:
                keep.append(seq_len - 1)

            keep_t = torch.tensor(keep, device=model.device, dtype=torch.long)
            x_prun = x[:, keep_t]
            pos_ids = keep_t.unsqueeze(0).expand(B, -1)

            out_full = model(x_prun, use_cache=True, position_ids=pos_ids)
            nfe += 1

            blk_logits = out_full.logits[:, s:e, :]
            past_kv_raw = out_full.past_key_values
            del out_full

            # Trim KV: remove block entries [s, e), keep prefix + suffix context
            context_kv = []
            for layer_kv in past_kv_raw:
                context_kv.append(tuple(
                    torch.cat([t[:, :, :s, :], t[:, :, e:, :]], dim=2)
                    for t in layer_kv
                ))
            del past_kv_raw

            # Refine position_ids: context positions (prefix + suffix) then block
            ctx_positions = keep[:s] + keep[e:]
            blk_positions = list(range(s, e))
            refine_pos_ids = torch.tensor(
                ctx_positions + blk_positions, device=model.device, dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)

            # Step 0: initial transfer from pruned logits (covers prefix + block)
            gmi = (x[:, :e] == mask_id)
            gmi[:, e:] = False
            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
            if factor is None:
                q0 = None if th_step is not None else num_transfer_tokens[:, 0]
                x0, ti = get_transfer_index(
                    blk_logits, temperature, remasking,
                    gmi[:, s:e], x[:, s:e], q0, th_step)
            else:
                x0, ti = get_transfer_index_dynamic(
                    blk_logits, temperature, remasking,
                    gmi[:, s:e], x[:, s:e], None, factor)
            blk_new = torch.where(ti, x0, x[:, s:e])
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

        else:
            # -- Original full forward warm-up --
            out_full = model(x, use_cache=True)
            past_key_values = out_full.past_key_values
            nfe += 1

            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, s:e] = True

            global_mask_index = (x == mask_id)
            global_mask_index[:, e:] = False
            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
            if factor is None:
                q0 = None if th_step is not None else num_transfer_tokens[:, 0]
                x0, ti = get_transfer_index(
                    out_full.logits, temperature, remasking, global_mask_index, x, q0, th_step)
            else:
                x0, ti = get_transfer_index_dynamic(
                    out_full.logits, temperature, remasking, global_mask_index, x, None, factor)
            x = torch.where(ti, x0, x)

        # 2) Refinement steps
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            if _prune:
                logits_blk = model(
                    x[:, s:e], past_key_values=context_kv,
                    use_cache=False, position_ids=refine_pos_ids,
                ).logits
            else:
                logits_blk = model(
                    x[:, s:e], past_key_values=past_key_values,
                    use_cache=True, replace_position=replace_position,
                ).logits

            mask_blk = (x[:, s:e] == mask_id)
            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
            if factor is None:
                qi = None if th_step is not None else num_transfer_tokens[:, i]
                x0_blk, ti_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], qi, th_step)
            else:
                x0_blk, ti_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor)

            blk_new = torch.where(ti_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            nfe += 1

    return x, nfe


@torch.no_grad()
def generate_with_streaming_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    window_blocks=2, adaptive_alpha=None,
):
    """
    Streaming-style dual-cache: pruned warm-up + suffix as INPUT during refine.

    Unlike generate_with_dual_cache (suffix_mode='window') which keeps suffix
    in the KV cache (frozen/stale), this function moves suffix tokens to the
    input side so they participate in bidirectional attention at every layer
    and every refine step.

    KV cache during refine contains only the prefix; suffix + block are both
    part of the model input.
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    seq_len = Lp + gen_length

    x = torch.full((B, seq_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # -- Build suffix indices (window + tail) --
        w_end = min(Lp + (nb + 1 + window_blocks) * block_length, seq_len)
        w_end = max(w_end, e)
        suffix_positions = list(range(e, w_end))
        if not suffix_positions or suffix_positions[-1] != seq_len - 1:
            suffix_positions.append(seq_len - 1)
        suffix_len = len(suffix_positions)
        suffix_t = torch.tensor(suffix_positions, device=model.device, dtype=torch.long)

        # -- Pruned warm-up: prefix + block + suffix --
        keep = list(range(e)) + suffix_positions
        keep_t = torch.tensor(keep, device=model.device, dtype=torch.long)
        x_prun = x[:, keep_t]
        pos_ids_warmup = keep_t.unsqueeze(0).expand(B, -1)

        out_full = model(x_prun, use_cache=True, position_ids=pos_ids_warmup)
        nfe += 1

        blk_logits = out_full.logits[:, s:e, :]
        past_kv_raw = out_full.past_key_values
        del out_full

        # -- Trim KV to prefix only [0, s) --
        prefix_kv = []
        for layer_kv in past_kv_raw:
            prefix_kv.append(tuple(t[:, :, :s, :] for t in layer_kv))
        del past_kv_raw

        # -- Position IDs for refine: prefix positions (in KV) + block + suffix (in input) --
        prefix_positions = list(range(s))
        blk_positions = list(range(s, e))
        input_positions = blk_positions + suffix_positions
        refine_pos_ids = torch.tensor(
            prefix_positions + input_positions, device=model.device, dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)

        # -- Step 0: initial transfer from warm-up logits --
        th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
        blk_mask_0 = (x[:, s:e] == mask_id)
        if factor is None:
            q0 = None if th_step is not None else num_transfer_tokens[:, 0]
            x0, ti = get_transfer_index(
                blk_logits, temperature, remasking,
                blk_mask_0, x[:, s:e], q0, th_step)
        else:
            x0, ti = get_transfer_index_dynamic(
                blk_logits, temperature, remasking,
                blk_mask_0, x[:, s:e], None, factor)
        blk_new = torch.where(ti, x0, x[:, s:e])
        x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

        # -- Refinement steps: block + suffix as input, prefix as KV --
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            input_tokens = torch.cat([x[:, s:e], x[:, suffix_t]], dim=1)
            out_ref = model(
                input_tokens, past_key_values=prefix_kv,
                use_cache=False, position_ids=refine_pos_ids,
            )
            logits_blk = out_ref.logits[:, :block_length, :]
            del out_ref

            mask_blk = (x[:, s:e] == mask_id)
            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
            if factor is None:
                qi = None if th_step is not None else num_transfer_tokens[:, i]
                x0_blk, ti_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], qi, th_step)
            else:
                x0_blk, ti_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor)

            blk_new = torch.where(ti_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            nfe += 1

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache_expand(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    mid_trigger_ratio=0.5,
    rewarm_on_expand=True,
    front_block_fallback_only=False,
    hfb=None,
    record_steps=False,
    suffix_mode=None,
    window_blocks=2,
    adaptive_alpha=None,
    streaming_refine=False,
    expand_trigger_mode='ratio',
    vel_threshold=0.5,
    conf_threshold=0.7,
):
    """
    Dual-cache block-wise generation with mid-block chain expansion.

    Behaviour identical to generate_with_dual_cache when expansion never
    triggers (i.e. all masks in a block are resolved before the midpoint).

    Expansion trigger modes
    -----------------------
    expand_trigger_mode : str, default 'ratio'
        'ratio'        – (original) expand when remaining masks <= block_length
                         * mid_trigger_ratio.
        'velocity'     – expand when transferred / block_length >= vel_threshold
                         (ratio of tokens decoded in the last step relative to
                         the block size; linear and directly interpretable).
        'confidence'   – expand when the mean softmax confidence of remaining
                         masked positions in the watched block >= conf_threshold
                         (measures model certainty, zero extra forward passes).
        'conf_decoded' – like 'confidence' but measures the mean confidence of
                         already-decoded positions (how much the model agrees
                         with its earlier decisions).
        'conf_all'     – like 'confidence' but averages over ALL positions in
                         the watched block (decoded + masked).
        'vel_and_conf' – expand only when BOTH velocity AND confidence
                         conditions are satisfied (most conservative).
    vel_threshold : float, default 0.5
        Velocity ratio threshold: transferred_last_step / block_length >=
        vel_threshold triggers expansion.  E.g. 0.2 = aggressive (20% of
        the block decoded in one step), 0.8 = conservative.
    conf_threshold : float, default 0.7
        Minimum mean confidence of remaining masks to trigger expansion
        (used by 'confidence' and 'vel_and_conf' modes).

    Other hyper-parameters
    ----------------------
    mid_trigger_ratio : float, default 0.5
        When the fraction of remaining masks in the *watched* original block
        drops to this ratio (e.g. 0.5 → half decoded), expansion is triggered.
    rewarm_on_expand : bool, default True
        If True, a full forward pass (re-warm KV cache) is performed on each
        expansion.  Set to False to skip re-warm and let the next block-level
        refinement step update the KV instead (saves 1 NFE per expansion).
    front_block_fallback_only : bool, default False
        If True, threshold fallback ("at least one token transfer") is only
        allowed on the current front unresolved block.
    hfb : float or None, default None
        Hybrid front-block fallback ratio.  When set (e.g. 0.66), the
        front_block_fallback_only restriction is *disabled* until the
        fraction of decoded (non-mask) tokens in the generation window
        reaches this ratio.  Before that point, fallback is unrestricted;
        after that point, it is limited to the front unresolved block.
        None or 0 disables this feature (fb follows front_block_fallback_only
        at all times).
    record_steps : bool, default False
        If True, return a third element: a list of per-step dicts recording
        tokens transferred, remaining masks, step type, etc.
        Return signature becomes (x, nfe, step_records).
    suffix_mode : str or None, default None
        Suffix pruning strategy for warm-up / rewarm.
        None     – standard full forward (backward compatible).
        'window' – keep window_blocks * block_length suffix + tail token.
                   window_blocks=0 gives tail-only (minimal suffix).
    window_blocks : int, default 2
        Number of extra block-lengths of suffix to keep in 'window' mode.

    Chain expansion example  (block_length=32, 3 blocks)
    -----------------------------------------------------
    Block 0 refinement → midpoint triggered → expand range to Block 1
    → Block 1 midpoint triggered → expand range to Block 2
    → finish refinement on the merged super-block → nb jumps by 3.
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    seq_len = Lp + gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, seq_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0
    trigger_thresh = int(block_length * mid_trigger_ratio)

    # ---- Expand-trigger tracking (velocity / confidence modes) ----
    _use_vel = expand_trigger_mode in ('velocity', 'vel_and_conf')
    _use_conf = expand_trigger_mode in ('confidence', 'conf_decoded', 'conf_all', 'vel_and_conf')
    last_transferred = 0       # tokens decoded in the most recent refinement step
    last_remaining_before = 0  # remaining masks BEFORE the most recent step
    last_logits_blk = None     # logits covering current [s, e) for confidence calc

    # ---- Suffix-pruning helpers ----
    _prune = (suffix_mode == 'window')
    _stream = streaming_refine
    if _stream:
        _prune = True

    def _build_keep_indices(e_pos, front_nb):
        """Position indices to keep for a pruned forward ending at e_pos.
        Window covers front_nb + 1 + window_blocks original blocks from Lp.
        window_blocks=0 → only tail token (no extra suffix)."""
        keep = list(range(e_pos))
        w_end = min(Lp + (front_nb + 1 + window_blocks) * block_length, seq_len)
        w_end = max(w_end, e_pos)
        keep.extend(range(e_pos, w_end))
        if keep[-1] != seq_len - 1:
            keep.append(seq_len - 1)
        return keep

    def _trim_kv(past_kv, rm_start, rm_end):
        """Remove KV entries at indices [rm_start, rm_end) from cache."""
        trimmed = []
        for layer_kv in past_kv:
            layer_t = []
            for kv_t in layer_kv:
                layer_t.append(torch.cat([kv_t[:, :, :rm_start, :],
                                          kv_t[:, :, rm_end:, :]], dim=2))
            trimmed.append(tuple(layer_t))
        return trimmed

    def _pruned_warmup(x_seq, s_pos, e_pos, front_nb):
        """
        1-forward pruned warm-up.
        Returns (block_logits, context_kv, refine_pos_ids, suffix_t).
        suffix_t is None when not streaming, or a LongTensor of suffix
        positions when streaming (those positions become model input
        during refine instead of staying in KV).
        """
        nonlocal nfe
        keep_idx = _build_keep_indices(e_pos, front_nb)
        keep_t = torch.tensor(keep_idx, device=x_seq.device, dtype=torch.long)
        x_prun = x_seq[:, keep_t]
        pos_ids = keep_t.unsqueeze(0).expand(B, -1)

        out = model(x_prun, use_cache=True, position_ids=pos_ids)
        nfe += 1
        blk_logits = out.logits[:, s_pos:e_pos, :]
        past_kv = out.past_key_values
        del out

        if _stream:
            prefix_kv = []
            for layer_kv in past_kv:
                prefix_kv.append(tuple(t[:, :, :s_pos, :] for t in layer_kv))
            del past_kv

            suffix_pos = keep_idx[e_pos:]
            s_t = torch.tensor(suffix_pos, device=x_seq.device, dtype=torch.long)
            prefix_positions = list(range(s_pos))
            blk_positions = list(range(s_pos, e_pos))
            ref_pos = torch.tensor(
                prefix_positions + blk_positions + suffix_pos,
                device=x_seq.device, dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)
            return blk_logits, prefix_kv, ref_pos, s_t
        else:
            ctx_kv = _trim_kv(past_kv, s_pos, e_pos)
            del past_kv

            ctx_positions = keep_idx[:s_pos] + keep_idx[e_pos:]
            blk_positions = list(range(s_pos, e_pos))
            ref_pos = torch.tensor(
                ctx_positions + blk_positions,
                device=x_seq.device, dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)
            return blk_logits, ctx_kv, ref_pos, None

    # hfb: hybrid front-block fallback – fb only activates after enough tokens decoded
    hfb_threshold = int(gen_length * hfb) if hfb else 0  # 0 means hfb disabled

    def _hfb_active() -> bool:
        """Return True when front_block_fallback should be enforced."""
        if not front_block_fallback_only:
            return False
        if hfb_threshold <= 0:
            # hfb disabled → fb always active (original behaviour)
            return True
        decoded = int((x[:, Lp:Lp + gen_length] != mask_id).sum(dim=1).max().item())
        return decoded >= hfb_threshold

    # Step recording for per-step analysis
    step_records = []
    global_step = 0

    nb = 0
    while nb < num_blocks:
        s = Lp + nb * block_length
        e = s + block_length

        # ---- Phase 1: warm KV-cache on current block ----
        block_mask = (x[:, s:e] == mask_id)
        num_tt = get_num_transfer_tokens(block_mask, steps_per_block)

        if _prune:
            # -- Suffix-pruned 1-forward warm-up --
            blk_logits, context_kv, refine_pos_ids, suffix_t = _pruned_warmup(x, s, e, nb)

            blk_mask_0 = (x[:, s:e] == mask_id)
            fb_mask_0 = None
            if _hfb_active():
                fb_mask_0 = torch.ones_like(blk_mask_0, dtype=torch.bool)
            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
            if factor is None:
                q0 = None if th_step is not None else num_tt[:, 0]
                x0_blk, ti_blk = get_transfer_index(
                    blk_logits, temperature, remasking,
                    blk_mask_0, x[:, s:e], q0, th_step,
                    allow_fallback=True, fallback_mask=fb_mask_0,
                )
            else:
                x0_blk, ti_blk = get_transfer_index_dynamic(
                    blk_logits, temperature, remasking,
                    blk_mask_0, x[:, s:e], None, factor)
            blk_new = torch.where(ti_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            ti = ti_blk  # for record_steps
            if _use_conf:
                last_logits_blk = blk_logits
        else:
            # -- Original full-forward warm-up --
            suffix_t = None
            out = model(x, use_cache=True)
            past_kv = out.past_key_values
            nfe += 1

            rp = torch.zeros_like(x, dtype=torch.bool)
            rp[:, s:e] = True

            gmi = (x == mask_id)
            gmi[:, e:] = False
            warm_fallback_mask = None
            if _hfb_active():
                warm_fallback_mask = torch.zeros_like(gmi, dtype=torch.bool)
                warm_fallback_mask[:, s:e] = True
            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
            if factor is None:
                q0 = None if th_step is not None else num_tt[:, 0]
                x0, ti = get_transfer_index(
                    out.logits, temperature, remasking, gmi, x, q0, th_step,
                    allow_fallback=True, fallback_mask=warm_fallback_mask,
                )
            else:
                x0, ti = get_transfer_index_dynamic(
                    out.logits, temperature, remasking, gmi, x, None, factor)
            x = torch.where(ti, x0, x)
            if _use_conf:
                last_logits_blk = out.logits[:, s:e, :].clone()

        if record_steps:
            step_records.append({
                'global_step': global_step, 'block': nb,
                'type': 'warm', 'transferred': int(ti.sum().item()),
                'remaining': int((x[:, s:e] == mask_id).sum().item()),
                'range': (s - Lp, e - Lp),
                'mask_snapshot': (x[0, Lp:] == mask_id).cpu().tolist(),
            })
            global_step += 1

        # ---- Phase 2: refinement with potential chain expansions ----
        watching_nb = nb          # which original block we check for midpoint
        blocks_consumed = 1       # total original blocks this iteration covers
        step_idx = 1              # pointer into current num_tt schedule
        # For rewarm_on_expand=False: track which blocks' completion triggers
        # a deferred full forward (block-boundary re-warm).
        # List to support chain: Block0 pending + Block1 midpoint before Block0 done.
        pending_rewarm_blocks = []

        def _get_front_unresolved_block() -> int:
            for blk_idx in range(nb, watching_nb + 1):
                blk_s = Lp + blk_idx * block_length
                blk_e = min(blk_s + block_length, Lp + gen_length)
                if bool((x[:, blk_s:blk_e] == mask_id).any().item()):
                    return blk_idx
            return watching_nb

        def _build_front_fallback_mask(mask_tensor: torch.Tensor, seq_start: int):
            if not _hfb_active():
                return None

            front_nb = _get_front_unresolved_block()
            front_s = Lp + front_nb * block_length
            front_e = min(front_s + block_length, Lp + gen_length)

            seq_end = seq_start + int(mask_tensor.shape[1])
            local_s = max(front_s, seq_start)
            local_e = min(front_e, seq_end)

            fallback_mask = torch.zeros_like(mask_tensor, dtype=torch.bool)
            if local_s < local_e:
                fallback_mask[:, local_s - seq_start:local_e - seq_start] = True
            return fallback_mask

        while step_idx < num_tt.shape[1]:
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            # -- [no_rewarm] Check if ANY pending block is fully decoded → full forward --
            if not rewarm_on_expand and pending_rewarm_blocks:
                completed = [
                    pb for pb in pending_rewarm_blocks
                    if int((x[:, Lp + pb * block_length : Lp + (pb + 1) * block_length]
                            == mask_id).sum(dim=1).max().item()) == 0
                ]
                if completed:
                    for pb in completed:
                        pending_rewarm_blocks.remove(pb)

                    if _prune:
                        blk_logits, context_kv, refine_pos_ids, suffix_t = _pruned_warmup(x, s, e, nb)
                        blk_mask_rw = (x[:, s:e] == mask_id)
                        fb_mask_rw = None
                        if _hfb_active():
                            fb_mask_rw = _build_front_fallback_mask(blk_mask_rw, s)
                        th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
                        if factor is None:
                            q0 = None if th_step is not None else num_tt[:, step_idx]
                            x0_blk, ti_blk = get_transfer_index(
                                blk_logits, temperature, remasking,
                                blk_mask_rw, x[:, s:e], q0, th_step,
                                allow_fallback=True, fallback_mask=fb_mask_rw,
                            )
                        else:
                            x0_blk, ti_blk = get_transfer_index_dynamic(
                                blk_logits, temperature, remasking,
                                blk_mask_rw, x[:, s:e], None, factor)
                        blk_new = torch.where(ti_blk, x0_blk, x[:, s:e])
                        x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
                        ti = ti_blk
                        if _use_vel:
                            last_remaining_before = int(blk_mask_rw.sum().item())
                            last_transferred = int(ti_blk.sum().item())
                        if _use_conf:
                            last_logits_blk = blk_logits
                    else:
                        out = model(x, use_cache=True)
                        past_kv = out.past_key_values
                        nfe += 1

                        rp = torch.zeros_like(x, dtype=torch.bool)
                        rp[:, s:e] = True

                        gmi = (x == mask_id)
                        gmi[:, e:] = False
                        th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
                        if factor is None:
                            q0 = None if th_step is not None else num_tt[:, step_idx]
                            x0, ti = get_transfer_index(
                                out.logits, temperature, remasking,
                                gmi, x, q0, th_step,
                                allow_fallback=True,
                                fallback_mask=_build_front_fallback_mask(gmi, 0),
                            )
                        else:
                            x0, ti = get_transfer_index_dynamic(
                                out.logits, temperature, remasking,
                                gmi, x, None, factor)
                        x = torch.where(ti, x0, x)
                        if _use_vel:
                            last_remaining_before = int(gmi[:, s:e].sum().item())
                            last_transferred = int(ti.sum().item())
                        if _use_conf:
                            last_logits_blk = out.logits[:, s:e, :].clone()
                    if record_steps:
                        step_records.append({
                            'global_step': global_step, 'block': watching_nb,
                            'type': 'block_rewarm',
                            'transferred': int(ti.sum().item()),
                            'remaining': int((x[:, s:e] == mask_id).sum().item()),
                            'range': (s - Lp, e - Lp),
                            'mask_snapshot': (x[0, Lp:] == mask_id).cpu().tolist(),
                        })
                        global_step += 1
                    step_idx += 1
                    continue

            # -- Midpoint check on the watched block --
            can_expand = (watching_nb + 1 < num_blocks)
            if can_expand:
                wb_s = Lp + watching_nb * block_length
                wb_e = wb_s + block_length
                remaining_masks = int(
                    (x[:, wb_s:wb_e] == mask_id).sum(dim=1).max().item())

                # --- Trigger decision based on expand_trigger_mode ---
                if expand_trigger_mode == 'ratio':
                    _trigger = (remaining_masks <= trigger_thresh)
                else:
                    _vel_ok = True
                    _conf_ok = True
                    if _use_vel:
                        _vel_ratio = last_transferred / max(block_length, 1)
                        _vel_ok = (_vel_ratio >= vel_threshold)
                    if _use_conf:
                        wb_mask = (x[:, wb_s:wb_e] == mask_id)
                        if last_logits_blk is None:
                            _conf_ok = not wb_mask.any()
                        else:
                            ov_s = max(wb_s, s) - s
                            ov_e = min(wb_e, e) - s
                            if ov_s >= ov_e:
                                _conf_ok = True
                            else:
                                _lg = last_logits_blk[:, ov_s:ov_e, :]
                                _probs = F.softmax(_lg.to(torch.float32), dim=-1)
                                _preds = torch.argmax(_lg, dim=-1)
                                mask_sl = wb_mask[:, max(wb_s, s) - wb_s : min(wb_e, e) - wb_s]
                                _x_sl = x[:, max(wb_s, s):min(wb_e, e)]
                                _tokens = torch.where(mask_sl, _preds, _x_sl)
                                _cv = torch.gather(_probs, -1, _tokens.unsqueeze(-1)).squeeze(-1)

                                _cmode = expand_trigger_mode
                                if _cmode in ('confidence', 'vel_and_conf'):
                                    # remaining masks only
                                    _conf_ok = ((_cv[mask_sl].mean().item() >= conf_threshold)
                                                if mask_sl.any() else True)
                                elif _cmode == 'conf_decoded':
                                    # already decoded positions only
                                    decoded_sl = ~mask_sl
                                    _conf_ok = ((_cv[decoded_sl].mean().item() >= conf_threshold)
                                                if decoded_sl.any() else True)
                                else:  # conf_all
                                    _conf_ok = (_cv.mean().item() >= conf_threshold)
                    if expand_trigger_mode == 'velocity':
                        _trigger = _vel_ok
                    elif expand_trigger_mode in ('confidence', 'conf_decoded', 'conf_all'):
                        _trigger = _conf_ok
                    else:
                        _trigger = _vel_ok and _conf_ok

                if _trigger:
                    # ========== EXPAND ==========
                    next_nb = watching_nb + 1
                    e_new = min(Lp + (next_nb + 1) * block_length,
                                Lp + gen_length)

                    # Shrink start to first remaining mask in [s, e_new)
                    mask_pos = (x[0, s:e_new] == mask_id).nonzero(as_tuple=True)[0]
                    s = (s + mask_pos[0].item()) if len(mask_pos) > 0 else s
                    e = e_new

                    if rewarm_on_expand:
                        if _prune:
                            blk_logits, context_kv, refine_pos_ids, suffix_t = _pruned_warmup(x, s, e, nb)
                        else:
                            out = model(x, use_cache=True)
                            past_kv = out.past_key_values
                            nfe += 1
                    else:
                        pending_rewarm_blocks.append(watching_nb)

                    if not _prune:
                        rp = torch.zeros_like(x, dtype=torch.bool)
                        rp[:, s:e] = True

                    # Fresh transfer schedule (one block's worth of steps)
                    exp_mask = (x[:, s:e] == mask_id)
                    num_tt = get_num_transfer_tokens(exp_mask, steps_per_block)
                    step_idx = 0

                    blocks_consumed += 1
                    watching_nb = next_nb

                    # If re-warmed, use fresh logits for step-0 of expanded
                    if rewarm_on_expand:
                        if _prune:
                            blk_mask_ex = (x[:, s:e] == mask_id)
                            fb_mask_ex = None
                            if _hfb_active():
                                fb_mask_ex = _build_front_fallback_mask(blk_mask_ex, s)
                            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
                            if factor is None:
                                q0 = None if th_step is not None else num_tt[:, 0]
                                x0_blk, ti_blk = get_transfer_index(
                                    blk_logits, temperature, remasking,
                                    blk_mask_ex, x[:, s:e], q0, th_step,
                                    allow_fallback=True, fallback_mask=fb_mask_ex,
                                )
                            else:
                                x0_blk, ti_blk = get_transfer_index_dynamic(
                                    blk_logits, temperature, remasking,
                                    blk_mask_ex, x[:, s:e], None, factor)
                            blk_new = torch.where(ti_blk, x0_blk, x[:, s:e])
                            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
                            ti = ti_blk
                            if _use_vel:
                                last_remaining_before = int(blk_mask_ex.sum().item())
                                last_transferred = int(ti_blk.sum().item())
                            if _use_conf:
                                last_logits_blk = blk_logits
                        else:
                            gmi = (x == mask_id)
                            gmi[:, e:] = False
                            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
                            if factor is None:
                                q0 = None if th_step is not None else num_tt[:, 0]
                                x0, ti = get_transfer_index(
                                    out.logits, temperature, remasking,
                                    gmi, x, q0, th_step,
                                    allow_fallback=True,
                                    fallback_mask=_build_front_fallback_mask(gmi, 0),
                                )
                            else:
                                x0, ti = get_transfer_index_dynamic(
                                    out.logits, temperature, remasking,
                                    gmi, x, None, factor)
                            x = torch.where(ti, x0, x)
                            if _use_vel:
                                last_remaining_before = int(gmi[:, s:e].sum().item())
                                last_transferred = int(ti.sum().item())
                            if _use_conf:
                                last_logits_blk = out.logits[:, s:e, :].clone()
                        if record_steps:
                            step_records.append({
                                'global_step': global_step, 'block': watching_nb,
                                'type': 'expand', 'transferred': int(ti.sum().item()),
                                'remaining': int((x[:, s:e] == mask_id).sum().item()),
                                'range': (s - Lp, e - Lp),
                                'mask_snapshot': (x[0, Lp:] == mask_id).cpu().tolist(),
                            })
                            global_step += 1
                        step_idx = 1
                        continue
                    # rewarm_on_expand=False: fall through to normal refinement
                    # ========== END EXPAND ==========

            # -- Normal refinement step --
            if _prune:
                if _stream and suffix_t is not None:
                    refine_input = torch.cat([x[:, s:e], x[:, suffix_t]], dim=1)
                else:
                    refine_input = x[:, s:e]
                logits_blk = model(
                    refine_input, past_key_values=context_kv,
                    use_cache=False, position_ids=refine_pos_ids,
                ).logits
                if _stream and suffix_t is not None:
                    logits_blk = logits_blk[:, :e - s, :]
            else:
                logits_blk = model(
                    x[:, s:e], past_key_values=past_kv,
                    use_cache=True, replace_position=rp
                ).logits

            mask_blk = (x[:, s:e] == mask_id)
            th_step = _adaptive_threshold(threshold, adaptive_alpha, x[:, s:e], mask_id)
            if factor is None:
                qi = None if th_step is not None else num_tt[:, step_idx]
                x0_blk, ti_blk = get_transfer_index(
                    logits_blk, temperature, remasking,
                    mask_blk, x[:, s:e], qi, th_step,
                    allow_fallback=True,
                    fallback_mask=_build_front_fallback_mask(mask_blk, s),
                )
            else:
                x0_blk, ti_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking,
                    mask_blk, x[:, s:e], None, factor)

            blk_new = torch.where(ti_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            nfe += 1
            if _use_vel:
                last_remaining_before = int(mask_blk.sum().item())
                last_transferred = int(ti_blk.sum().item())
            if _use_conf:
                last_logits_blk = logits_blk
            if record_steps:
                step_records.append({
                    'global_step': global_step, 'block': watching_nb,
                    'type': 'refine', 'transferred': int(ti_blk.sum().item()),
                    'remaining': int((blk_new == mask_id).sum().item()),
                    'range': (s - Lp, e - Lp),
                    'mask_snapshot': (x[0, Lp:] == mask_id).cpu().tolist(),
                })
                global_step += 1
            step_idx += 1

        nb += blocks_consumed

    if record_steps:
        return x, nfe, step_records
    return x, nfe


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
    allow_fallback: bool = True,
    fallback_mask: torch.Tensor = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        if allow_fallback:
            # at least one token is transferred "always unmask max c^i"
            if fallback_mask is None:
                fallback_candidates = mask_index
            else:
                fallback_candidates = mask_index & fallback_mask

            cand_conf = torch.where(fallback_candidates, confidence, neg_inf)
            has_candidate = fallback_candidates.any(dim=1, keepdim=True)
            max_conf_indices = torch.argmax(cand_conf, dim=1, keepdim=True)  # (B, 1)
            force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
            force_mask = force_mask & has_candidate

            # (Above Threshold) OR (Is Max Confidence)
            transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

# ---------------------------------------------------------------------------
# Saber-integrated generation methods (arXiv:2510.18165)
#   AADU = Adaptive Acceleration via Dynamic Unmasking
#   BERM = Backtracking-Enhanced Remasking Mechanism
# ---------------------------------------------------------------------------

def _saber_select(confidence, mask, conf_sum, conf_count, saber_n):
    """Per-sample AADU token selection.

    Returns selected block-relative indices and the tau used.
    """
    n_masked = int(mask.sum().item())
    if n_masked == 0:
        return torch.empty(0, dtype=torch.long, device=confidence.device), 0.0

    if conf_count > 0:
        tau = (conf_sum / conf_count).item()
    else:
        tau = confidence[mask].max().item()

    sel = torch.where(confidence >= tau)[0]
    if sel.numel() < saber_n:
        k = min(saber_n, n_masked)
        _, sel = torch.topk(confidence, k=k)
    return sel, tau


def _saber_berm_cross_step(
    x_block, full_conf, last_conf, unmask_time_conf,
    conf_sum, conf_count, n_unmask, saber_n, saber_mu, mask_id,
):
    """In-place BERM via cross-step confidence delta (no extra forward).

    Mutates x_block, unmask_time_conf, conf_sum, conf_count.
    """
    BL = x_block.shape[0]
    delta = full_conf - last_conf
    still_masked = (x_block == mask_id)
    delta_rem = delta.clone()
    delta_rem[still_masked] = float('inf')

    n_eligible = int((~still_masked).sum().item())
    mu_t = max(saber_n // 2, (n_unmask + saber_mu - 1) // saber_mu)
    mu_t = min(mu_t, max(0, n_unmask - 1))
    mu_t = min(mu_t, n_eligible)
    if mu_t <= 0:
        return 0, conf_sum, conf_count

    _, rem_idx = torch.topk(delta_rem, k=mu_t, largest=False)
    remasked = 0
    for ri in rem_idx:
        ri_v = ri.item()
        if x_block[ri_v] != mask_id:
            x_block[ri_v] = mask_id
            conf_sum -= unmask_time_conf[ri_v]
            conf_count -= 1
            unmask_time_conf[ri_v] = 0.0
            remasked += 1
    return remasked, conf_sum, conf_count


def _saber_berm_extra_forward(
    x_block, unmask_time_conf, conf_sum, conf_count,
    saber_n, saber_mu, n_unmask, mask_id,
    blk_logits,
):
    """BERM via extra forward: compare unmask-time confidence with
    re-evaluated P(actual_token | new context).

    blk_logits is from a fresh block forward AFTER AADU unmasking.
    Mutates x_block, unmask_time_conf, conf_sum, conf_count.
    """
    p2 = F.softmax(blk_logits.to(torch.float64), dim=-1)
    actual_conf = torch.gather(p2, -1, x_block.unsqueeze(-1)).squeeze(-1)

    still_masked = (x_block == mask_id)
    has_unmask_history = (unmask_time_conf > 0)
    eligible = (~still_masked) & has_unmask_history

    n_eligible = int(eligible.sum().item())
    mu_t = max(saber_n // 2, (n_unmask + saber_mu - 1) // saber_mu)
    mu_t = min(mu_t, max(0, n_unmask - 1))
    mu_t = min(mu_t, n_eligible)
    if mu_t <= 0:
        return 0, conf_sum, conf_count

    delta = unmask_time_conf - actual_conf
    delta[~eligible] = float('-inf')
    _, rem_idx = torch.topk(delta, k=mu_t, largest=True)

    remasked = 0
    for ri in rem_idx:
        ri_v = ri.item()
        if x_block[ri_v] != mask_id and delta[ri_v] > 0:
            x_block[ri_v] = mask_id
            conf_sum -= unmask_time_conf[ri_v]
            conf_count -= 1
            unmask_time_conf[ri_v] = 0.0
            remasked += 1
    return remasked, conf_sum, conf_count


@torch.no_grad()
def generate_with_dual_cache_saber(
    model, prompt, steps=128, gen_length=128, block_length=128,
    temperature=0., remasking='low_confidence', mask_id=126336,
    saber_n=2, saber_mu=8, berm_mode='cross_step', global_aadu=False,
):
    """DualCache generation with Saber AADU + BERM.

    Parameters
    ----------
    saber_n : int
        Min tokens to unmask per step (AADU floor).
    saber_mu : int
        BERM divisor: remask ~ ceil(unmasked_this_step / saber_mu).
    berm_mode : str
        'cross_step'    – compare consecutive steps (no extra forward).
        'extra_forward' – additional block forward for re-evaluation.
    global_aadu : bool
        True  – tau history persists across blocks.
        False – tau resets at each block boundary.
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    seq_len = Lp + gen_length
    device = model.device
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    x = torch.full((B, seq_len), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt
    nfe = 0

    g_conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
    g_conf_count = torch.zeros(B, device=device, dtype=torch.long)
    NEG_INF = torch.tensor(float('-inf'), device=device, dtype=torch.float64)

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length
        BL = e - s

        if global_aadu:
            conf_sum = g_conf_sum.clone()
            conf_count = g_conf_count.clone()
        else:
            conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
            conf_count = torch.zeros(B, device=device, dtype=torch.long)

        last_conf = torch.zeros(B, BL, device=device, dtype=torch.float64)
        unmask_tc = torch.zeros(B, BL, device=device, dtype=torch.float64)

        # ==== Phase 1: warm-up (full forward) ====
        out = model(x, use_cache=True)
        past_kv = out.past_key_values
        nfe += 1
        blk_logits = out.logits[:, s:e, :]
        del out

        x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
        p = F.softmax(blk_logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

        blk_mask = (x[:, s:e] == mask_id)

        for j in range(B):
            conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
            sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], saber_n)
            if sel.numel() == 0:
                continue
            x[j, s + sel] = x0[j, sel]
            sc = x0_p[j, sel]
            conf_sum[j] += sc.sum()
            conf_count[j] += sel.numel()
            unmask_tc[j, sel] = sc

        last_conf = x0_p.clone()

        # ==== Phase 2: refinement (block forward) ====
        rp = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        rp[:, s:e] = True
        max_refine = steps  # safety cap

        for _step in range(max_refine):
            if not (x[:, s:e] == mask_id).any():
                break

            blk_logits = model(
                x[:, s:e], past_key_values=past_kv,
                use_cache=True, replace_position=rp,
            ).logits
            nfe += 1

            x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
            p = F.softmax(blk_logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            blk_mask = (x[:, s:e] == mask_id)
            full_conf = x0_p.clone()

            step_n_unmask = [0] * B  # track per-sample unmask count this step
            for j in range(B):
                conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
                sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], saber_n)
                n_unmask = sel.numel()
                step_n_unmask[j] = n_unmask
                if n_unmask == 0:
                    continue
                x[j, s + sel] = x0[j, sel]
                sc = x0_p[j, sel]
                conf_sum[j] += sc.sum()
                conf_count[j] += sel.numel()
                unmask_tc[j, sel] = sc

                if berm_mode == 'cross_step':
                    _, conf_sum[j], conf_count[j] = _saber_berm_cross_step(
                        x[j, s:e], full_conf[j], last_conf[j], unmask_tc[j],
                        conf_sum[j], conf_count[j], n_unmask, saber_n, saber_mu, mask_id,
                    )

            if berm_mode == 'extra_forward':
                blk_logits2 = model(
                    x[:, s:e], past_key_values=past_kv,
                    use_cache=True, replace_position=rp,
                ).logits
                nfe += 1
                for j in range(B):
                    # use this step's actual unmask count (not accumulated total)
                    _, conf_sum[j], conf_count[j] = _saber_berm_extra_forward(
                        x[j, s:e], unmask_tc[j], conf_sum[j], conf_count[j],
                        saber_n, saber_mu, step_n_unmask[j], mask_id, blk_logits2[j],
                    )

            last_conf = full_conf.clone()

        if global_aadu:
            g_conf_sum = conf_sum.clone()
            g_conf_count = conf_count.clone()

    return x, nfe


@torch.no_grad()
def generate_with_expand_saber(
    model, prompt, steps=128, gen_length=128, block_length=128,
    temperature=0., remasking='low_confidence', mask_id=126336,
    mid_trigger_ratio=0.8,
    saber_n=2, saber_mu=8, berm_mode='cross_step', global_aadu=True,
    post_global_berm_rounds=0, post_global_berm_window_mul=1,
    fullstage_berm=False, fullstage_berm_on_rewarm=True,
    berm_scope='window',
):
    """DualCache + Expand generation with Saber AADU + BERM.

    Expand triggers when remaining masks in the watched block <=
    block_length * mid_trigger_ratio.  On expand a full rewarm is
    performed; AADU selects tokens from the fresh logits but BERM
    is skipped on the rewarm step.

    Parameters
    ----------
    mid_trigger_ratio : float
        Fraction of block decoded before expansion triggers.
    global_aadu : bool
        True  – tau history persists across blocks / expansions.
        False – tau resets at each block boundary / expansion.
    post_global_berm_rounds : int
        Extra global BERM rounds after local decode completes for current window.
    post_global_berm_window_mul : int
        Repair temp-window length multiplier of block_length.
    fullstage_berm : bool
        If True, also apply BERM right after warm-up stage.
    fullstage_berm_on_rewarm : bool
        If True, apply fullstage BERM after expand rewarm as well.
    berm_scope : str
        'window'        – BERM operates on entire expanded window [s, e) (default, legacy).
        'current_block' – BERM only in the latest watching block; no BERM on full-forward.
        'fullforward_berm' – same as current_block, but also allow BERM on full-forward (rewarm).
        'ffberm_v2'     – like fullforward_berm, but uses pre-expand last_conf as baseline
                          (not zeros), so delta comparison is meaningful on rewarm.
        'trail_block'   – BERM on [watching_nb - 1 .. e), i.e. current + one trailing block.
    (Other params same as generate_with_dual_cache_saber.)
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    seq_len = Lp + gen_length
    device = model.device
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    trigger_thresh = int(block_length * mid_trigger_ratio)

    x = torch.full((B, seq_len), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt
    nfe = 0

    g_conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
    g_conf_count = torch.zeros(B, device=device, dtype=torch.long)
    NEG_INF = torch.tensor(float('-inf'), device=device, dtype=torch.float64)
    post_global_berm_rounds = max(0, int(post_global_berm_rounds))
    post_global_berm_window_mul = max(1, int(post_global_berm_window_mul))
    berm_scope = str(berm_scope).strip().lower()

    def _berm_left_offset(s, e, watching_nb):
        """Return the offset (relative to s) below which BERM cannot remask."""
        if berm_scope == 'window':
            return 0
        elif berm_scope in ('current_block', 'fullforward_berm', 'ffberm_v2'):
            return Lp + watching_nb * block_length - s
        elif berm_scope == 'trail_block':
            trail_start = Lp + max(0, watching_nb - 1) * block_length
            return trail_start - s
        return 0

    def _scoped_berm(x_slice_j, full_conf_j, last_conf_j, unmask_tc_j,
                     cs, cc, n_unmask, left_offset):
        """Call _saber_berm_cross_step with position protection."""
        if left_offset <= 0:
            return _saber_berm_cross_step(
                x_slice_j, full_conf_j, last_conf_j, unmask_tc_j,
                cs, cc, n_unmask, saber_n, saber_mu, mask_id,
            )
        saved_utc = unmask_tc_j[:left_offset].clone()
        unmask_tc_j[:left_offset] = 0.0
        saved_x = x_slice_j[:left_offset].clone()
        result = _saber_berm_cross_step(
            x_slice_j, full_conf_j, last_conf_j, unmask_tc_j,
            cs, cc, n_unmask, saber_n, saber_mu, mask_id,
        )
        x_slice_j[:left_offset] = saved_x
        unmask_tc_j[:left_offset] = saved_utc
        return result

    def _apply_fullstage_berm_window(x_slice, conf_slice, last_conf_slice, unmask_tc_slice):
        earliest_local = None
        for j in range(B):
            n_unmask_full = int((x_slice[j] != mask_id).sum().item())
            if n_unmask_full <= 1:
                continue
            before_mask = (x_slice[j] == mask_id).clone()
            _saber_berm_cross_step(
                x_slice[j], conf_slice[j], last_conf_slice[j], unmask_tc_slice[j],
                torch.tensor(0.0, device=device, dtype=torch.float64),
                torch.tensor(0, device=device, dtype=torch.long),
                max(saber_n, n_unmask_full), saber_n, saber_mu, mask_id,
            )
            after_mask = (x_slice[j] == mask_id)
            newly_masked = (after_mask & ~before_mask).nonzero(as_tuple=False).squeeze(-1)
            if newly_masked.numel() > 0:
                _m = int(newly_masked.min().item())
                earliest_local = _m if earliest_local is None else min(earliest_local, _m)
        return earliest_local

    nb = 0
    while nb < num_blocks:
        s = Lp + nb * block_length
        e = s + block_length

        if global_aadu:
            conf_sum = g_conf_sum.clone()
            conf_count = g_conf_count.clone()
        else:
            conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
            conf_count = torch.zeros(B, device=device, dtype=torch.long)

        BL = e - s
        last_conf = torch.zeros(B, BL, device=device, dtype=torch.float64)
        unmask_tc = torch.zeros(B, BL, device=device, dtype=torch.float64)

        # ==== Phase 1: warm-up (full forward) ====
        out = model(x, use_cache=True)
        past_kv = out.past_key_values
        nfe += 1
        blk_logits = out.logits[:, s:e, :]
        del out

        x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
        p = F.softmax(blk_logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        blk_mask = (x[:, s:e] == mask_id)

        for j in range(B):
            conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
            sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], saber_n)
            if sel.numel() == 0:
                continue
            x[j, s + sel] = x0[j, sel]
            sc = x0_p[j, sel]
            conf_sum[j] += sc.sum()
            conf_count[j] += sel.numel()
            unmask_tc[j, sel] = sc

        if fullstage_berm and berm_mode == 'cross_step':
            _ = _apply_fullstage_berm_window(
                x[:, s:e], x0_p, torch.zeros_like(x0_p), unmask_tc
            )

        last_conf = x0_p.clone()

        # ==== Phase 2: refinement with potential expansion ====
        rp = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        rp[:, s:e] = True
        watching_nb = nb
        blocks_consumed = 1

        for _step in range(steps):
            if not (x[:, s:e] == mask_id).any():
                break

            # -- Expand check --
            can_expand = (watching_nb + 1 < num_blocks)
            if can_expand:
                wb_s = Lp + watching_nb * block_length
                wb_e = wb_s + block_length
                remaining = int((x[:, wb_s:wb_e] == mask_id).sum(dim=1).max().item())
                if remaining <= trigger_thresh:
                    next_nb = watching_nb + 1
                    e_new = min(Lp + (next_nb + 1) * block_length, seq_len)
                    e = e_new
                    BL = e - s

                    pre_expand_last_conf = last_conf.clone()
                    pre_expand_unmask_tc = unmask_tc.clone()
                    pre_expand_BL = last_conf.shape[1]

                    if not global_aadu:
                        conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
                        conf_count = torch.zeros(B, device=device, dtype=torch.long)

                    # Rewarm
                    out = model(x, use_cache=True)
                    past_kv = out.past_key_values
                    nfe += 1
                    blk_logits = out.logits[:, s:e, :]
                    del out

                    x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
                    p = F.softmax(blk_logits.to(torch.float64), dim=-1)
                    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    blk_mask = (x[:, s:e] == mask_id)
                    unmask_tc_rewarm = torch.zeros(B, BL, device=device, dtype=torch.float64)

                    for j in range(B):
                        conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
                        sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], saber_n)
                        if sel.numel() == 0:
                            continue
                        x[j, s + sel] = x0[j, sel]
                        sc = x0_p[j, sel]
                        conf_sum[j] += sc.sum()
                        conf_count[j] += sel.numel()
                        unmask_tc_rewarm[j, sel] = sc

                    if fullstage_berm and fullstage_berm_on_rewarm and berm_mode == 'cross_step':
                        _ = _apply_fullstage_berm_window(
                            x[:, s:e], x0_p, torch.zeros_like(x0_p), unmask_tc_rewarm
                        )

                    if berm_scope == 'fullforward_berm' and berm_mode == 'cross_step':
                        for j in range(B):
                            n_unm = int((x[j, s:e] != mask_id).sum().item())
                            if n_unm <= 1:
                                continue
                            _saber_berm_cross_step(
                                x[j, s:e], x0_p[j], torch.zeros_like(x0_p[j]), unmask_tc_rewarm[j],
                                conf_sum[j], conf_count[j], max(saber_n, n_unm),
                                saber_n, saber_mu, mask_id,
                            )

                    if berm_scope == 'ffberm_v2' and berm_mode == 'cross_step':
                        rewarm_last_conf = torch.zeros(B, BL, device=device, dtype=torch.float64)
                        rewarm_last_conf[:, :pre_expand_BL] = pre_expand_last_conf
                        rewarm_last_conf[:, pre_expand_BL:] = x0_p[:, pre_expand_BL:]
                        rewarm_utc = torch.zeros(B, BL, device=device, dtype=torch.float64)
                        rewarm_utc[:, :pre_expand_BL] = pre_expand_unmask_tc
                        for j in range(B):
                            new_unm = unmask_tc_rewarm[j]
                            rewarm_utc[j] = torch.where(new_unm > 0, new_unm, rewarm_utc[j])

                        before_berm = (x[:, s:e] == mask_id).clone()
                        for j in range(B):
                            n_unm = int((x[j, s:e] != mask_id).sum().item())
                            if n_unm <= 1:
                                continue
                            _saber_berm_cross_step(
                                x[j, s:e], x0_p[j], rewarm_last_conf[j], rewarm_utc[j],
                                conf_sum[j], conf_count[j], max(saber_n, n_unm),
                                saber_n, saber_mu, mask_id,
                            )
                        after_berm = (x[:, s:e] == mask_id)

                        # Temp-window repair: fix remasked tokens with a small 2-block window
                        earliest_remask_offset = None
                        for j in range(B):
                            newly = (after_berm[j] & ~before_berm[j]).nonzero(as_tuple=False).squeeze(-1)
                            if newly.numel() > 0:
                                _m = int(newly.min().item())
                                earliest_remask_offset = _m if earliest_remask_offset is None else min(earliest_remask_offset, _m)

                        if earliest_remask_offset is not None:
                            repair_len = 2 * block_length
                            repair_abs_start = s + earliest_remask_offset
                            wb_abs_start = Lp + next_nb * block_length
                            repair_abs_end = min(wb_abs_start, seq_len)

                            if repair_abs_start < repair_abs_end:
                                out_repair = model(x, use_cache=True)
                                past_kv_repair = out_repair.past_key_values
                                nfe += 1
                                del out_repair

                                for rs in range(repair_abs_start, repair_abs_end, repair_len):
                                    re_ = min(rs + repair_len, repair_abs_end)
                                    rp_repair = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                                    rp_repair[:, rs:re_] = True
                                    rep_logits = model(
                                        x[:, rs:re_], past_key_values=past_kv_repair,
                                        use_cache=True, replace_position=rp_repair,
                                    ).logits
                                    nfe += 1
                                    rep_x0 = torch.argmax(add_gumbel_noise(rep_logits, temperature), dim=-1)
                                    rep_p = F.softmax(rep_logits.to(torch.float64), dim=-1)
                                    rep_x0_p = torch.gather(rep_p, -1, rep_x0.unsqueeze(-1)).squeeze(-1)
                                    rep_mask = (x[:, rs:re_] == mask_id)
                                    for j in range(B):
                                        cj = torch.where(rep_mask[j], rep_x0_p[j], NEG_INF)
                                        sel_r, _ = _saber_select(cj, rep_mask[j], conf_sum[j], conf_count[j], saber_n)
                                        if sel_r.numel() > 0:
                                            x[j, rs + sel_r] = rep_x0[j, sel_r]
                                            sc_r = rep_x0_p[j, sel_r]
                                            conf_sum[j] += sc_r.sum()
                                            conf_count[j] += sel_r.numel()

                    # Reset BERM state for expanded window
                    last_conf = torch.zeros(B, BL, device=device, dtype=torch.float64)
                    unmask_tc = torch.zeros(B, BL, device=device, dtype=torch.float64)
                    last_conf[:, :x0_p.shape[1]] = x0_p
                    for j in range(B):
                        unmasked_j = (x[j, s:e] != mask_id)
                        unmask_tc[j, :unmasked_j.shape[0]][unmasked_j] = x0_p[j][unmasked_j]

                    rp = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                    rp[:, s:e] = True
                    blocks_consumed += 1
                    watching_nb = next_nb
                    continue  # skip to next iteration (no BERM on rewarm)

            # -- Normal refinement step --
            blk_logits = model(
                x[:, s:e], past_key_values=past_kv,
                use_cache=True, replace_position=rp,
            ).logits
            nfe += 1

            x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
            p = F.softmax(blk_logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            blk_mask = (x[:, s:e] == mask_id)
            full_conf = x0_p.clone()

            step_n_unmask = [0] * B  # track per-sample unmask count this step
            berm_left_off = max(0, _berm_left_offset(s, e, watching_nb))
            for j in range(B):
                conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
                sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], saber_n)
                n_unmask = sel.numel()
                step_n_unmask[j] = n_unmask
                if n_unmask == 0:
                    continue
                x[j, s + sel] = x0[j, sel]
                sc = x0_p[j, sel]
                conf_sum[j] += sc.sum()
                conf_count[j] += sel.numel()
                unmask_tc[j, sel] = sc

                if berm_mode == 'cross_step':
                    _, conf_sum[j], conf_count[j] = _scoped_berm(
                        x[j, s:e], full_conf[j], last_conf[j], unmask_tc[j],
                        conf_sum[j], conf_count[j], n_unmask, berm_left_off,
                    )

            if berm_mode == 'extra_forward':
                blk_logits2 = model(
                    x[:, s:e], past_key_values=past_kv,
                    use_cache=True, replace_position=rp,
                ).logits
                nfe += 1
                for j in range(B):
                    _, conf_sum[j], conf_count[j] = _saber_berm_extra_forward(
                        x[j, s:e], unmask_tc[j], conf_sum[j], conf_count[j],
                        saber_n, saber_mu, step_n_unmask[j], mask_id, blk_logits2[j],
                    )

            last_conf = full_conf.clone()

        # ==== Phase 3: optional post-global BERM rounds ====
        # Trigger only when current window has no masks left.
        if post_global_berm_rounds > 0 and berm_mode == 'cross_step' and not (x[:, s:e] == mask_id).any():
            gs = Lp
            ge = e
            temp_len = max(block_length, block_length * post_global_berm_window_mul)
            zero_f = torch.tensor(0.0, device=device, dtype=torch.float64)
            zero_i = torch.tensor(0, device=device, dtype=torch.long)

            for _r in range(post_global_berm_rounds):
                # Global confidence pass
                out_g = model(x, use_cache=True)
                nfe += 1
                g_logits = out_g.logits[:, gs:ge, :]
                del out_g

                g_x0 = torch.argmax(add_gumbel_noise(g_logits, temperature), dim=-1)
                g_p = F.softmax(g_logits.to(torch.float64), dim=-1)
                g_x0_p = torch.gather(g_p, -1, g_x0.unsqueeze(-1)).squeeze(-1)
                g_mask = (x[:, gs:ge] == mask_id)

                unmask_tc_g = torch.zeros(B, ge - gs, device=device, dtype=torch.float64)
                for j in range(B):
                    unmasked_j = ~g_mask[j]
                    unmask_tc_g[j, unmasked_j] = g_x0_p[j, unmasked_j]

                earliest_local = None
                for j in range(B):
                    n_unmask_full = int((~g_mask[j]).sum().item())
                    if n_unmask_full <= 1:
                        continue
                    before_mask = (x[j, gs:ge] == mask_id).clone()
                    _saber_berm_cross_step(
                        x[j, gs:ge], g_x0_p[j], torch.zeros_like(g_x0_p[j]), unmask_tc_g[j],
                        zero_f, zero_i, max(saber_n, n_unmask_full), saber_n, saber_mu, mask_id,
                    )
                    after_mask = (x[j, gs:ge] == mask_id)
                    newly_masked = (after_mask & ~before_mask).nonzero(as_tuple=False).squeeze(-1)
                    if newly_masked.numel() > 0:
                        _m = int(newly_masked.min().item())
                        earliest_local = _m if earliest_local is None else min(earliest_local, _m)

                if earliest_local is None:
                    break

                # Temp-window repair from earliest remasked position until ge
                start_abs = gs + earliest_local
                for rs in range(start_abs, ge, temp_len):
                    re = min(rs + temp_len, ge)
                    out_r = model(x, use_cache=True)
                    past_kv_r = out_r.past_key_values
                    nfe += 1
                    del out_r

                    rp_tmp = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                    rp_tmp[:, rs:re] = True
                    tmp_logits = model(
                        x[:, rs:re], past_key_values=past_kv_r,
                        use_cache=True, replace_position=rp_tmp,
                    ).logits
                    nfe += 1
                    tmp_x0 = torch.argmax(add_gumbel_noise(tmp_logits, temperature), dim=-1)
                    tmp_p = F.softmax(tmp_logits.to(torch.float64), dim=-1)
                    tmp_x0_p = torch.gather(tmp_p, -1, tmp_x0.unsqueeze(-1)).squeeze(-1)
                    tmp_mask = (x[:, rs:re] == mask_id)

                    for j in range(B):
                        conf_j = torch.where(tmp_mask[j], tmp_x0_p[j], NEG_INF)
                        sel, _ = _saber_select(conf_j, tmp_mask[j], zero_f, zero_i, saber_n)
                        if sel.numel() == 0:
                            continue
                        x[j, rs + sel] = tmp_x0[j, sel]

                # One final global refresh without BERM after reaching current window.
                _ = model(x, use_cache=True)
                nfe += 1

        if global_aadu:
            g_conf_sum = conf_sum.clone()
            g_conf_count = conf_count.clone()
        nb += blocks_consumed

    return x, nfe


@torch.no_grad()
def generate_with_expand_saber_dynamic(
    model, prompt, steps=128, gen_length=128, block_length=128,
    temperature=0., remasking='low_confidence', mask_id=126336,
    mid_trigger_ratio=0.8, global_aadu=True, berm_mode='cross_step',
    n_hi=8, n_lo=3, mu_lo=4, mu_hi=16,
    gamma_n=1.0, gamma_mu=1.0, gamma_floor=1.5,
    remask_ratio_hi=0.35, floor_lo=0,
    berm_scope='trail_block',
    eos_penalty=0.0, eos_token_id=None,
):
    """Expand-Saber with dynamic n / mu / remask-floor that adapt to decode progress.

    Early decoding: large n (write fast), small mu (remask aggressively), high floor.
    Late  decoding: small n (write carefully), large mu (remask rarely), low floor.

    Progress metric: m = n_masked_in_window / window_size  (1.0=all mask, 0.0=done).

    EOS suppression (eos_penalty > 0):
        Penalise EOS logit proportionally to remaining mask ratio.
        Formula: logits[..., eos_token_id] += eos_penalty * log(1 - m + eps)
        When m≈1 (early), penalty is large negative; when m≈0 (late), penalty→0.
    """
    import math

    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    seq_len = Lp + gen_length
    device = model.device
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    trigger_thresh = int(block_length * mid_trigger_ratio)
    NEG_INF = torch.tensor(float('-inf'), device=device, dtype=torch.float64)

    x = torch.full((B, seq_len), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt
    nfe = 0

    g_conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
    g_conf_count = torch.zeros(B, device=device, dtype=torch.long)

    def _progress(s, e):
        """Return mask ratio m in [0, 1] for current window."""
        wlen = e - s
        if wlen == 0:
            return 0.0
        return float((x[:, s:e] == mask_id).sum().item()) / (B * wlen)

    _eos_eps = 1e-3

    def _apply_eos_bias(logits, m):
        """Suppress EOS token logit when many masks remain (early decoding).

        m is mask ratio: 1.0 = all masked (heavy penalty), 0.0 = done (no penalty).
        Mirrors Dream's formula: logits[eos] += penalty * log(1 - m + eps).
        """
        if eos_penalty == 0.0 or eos_token_id is None:
            return logits
        bias = eos_penalty * math.log(max(1.0 - m + _eos_eps, _eos_eps))
        logits = logits.clone()
        logits[..., eos_token_id] += bias
        return logits

    def _dynamic_n(m):
        return max(1, round(n_lo + (n_hi - n_lo) * (m ** gamma_n)))

    def _dynamic_mu(m):
        return max(1, round(mu_lo + (mu_hi - mu_lo) * ((1 - m) ** gamma_mu)))

    def _dynamic_floor(m, cur_n):
        return max(floor_lo, round(remask_ratio_hi * cur_n * (m ** gamma_floor)))

    def _dynamic_berm(x_block, full_conf, last_conf, unmask_tc, cs, cc, n_unmask, cur_mu, cur_floor):
        """BERM with dynamic mu and floor."""
        still_masked = (x_block == mask_id)
        delta = full_conf - last_conf
        delta_rem = delta.clone()
        delta_rem[still_masked] = float('inf')

        n_eligible = int((~still_masked).sum().item())
        mu_t = max(cur_floor, math.ceil(n_unmask / cur_mu))
        mu_t = min(mu_t, max(0, n_unmask - 1))
        mu_t = min(mu_t, n_eligible)
        if mu_t <= 0:
            return 0, cs, cc

        _, rem_idx = torch.topk(delta_rem, k=mu_t, largest=False)
        remasked = 0
        for ri in rem_idx:
            ri_v = ri.item()
            if x_block[ri_v] != mask_id:
                x_block[ri_v] = mask_id
                cs -= unmask_tc[ri_v]
                cc -= 1
                unmask_tc[ri_v] = 0.0
                remasked += 1
        return remasked, cs, cc

    def _berm_left_offset_dyn(s, e, watching_nb):
        if berm_scope == 'window':
            return 0
        elif berm_scope == 'current_block':
            return Lp + watching_nb * block_length - s
        elif berm_scope == 'trail_block':
            return Lp + max(0, watching_nb - 1) * block_length - s
        return 0

    def _scoped_dynamic_berm(x_slice_j, full_conf_j, last_conf_j, unmask_tc_j,
                              cs, cc, n_unmask, cur_mu, cur_floor, left_offset):
        if left_offset <= 0:
            return _dynamic_berm(x_slice_j, full_conf_j, last_conf_j, unmask_tc_j,
                                 cs, cc, n_unmask, cur_mu, cur_floor)
        saved_utc = unmask_tc_j[:left_offset].clone()
        unmask_tc_j[:left_offset] = 0.0
        saved_x = x_slice_j[:left_offset].clone()
        result = _dynamic_berm(x_slice_j, full_conf_j, last_conf_j, unmask_tc_j,
                               cs, cc, n_unmask, cur_mu, cur_floor)
        x_slice_j[:left_offset] = saved_x
        unmask_tc_j[:left_offset] = saved_utc
        return result

    nb = 0
    while nb < num_blocks:
        s = Lp + nb * block_length
        e = s + block_length

        if global_aadu:
            conf_sum = g_conf_sum.clone()
            conf_count = g_conf_count.clone()
        else:
            conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
            conf_count = torch.zeros(B, device=device, dtype=torch.long)

        BL = e - s
        last_conf = torch.zeros(B, BL, device=device, dtype=torch.float64)
        unmask_tc = torch.zeros(B, BL, device=device, dtype=torch.float64)

        m = _progress(s, e)
        cur_n = _dynamic_n(m)

        out = model(x, use_cache=True)
        past_kv = out.past_key_values
        nfe += 1
        blk_logits = _apply_eos_bias(out.logits[:, s:e, :], m)
        del out

        x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
        p = F.softmax(blk_logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        blk_mask = (x[:, s:e] == mask_id)

        for j in range(B):
            conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
            sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], cur_n)
            if sel.numel() == 0:
                continue
            x[j, s + sel] = x0[j, sel]
            sc = x0_p[j, sel]
            conf_sum[j] += sc.sum()
            conf_count[j] += sel.numel()
            unmask_tc[j, sel] = sc

        last_conf = x0_p.clone()

        rp = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        rp[:, s:e] = True
        watching_nb = nb
        blocks_consumed = 1

        for _step in range(steps):
            if not (x[:, s:e] == mask_id).any():
                break

            m = _progress(s, e)
            cur_n = _dynamic_n(m)
            cur_mu = _dynamic_mu(m)
            cur_floor = _dynamic_floor(m, cur_n)

            can_expand = (watching_nb + 1 < num_blocks)
            if can_expand:
                wb_s = Lp + watching_nb * block_length
                wb_e = wb_s + block_length
                remaining = int((x[:, wb_s:wb_e] == mask_id).sum(dim=1).max().item())
                if remaining <= trigger_thresh:
                    next_nb = watching_nb + 1
                    e_new = min(Lp + (next_nb + 1) * block_length, seq_len)
                    e = e_new
                    BL = e - s

                    if not global_aadu:
                        conf_sum = torch.zeros(B, device=device, dtype=torch.float64)
                        conf_count = torch.zeros(B, device=device, dtype=torch.long)

                    out = model(x, use_cache=True)
                    past_kv = out.past_key_values
                    nfe += 1
                    m = _progress(s, e)
                    blk_logits = _apply_eos_bias(out.logits[:, s:e, :], m)
                    del out

                    cur_n = _dynamic_n(m)

                    x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
                    p = F.softmax(blk_logits.to(torch.float64), dim=-1)
                    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    blk_mask = (x[:, s:e] == mask_id)

                    for j in range(B):
                        conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
                        sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], cur_n)
                        if sel.numel() == 0:
                            continue
                        x[j, s + sel] = x0[j, sel]
                        sc = x0_p[j, sel]
                        conf_sum[j] += sc.sum()
                        conf_count[j] += sel.numel()

                    last_conf = torch.zeros(B, BL, device=device, dtype=torch.float64)
                    unmask_tc = torch.zeros(B, BL, device=device, dtype=torch.float64)
                    last_conf[:, :x0_p.shape[1]] = x0_p
                    for j in range(B):
                        unmasked_j = (x[j, s:e] != mask_id)
                        unmask_tc[j, :unmasked_j.shape[0]][unmasked_j] = x0_p[j][unmasked_j]

                    rp = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                    rp[:, s:e] = True
                    blocks_consumed += 1
                    watching_nb = next_nb
                    continue

            blk_logits = _apply_eos_bias(model(
                x[:, s:e], past_key_values=past_kv,
                use_cache=True, replace_position=rp,
            ).logits, m)
            nfe += 1

            x0 = torch.argmax(add_gumbel_noise(blk_logits, temperature), dim=-1)
            p = F.softmax(blk_logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            blk_mask = (x[:, s:e] == mask_id)
            full_conf = x0_p.clone()

            left_off = max(0, _berm_left_offset_dyn(s, e, watching_nb))
            for j in range(B):
                conf_j = torch.where(blk_mask[j], x0_p[j], NEG_INF)
                sel, _ = _saber_select(conf_j, blk_mask[j], conf_sum[j], conf_count[j], cur_n)
                n_unmask = sel.numel()
                if n_unmask == 0:
                    continue
                x[j, s + sel] = x0[j, sel]
                sc = x0_p[j, sel]
                conf_sum[j] += sc.sum()
                conf_count[j] += sel.numel()
                unmask_tc[j, sel] = sc

                if berm_mode == 'cross_step':
                    _, conf_sum[j], conf_count[j] = _scoped_dynamic_berm(
                        x[j, s:e], full_conf[j], last_conf[j], unmask_tc[j],
                        conf_sum[j], conf_count[j], n_unmask, cur_mu, cur_floor, left_off,
                    )

            last_conf = full_conf.clone()

        if global_aadu:
            g_conf_sum = conf_sum.clone()
            g_conf_count = conf_count.clone()
        nb += blocks_consumed

    return x, nfe


def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
