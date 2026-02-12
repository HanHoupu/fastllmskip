"""
Multipass generation variants for DualCache.
All functions are self-contained — do NOT modify generate.py or modeling_llada.py.

Methods:
  1. generate_multipass       — Plan A: 2-pass (no guarantee → with guarantee)
  2. generate_adaptive        — Adaptive: multi-pass, threshold auto-lowers per pass
  3. generate_factor_multipass — Factor: multi-pass with factor strategy, no per-step guarantee
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index,
    get_transfer_index_dynamic,
)


# ============================================================
# Helper: threshold transfer WITHOUT force_mask guarantee
# ============================================================

def get_transfer_index_no_guarantee(logits, temperature, remasking, mask_index, x, threshold):
    """get_transfer_index (threshold branch) WITHOUT the force_mask guarantee."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)
    transfer_index = mask_index & (confidence >= threshold)
    return x0, transfer_index


# ============================================================
# Helper: factor transfer WITHOUT threshs[0]=-1 guarantee
# ============================================================

def get_transfer_index_dynamic_no_guarantee(logits, temperature, remasking,
                                            mask_index, x, num_transfer_tokens, factor=1):
    """get_transfer_index_dynamic WITHOUT the threshs[0]=-1 guarantee."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
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

        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]
        # NO guarantee: do NOT set threshs[0] = -1

        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        assert len(sorted_confidence) == len(threshs)

        top_i = 0
        broke = False
        for check_i in range(len(threshs)):
            if sorted_confidence[check_i] < threshs[check_i]:
                top_i = check_i
                broke = True
                break
        if not broke:
            top_i = len(threshs)  # all passed

        if top_i == 0:
            continue  # nothing passed → skip

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


# ============================================================
# Shared: single-pass block scanner
# ============================================================

def _scan_blocks(model, x, Lp, num_blocks, block_length, steps_per_block,
                 temperature, remasking, mask_id,
                 transfer_fn, transfer_fn_blk):
    """
    Scan all blocks with given transfer functions. Returns updated x, nfe, unmasked_count.
    transfer_fn:     called with (logits, temp, remasking, mask_index, x) for step 0
    transfer_fn_blk: called with (logits, temp, remasking, mask_index, x_blk) for refinement
    """
    nfe = 0
    unmasked = 0
    seq_len = x.shape[1]

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length
        if (x[:, s:e] == mask_id).sum() == 0:
            continue

        block_mask = (x[:, s:e] == mask_id)
        num_tt = get_num_transfer_tokens(block_mask, steps_per_block)

        out = model(x, use_cache=True)
        past_kv = out.past_key_values
        nfe += 1

        rp = torch.zeros_like(x, dtype=torch.bool)
        rp[:, s:e] = True

        gmi = (x == mask_id)
        gmi[:, e:] = False

        x0, ti = transfer_fn(out.logits, temperature, remasking, gmi, x)
        t = int(ti.sum().item())
        if t == 0:
            continue
        unmasked += t
        x = torch.where(ti, x0, x)

        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            lb = model(x[:, s:e], past_key_values=past_kv,
                       use_cache=True, replace_position=rp).logits
            nfe += 1

            mb = (x[:, s:e] == mask_id)
            x0b, tib = transfer_fn_blk(lb, temperature, remasking, mb, x[:, s:e])
            t = int(tib.sum().item())
            if t == 0:
                break
            unmasked += t
            bn = torch.where(tib, x0b, x[:, s:e])
            x = torch.cat([x[:, :s], bn, x[:, e:]], dim=1)

    return x, nfe, unmasked


# ============================================================
# Method 1: Plan A — 2-pass (no guarantee → with guarantee)
# ============================================================

@torch.no_grad()
def generate_multipass(model, prompt, steps=256, gen_length=256, block_length=32,
                       temperature=0., remasking="low_confidence", mask_id=126336,
                       threshold=0.9):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    nfe = 0

    # Pass 1: no guarantee
    fn = lambda lg, t, r, mi, xi: get_transfer_index_no_guarantee(lg, t, r, mi, xi, threshold)
    x, nfe1, um1 = _scan_blocks(model, x, Lp, num_blocks, block_length, steps_per_block,
                                 temperature, remasking, mask_id, fn, fn)
    nfe += nfe1
    remaining = int((x[:, Lp:] == mask_id).sum().item())

    # Pass 2: with guarantee
    nfe2 = 0
    um2 = 0
    if remaining > 0:
        fn_g = lambda lg, t, r, mi, xi: get_transfer_index(lg, t, r, mi, xi, None, threshold)
        x, nfe2, um2 = _scan_blocks(model, x, Lp, num_blocks, block_length, steps_per_block,
                                     temperature, remasking, mask_id, fn_g, fn_g)
        nfe += nfe2

    stats = {
        'total_nfe': nfe, 'pass1_nfe': nfe1, 'pass2_nfe': nfe2,
        'pass1_unmasked': um1, 'pass2_unmasked': um2,
        'remaining_after_pass1': remaining,
    }
    return x, nfe, stats


# ============================================================
# Method 2: Adaptive — multi-pass, threshold auto-lowers
# ============================================================

@torch.no_grad()
def generate_adaptive(model, prompt, steps=256, gen_length=256, block_length=32,
                      temperature=0., remasking="low_confidence", mask_id=126336,
                      initial_threshold=0.9, target_ratio=0.25, max_passes=10):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    nfe = 0
    pass_log = []

    for pass_num in range(max_passes):
        remaining_before = int((x[:, Lp:] == mask_id).sum().item())
        if remaining_before == 0:
            break

        # Determine threshold
        if pass_num == 0:
            threshold = initial_threshold
        else:
            out_probe = model(x)
            nfe += 1
            probs = F.softmax(out_probe.logits.to(torch.float64), dim=-1)
            x0_probe = torch.argmax(out_probe.logits, dim=-1)
            x0_p = torch.gather(probs, dim=-1, index=x0_probe.unsqueeze(-1)).squeeze(-1)
            masked_confs = x0_p[x == mask_id]
            if len(masked_confs) == 0:
                break
            threshold = float(torch.quantile(masked_confs.float(),
                                             1.0 - target_ratio).item())
            threshold = max(threshold, 0.0)

        use_guarantee = (threshold <= 0.0)
        if use_guarantee:
            fn = lambda lg, t, r, mi, xi: get_transfer_index(lg, t, r, mi, xi, None, threshold)
        else:
            fn = lambda lg, t, r, mi, xi, _th=threshold: get_transfer_index_no_guarantee(
                lg, t, r, mi, xi, _th)

        x, pass_nfe, pass_um = _scan_blocks(
            model, x, Lp, num_blocks, block_length, steps_per_block,
            temperature, remasking, mask_id, fn, fn)
        nfe += pass_nfe

        remaining_after = int((x[:, Lp:] == mask_id).sum().item())
        eps_ub = round(pass_um * (1 - threshold), 4)

        pass_log.append({
            'pass': pass_num, 'threshold': round(threshold, 4),
            'remaining_before': remaining_before, 'remaining_after': remaining_after,
            'unmasked': pass_um, 'nfe': pass_nfe, 'epsilon_ub': eps_ub,
            'use_guarantee': use_guarantee,
        })

    cum = 0
    for p in pass_log:
        cum += p['epsilon_ub']
        p['cum_epsilon'] = round(cum, 4)

    stats = {'total_nfe': nfe, 'passes': pass_log, 'num_passes': len(pass_log)}
    return x, nfe, stats


# ============================================================
# Method 3: Factor multipass — factor strategy, no per-step guarantee
# ============================================================

@torch.no_grad()
def generate_factor_multipass(model, prompt, steps=256, gen_length=256, block_length=32,
                              temperature=0., remasking="low_confidence", mask_id=126336,
                              factor=1.0, max_passes=10):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    nfe = 0
    pass_log = []

    for pass_num in range(max_passes):
        remaining_before = int((x[:, Lp:] == mask_id).sum().item())
        if remaining_before == 0:
            break

        # Enable original guarantee (with threshs[0]=-1) ONLY if last pass made 0 progress
        use_guarantee = (pass_num > 0 and len(pass_log) > 0
                         and pass_log[-1]['unmasked'] == 0)

        if use_guarantee:
            fn = lambda lg, t, r, mi, xi: get_transfer_index_dynamic(
                lg, t, r, mi, xi, None, factor)
        else:
            fn = lambda lg, t, r, mi, xi: get_transfer_index_dynamic_no_guarantee(
                lg, t, r, mi, xi, None, factor)

        x, pass_nfe, pass_um = _scan_blocks(
            model, x, Lp, num_blocks, block_length, steps_per_block,
            temperature, remasking, mask_id, fn, fn)
        nfe += pass_nfe

        remaining_after = int((x[:, Lp:] == mask_id).sum().item())

        pass_log.append({
            'pass': pass_num, 'factor': factor,
            'remaining_before': remaining_before, 'remaining_after': remaining_after,
            'unmasked': pass_um, 'nfe': pass_nfe,
            'use_guarantee': use_guarantee,
        })

    stats = {'total_nfe': nfe, 'passes': pass_log, 'num_passes': len(pass_log)}
    return x, nfe, stats
