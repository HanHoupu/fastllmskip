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
    """
    Gumbel-max 用于从分类分布中采样。
    根据 arXiv:2409.02908，对于 MDM，低精度 Gumbel Max 可改善困惑度但降低生成质量。
    因此使用 float64。
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


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



@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, nfe



def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
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

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

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

# =============================================================================
# Layer-Level Skip: 基于 dual_cache，真正跳过层计算
# =============================================================================

@torch.no_grad()
def generate_with_layer_skip(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    layer_skip_threshold=0.99,
):
    """
    基于 dual_cache 的 layer skip 版本。
    
    Layer Skip 逻辑：
    - 如果前一层的 input-output 相似度 > layer_skip_threshold，跳过当前层
    - 跳过时 output = input (identity mapping)
    - 上一步跳过的层，这一步必须重算
    
    Args:
        layer_skip_threshold: 触发跳过的相似度阈值 (default: 0.99)
                              设为 1.0 时退化成普通 dual_cache 版本
    
    Returns:
        (x, nfe, skip_stats): 生成结果、前向次数、跳过统计
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    n_layers = model.model.config.n_layers
    
    # Layer skip 状态
    last_step_io_sim = {}      # 上一步每层的 input-output 相似度
    last_step_skipped = set()  # 上一步跳过的层
    
    # 统计
    total_layers_computed = 0
    total_layers_skipped = 0

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # 每个 block 开始时重置状态
        last_step_io_sim.clear()
        last_step_skipped.clear()

        # 1) Warm KV-cache（第一次 forward，不跳过任何层，收集相似度）
        out_full = model(x, use_cache=True, output_hidden_states=True)
        past_key_values = out_full.past_key_values
        hidden_states = out_full.hidden_states  # tuple of (n_layers+1) tensors
        nfe += 1
        
        # 计算每层的 input-output 相似度（用于下一步决策）
        # hidden_states[i] 是 layer i 的输入，hidden_states[i+1] 是 layer i 的输出
        for layer_idx in range(n_layers):
            inp = hidden_states[layer_idx]
            out = hidden_states[layer_idx + 1]
            # 计算余弦相似度
            flat_inp = inp.reshape(-1).float()
            flat_out = out.reshape(-1).float()
            sim = F.cosine_similarity(flat_inp.unsqueeze(0), flat_out.unsqueeze(0)).item()
            last_step_io_sim[layer_idx] = sim
        
        total_layers_computed += n_layers

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement with layer skip
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            
            # 根据上一步的相似度决定跳过哪些层
            # layer 0 不跳过（第一层），后8层不跳过（保证输出质量）
            skip_layers = set()
            for layer_idx in range(1, n_layers - 8):  # layer 1 到 n_layers-9
                prev_layer = layer_idx - 1
                # 条件：前一层相似度高 且 上一步没跳过当前层
                if (prev_layer in last_step_io_sim and 
                    last_step_io_sim[prev_layer] > layer_skip_threshold and
                    layer_idx not in last_step_skipped):
                    skip_layers.add(layer_idx)
            
            # Forward with skip_layers
            out_blk = model(
                x[:, s:e], 
                past_key_values=past_key_values, 
                use_cache=True, 
                replace_position=replace_position,
                output_hidden_states=True,
                skip_layers=skip_layers,
            )
            logits_blk = out_blk.logits
            hidden_states = out_blk.hidden_states
            
            # 更新统计
            total_layers_computed += (n_layers - len(skip_layers))
            total_layers_skipped += len(skip_layers)
            
            # 更新 last_step 状态
            last_step_skipped = skip_layers.copy()
            last_step_io_sim.clear()
            for layer_idx in range(n_layers):
                if layer_idx not in skip_layers:
                    inp = hidden_states[layer_idx]
                    out = hidden_states[layer_idx + 1]
                    flat_inp = inp.reshape(-1).float()
                    flat_out = out.reshape(-1).float()
                    sim = F.cosine_similarity(flat_inp.unsqueeze(0), flat_out.unsqueeze(0)).item()
                    last_step_io_sim[layer_idx] = sim

            mask_blk = (x[:, s:e] == mask_id)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

            nfe += 1

    total = total_layers_computed + total_layers_skipped
    skip_stats = {
        'total_layers_computed': total_layers_computed,
        'total_layers_skipped': total_layers_skipped,
        'skip_rate': total_layers_skipped / total if total > 0 else 0,
    }

    return x, nfe, skip_stats


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

# =============================================================================
# Layer-Level Analysis: Hook-based generation for studying inter-layer similarity
# =============================================================================

class LayerHiddenStatesCollector:
    """
    收集每一层 Transformer Block 的输入和输出 hidden states。
    
    用途：分析相邻层之间的余弦相似度，为 layer-level skip 策略提供依据。
    
    Hook 机制：
    - 在每个 LLaDALlamaBlock 的 forward 前后注册 hook
    - 收集输入 x 和输出 (output, cache) 中的 output
    """
    
    def __init__(self):
        self.layer_inputs = {}   # {layer_idx: tensor}
        self.layer_outputs = {}  # {layer_idx: tensor}
        self.handles = []
        self._enabled = True
    
    def clear(self):
        """清空收集的数据，保留 hook"""
        self.layer_inputs.clear()
        self.layer_outputs.clear()
    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False
    
    def _make_pre_hook(self, layer_idx):
        """创建 forward pre-hook，收集层输入"""
        def hook(module, args):
            if not self._enabled:
                return
            # args[0] is x (hidden states input)
            x = args[0]
            # 保存 detach 后的 clone，避免影响计算图
            self.layer_inputs[layer_idx] = x.detach().clone()
        return hook
    
    def _make_post_hook(self, layer_idx):
        """创建 forward hook，收集层输出"""
        def hook(module, args, output):
            if not self._enabled:
                return
            # output is (hidden_states, cache)
            hidden_states = output[0]
            self.layer_outputs[layer_idx] = hidden_states.detach().clone()
        return hook
    
    def register_hooks(self, model):
        """
        在模型的所有 Transformer Block 上注册 hook。
        
        Args:
            model: LLaDAModelLM 实例
        """
        self.remove_hooks()  # 先移除旧的 hook
        
        # 获取 transformer blocks
        llada_model = model.model  # LLaDAModel
        if hasattr(llada_model.transformer, 'blocks'):
            blocks = llada_model.transformer.blocks
        else:
            # 如果使用 block_groups，需要展开
            blocks = []
            for group in llada_model.transformer.block_groups:
                blocks.extend(list(group))
        
        for layer_idx, block in enumerate(blocks):
            # 注册 pre-forward hook（收集输入）
            handle_pre = block.register_forward_pre_hook(self._make_pre_hook(layer_idx))
            self.handles.append(handle_pre)
            
            # 注册 forward hook（收集输出）
            handle_post = block.register_forward_hook(self._make_post_hook(layer_idx))
            self.handles.append(handle_post)
        
        return len(blocks)
    
    def remove_hooks(self):
        """移除所有注册的 hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def get_layer_hidden_states(self):
        """
        获取收集的 hidden states。
        
        Returns:
            dict: {
                'inputs': {layer_idx: tensor},
                'outputs': {layer_idx: tensor}
            }
        """
        return {
            'inputs': dict(self.layer_inputs),
            'outputs': dict(self.layer_outputs)
        }
    
    def compute_layer_similarities(self, token_indices=None):
        """
        计算相邻层输出之间的余弦相似度。
        
        Args:
            token_indices: 可选，指定要分析的 token 位置（列表或 None 表示全部）
        
        Returns:
            dict: {
                'adjacent_similarities': [(layer_i, layer_i+1, similarity), ...],
                'per_token_similarities': {layer_pair: [sim_per_token]},  # 如果提供了 token_indices
            }
        """
        n_layers = len(self.layer_outputs)
        if n_layers < 2:
            return {'adjacent_similarities': [], 'per_token_similarities': {}}
        
        adjacent_sims = []
        per_token_sims = {}
        
        for i in range(n_layers - 1):
            if i not in self.layer_outputs or (i + 1) not in self.layer_outputs:
                continue
            
            out_i = self.layer_outputs[i]      # (B, L, D)
            out_j = self.layer_outputs[i + 1]  # (B, L, D)
            
            # 全局余弦相似度（flatten 后计算）
            flat_i = out_i.reshape(-1, out_i.shape[-1])  # (B*L, D)
            flat_j = out_j.reshape(-1, out_j.shape[-1])
            
            # 归一化
            norm_i = F.normalize(flat_i, p=2, dim=-1)
            norm_j = F.normalize(flat_j, p=2, dim=-1)
            
            # 平均余弦相似度
            cos_sim = (norm_i * norm_j).sum(dim=-1).mean().item()
            adjacent_sims.append((i, i + 1, cos_sim))
            
            # 逐 token 相似度
            if token_indices is not None:
                # (B, L, D) -> 选择特定 token
                per_token_sim = []
                for t_idx in token_indices:
                    if t_idx < out_i.shape[1]:
                        tok_i = out_i[:, t_idx, :]  # (B, D)
                        tok_j = out_j[:, t_idx, :]
                        tok_norm_i = F.normalize(tok_i, p=2, dim=-1)
                        tok_norm_j = F.normalize(tok_j, p=2, dim=-1)
                        tok_sim = (tok_norm_i * tok_norm_j).sum(dim=-1).mean().item()
                        per_token_sim.append(tok_sim)
                per_token_sims[(i, i + 1)] = per_token_sim
        
        return {
            'adjacent_similarities': adjacent_sims,
            'per_token_similarities': per_token_sims
        }
    
    def compute_input_output_similarity(self, layer_idx):
        """
        计算某一层的输入和输出之间的余弦相似度。
        
        这表示该层对 hidden states 的"改变程度"。
        
        Args:
            layer_idx: 层索引
        
        Returns:
            float: 余弦相似度
        """
        if layer_idx not in self.layer_inputs or layer_idx not in self.layer_outputs:
            return None
        
        inp = self.layer_inputs[layer_idx]   # (B, L, D)
        out = self.layer_outputs[layer_idx]  # (B, L, D)
        
        flat_inp = inp.reshape(-1, inp.shape[-1])
        flat_out = out.reshape(-1, out.shape[-1])
        
        norm_inp = F.normalize(flat_inp, p=2, dim=-1)
        norm_out = F.normalize(flat_out, p=2, dim=-1)
        
        cos_sim = (norm_inp * norm_out).sum(dim=-1).mean().item()
        return cos_sim


@torch.no_grad()
def generate_with_dual_cache_hooked(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    collector: LayerHiddenStatesCollector = None,
    collect_per_step=False
):
    """
    基于 generate_with_dual_cache 的 hook 版本，用于收集各层 hidden states。
    
    与原版功能完全一致，仅额外支持通过 collector 收集中间层信息。
    
    Args:
        model, prompt, steps, gen_length, block_length, temperature, 
        remasking, mask_id, threshold, factor: 同 generate_with_dual_cache
        
        collector: LayerHiddenStatesCollector 实例，用于收集 hidden states。
                   如果为 None，则行为与原版完全一致。
        
        collect_per_step: 如果为 True，每个 step 后会保存 hidden states 快照
                          （用于详细分析，但会占用较多内存）
    
    Returns:
        如果 collector 为 None:
            (x, nfe) - 同原版
        如果 collector 不为 None:
            (x, nfe, step_hidden_states) 
            其中 step_hidden_states 是一个 list，每个元素是对应 step 的 
            {'inputs': {...}, 'outputs': {...}}（仅当 collect_per_step=True 时有数据）
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0
    step_hidden_states = []  # 收集每步的 hidden states

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # 1) Warm KV-cache
        if collector is not None:
            collector.clear()
        
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        if collector is not None and collect_per_step:
            step_hidden_states.append(collector.get_layer_hidden_states())

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            
            if collector is not None:
                collector.clear()
            
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits

            if collector is not None and collect_per_step:
                step_hidden_states.append(collector.get_layer_hidden_states())

            mask_blk = (x[:, s:e] == mask_id)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

            nfe += 1

    if collector is None:
        return x, nfe
    else:
        return x, nfe, step_hidden_states


def compute_all_layer_similarities(collector: LayerHiddenStatesCollector):
    """
    便捷函数：计算并返回所有层间相似度分析结果。
    
    Args:
        collector: 已收集数据的 LayerHiddenStatesCollector
    
    Returns:
        dict: {
            'adjacent_output_similarities': [(i, i+1, sim), ...],
            'input_output_similarities': {layer_idx: sim, ...}
        }
    """
    # 相邻层输出相似度
    adj_sims = collector.compute_layer_similarities()['adjacent_similarities']
    
    # 每层的输入输出相似度
    io_sims = {}
    for layer_idx in collector.layer_outputs.keys():
        sim = collector.compute_input_output_similarity(layer_idx)
        if sim is not None:
            io_sims[layer_idx] = sim
    
    return {
        'adjacent_output_similarities': adj_sims,
        'input_output_similarities': io_sims
    }


if __name__ == '__main__':
    main()
