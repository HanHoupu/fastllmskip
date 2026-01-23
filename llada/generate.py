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

@torch.no_grad()  # 禁用梯度计算，推理时节省显存
def generate_with_dual_cache_tokenskip(
    model,           # LLaDA 模型
    prompt,          # 输入 prompt 的 token ids，形状 (B, Lp)，B=batch_size, Lp=prompt长度
    steps=128,       # 总的去噪步数
    gen_length=128,  # 要生成的 token 数量
    block_length=128,# 每个 block 的长度（semi-autoregressive 的单位）
    temperature=0.,  # 采样温度，0 表示贪婪解码
    remasking="low_confidence",  # 重掩码策略："low_confidence" 或 "random"
    mask_id=126336,  # [MASK] token 的 id
    threshold=None,  # 置信度阈值，超过这个值的 token 会被 unmask（并行解码模式）
    factor=None,     # 动态阈值因子（用于 get_transfer_index_dynamic）
    # Token Skip 超参
    skip_layer_k=8,       # 判定用的前 K 层
    skip_threshold=0.95,  # 平均 cos sim 阈值
    skip_outlier=0.8,     # 偏离阈值
):
    """
    使用 Dual Cache 的生成函数（你可以在这里添加 token skip 优化）
    
    整体流程：
    1. 初始化：[prompt] + [MASK, MASK, ..., MASK]
    2. 分 block 处理，每个 block 内迭代去噪
    3. 每步选择高置信度的 token 进行 unmask
    4. 最终输出：[prompt] + [生成的 tokens]
    """
    
    # ==================== 初始化阶段 ====================
    
    # B = batch size（一次处理多少个样本）
    B = prompt.shape[0]
    
    # Lp = prompt 的长度（token 数量）
    Lp = int(prompt.shape[1])
    
    # 检查：生成长度必须能被 block 长度整除
    assert gen_length % block_length == 0
    
    # 计算总共有多少个 block
    # 例如：gen_length=128, block_length=32 → num_blocks=4
    num_blocks = gen_length // block_length

    # 检查：总步数必须能被 block 数整除
    assert steps % num_blocks == 0
    
    # 每个 block 内的去噪步数
    # 例如：steps=128, num_blocks=4 → steps_per_block=32
    steps_per_block = steps // num_blocks

    # 创建输出序列 x，初始化为全 [MASK]
    # 形状：(B, Lp + gen_length)
    # 例如：prompt 长度 100，生成长度 128 → x 形状 (B, 228)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    
    # 把 prompt 复制到 x 的前面部分
    # x = [prompt_token_1, prompt_token_2, ..., MASK, MASK, ..., MASK]
    x[:, :Lp] = prompt

    # NFE = Number of Forward Evaluations（模型前向传播次数，用于统计计算量）
    nfe = 0

    # ==================== 逐 Block 处理 ====================
    
    # 遍历每个 block
    # 例如：num_blocks=4，则 nb = 0, 1, 2, 3
    for nb in range(num_blocks):
        
        # s = 当前 block 的起始位置（在 x 中的索引）
        # e = 当前 block 的结束位置
        # 例如：nb=0, Lp=100, block_length=32 → s=100, e=132
        # 例如：nb=1, Lp=100, block_length=32 → s=132, e=164
        s = Lp + nb * block_length
        e = s + block_length

        # 找出当前 block 中哪些位置是 [MASK]
        # block_mask_index: (B, block_length) 的 bool tensor
        # True 表示该位置是 [MASK]，需要被预测
        block_mask_index = (x[:, s:e] == mask_id)
        
        # 计算每一步应该 unmask 多少个 token（用于 quota 模式）
        # num_transfer_tokens: (B, steps_per_block)
        # 如果用 threshold 模式，这个值不会被使用
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # ==================== 第 1 步：完整前向，预热 KV Cache ====================
        
        # 对完整序列做一次前向传播
        # use_cache=True 表示要保存 KV Cache
        out_full = model(x, use_cache=True)
        
        # 保存 KV Cache，后续步骤会复用
        # past_key_values 是一个 tuple，每层一个 (key, value) 对
        past_key_values = out_full.past_key_values
        
        # 计数：完成一次前向传播
        nfe += 1

        # 创建 replace_position：标记当前 block 的位置
        # 这个 tensor 告诉模型：在后续前向中，只更新这些位置的 KV Cache
        # 形状：(B, Lp + gen_length)，全 False，只有 [s:e] 区间是 True
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        # ==================== Step 0：基于完整 logits 做第一次 unmask ====================
        
        # 找出当前序列中所有的 [MASK] 位置
        global_mask_index = (x == mask_id)
        
        # 只处理当前 block 及之前的 mask，不处理后面 block 的 mask
        # 把 e 位置之后的 mask 标记设为 False
        global_mask_index[:, e:] = False

        # 根据 factor 参数选择使用哪种 transfer 策略
        if factor is None:
            # 使用 threshold 模式或 quota 模式
            # 如果 threshold 不为 None，quota0 = None（使用 threshold 模式）
            # 如果 threshold 为 None，quota0 = num_transfer_tokens[:, 0]（使用 quota 模式）
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]
            
            # 调用 get_transfer_index 获取：
            # - x0: 模型预测的 token（所有位置）
            # - transfer_index: 哪些位置应该被 unmask（bool tensor）
            x0, transfer_index = get_transfer_index(
                out_full.logits,     # 模型输出的 logits
                temperature,         # 采样温度
                remasking,           # 重掩码策略
                global_mask_index,   # 哪些位置是 [MASK]
                x,                   # 当前序列
                quota0,              # 这一步要 unmask 多少个（quota 模式）
                threshold            # 置信度阈值（threshold 模式）
            )
        else:
            # 使用动态阈值模式
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # 更新序列 x：把选中位置的 [MASK] 替换为预测的 token
        # torch.where(condition, x, y)：condition 为 True 时取 x，否则取 y
        # transfer_index 为 True 的位置 → 用 x0（预测的 token）
        # transfer_index 为 False 的位置 → 保持原来的 x
        x = torch.where(transfer_index, x0, x)

        # ==================== Step 1 ~ N：迭代 refinement ====================
        
        prev_hidden = None  # 用于 Token Skip 判定
        
        for i in range(1, steps_per_block):
            # 提前退出：如果当前 block 已经没有 [MASK] 了，就不用继续了
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            
            # 模型前向（Token Skip 逻辑在 model 内部双 loop 实现）
            out_blk = model(
                x[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
                output_hidden_states=True,
                skip_layer_k=skip_layer_k,
                skip_threshold=skip_threshold,
                skip_outlier=skip_outlier,
                prev_hidden=prev_hidden,
            )
            logits_blk = out_blk.logits
            prev_hidden = out_blk.hidden_states  # 保存供下一 step 判定

            # 找出当前 block 中哪些位置还是 [MASK]
            mask_blk = (x[:, s:e] == mask_id)

            # 选择 transfer 策略
            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # 更新当前 block 的 tokens
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

            nfe += 1

    # ==================== 返回结果 ====================
    # x: 完整的输出序列，形状 (B, Lp + gen_length)
    # nfe: 总共的前向传播次数
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
