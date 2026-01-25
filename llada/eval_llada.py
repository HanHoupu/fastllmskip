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

"""
================================================================================
LLaDA 评估适配器 (lm-evaluation-harness)
================================================================================

这个文件的作用：
    让 LLaDA（Diffusion LLM）能够在 lm-evaluation-harness 库的标准 benchmark 上评估。
    lm-evaluation-harness 是 EleutherAI 开发的通用 LLM 评估框架。

核心挑战：
    传统自回归 LLM（如 GPT）可以直接计算 log P(target | prefix)。
    但 Diffusion LLM 是非自回归的，没有直接的概率公式。
    
解决方案：
    使用 **Monte Carlo 估计** 来近似计算 log-likelihood。
    
评估流程：
    1. loglikelihood() - 计算 log P(target | prefix)，用于 perplexity、accuracy 等
    2. generate_until() - 生成文本直到遇到停止符，用于生成任务（GSM8K、HumanEval等）

使用方法：
    lm_eval --model llada_dist --model_args model_path=GSAI-ML/LLaDA-8B-Base --tasks hellaswag
"""

import accelerate  # HuggingFace 分布式训练库
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset  # HuggingFace datasets 库
from lm_eval.__main__ import cli_evaluate  # lm-evaluation-harness 的命令行入口
from lm_eval.api.instance import Instance  # 评估实例
from lm_eval.api.model import LM  # LM 基类，需要继承并实现特定方法
from lm_eval.api.registry import register_model  # 模型注册装饰器
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache, generate_with_dual_cache_tokenskip
from model.modeling_llada import LLaDAModelLM
import json
import time


def set_seed(seed):
    """设置随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 禁用 cuDNN 的非确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================================
# 核心类：LLaDA 评估适配器
# ================================================================================

@register_model("llada_dist")  # 注册模型名称，使用时通过 --model llada_dist 调用
class LLaDAEvalHarness(LM):
    """
    LLaDA 的 lm-evaluation-harness 适配器。
    
    继承自 lm_eval.api.model.LM，需要实现以下方法：
        - loglikelihood(): 计算 log P(target | prefix)
        - loglikelihood_rolling(): 滚动计算 log-likelihood（未实现）
        - generate_until(): 生成文本直到停止符
    
    核心技术点：
        由于 Diffusion LLM 无法直接计算 log-likelihood，
        我们使用 Monte Carlo 估计：
        
        log P(target | prefix) ≈ -E[loss / mask_ratio]
        
        其中 loss 是在随机 mask 后计算的交叉熵损失。
    """
    
    def __init__(
        self,
        model_path='',           # 模型路径（HuggingFace hub 或本地）
        mask_id=126336,          # [MASK] token 的 id
        max_length=4096,         # 最大序列长度
        batch_size=32,           # 批量大小
        mc_num=128,              # Monte Carlo 采样次数（越大估计越准，但越慢）
        is_check_greedy=True,    # 是否检查 greedy decoding（用于 LAMBADA 等任务）
        steps=1024,              # 生成时的去噪步数
        gen_length=1024,         # 生成长度
        block_length=1024,       # block 长度（用于 semi-autoregressive 生成）
        remasking='low_confidence',  # 重掩码策略
        device="cuda",
        use_cache=False,         # 是否使用 KV Cache
        threshold=None,          # 置信度阈值
        factor=None,             # 动态阈值因子
        save_dir=None,           # 保存生成结果的目录（用于断点续跑）
        show_speed=False,        # 是否显示速度统计
        dual_cache=False,        # 是否使用 dual cache
        **kwargs,
    ):
        """
        初始化评估适配器。
        
        参数说明：
            mc_num: Monte Carlo 采样次数
                    - 用于估计 log-likelihood
                    - 建议值：128（越大越准，但越慢）
                    - 必须是 batch_size 的整数倍
            
            is_check_greedy: 是否启用 greedy 验证
                    - 某些任务（如 LAMBADA）需要验证模型是否能通过贪婪解码生成目标
                    - 设为 False 可以大幅加速评估（LLaDA 论文中的任务不需要）
                    
            steps / gen_length / block_length: 生成参数
                    - 与 generate.py 中的参数一致
        """
        super().__init__()

        # ==================== 分布式初始化 ====================
        # 使用 HuggingFace Accelerate 进行多 GPU 分布式评估
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator  # 多卡模式
        else:
            self.accelerator = None  # 单卡模式
        
        # ==================== 加载模型 ====================
        model_kwargs = {}
        if self.accelerator is not None:
            # 分布式模式：指定 device_map
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        
        # 加载配置并启用 Flash Attention（加速注意力计算）
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        
        # 加载模型，使用 bfloat16 精度节省显存
        self.model = LLaDAModelLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            config=config, 
            **model_kwargs
        )
        self.model.eval()  # 切换到评估模式（禁用 dropout 等）

        # ==================== 设备配置 ====================
        self.device = torch.device(device)
        if self.accelerator is not None:
            # 分布式模式
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index  # 当前进程编号
            self._world_size = self.accelerator.num_processes  # 总进程数
        else: 
            # 单卡模式
            self.model = self.model.to(device)

        # ==================== Tokenizer ====================
        self.mask_id = mask_id  # [MASK] token id = 126336
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ==================== Monte Carlo 参数 ====================
        self.mc_num = mc_num  # MC 采样次数
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0  # mc_num 必须是 batch_size 的整数倍
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        # ==================== 生成参数 ====================
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        
        # 判断是否为 Instruct 模型（影响 prompt 格式）
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        
        # ==================== 其他配置 ====================
        self.save_dir = save_dir  # 保存路径（断点续跑）
        self.show_speed = show_speed  # 速度统计
        self.dual_cache = dual_cache  # 是否用 dual cache
        
        # Token Skip 参数（新版：基于最终 hidden state 判定）
        self.token_skip = kwargs.get('token_skip', False)  # 是否启用 Token Skip
        self.skip_threshold = float(kwargs.get('skip_threshold', 0.95))  # cos sim 阈值
        self.force_full_every_k = int(kwargs.get('force_full_every_k', 3))  # 每 K 步强制全算
    # ==================== 分布式相关属性 ====================
    
    @property
    def rank(self):
        """当前进程编号（分布式评估时使用）"""
        return self._rank
    
    @property
    def world_size(self):
        """总进程数（分布式评估时使用）"""
        return self._world_size

    # ==================== Monte Carlo 核心：前向扩散过程 ====================
    
    def _forward_process(self, batch, prompt_index):
        """
        前向扩散过程：给序列添加噪声（随机 mask 一些 token）。
        
        这是 Monte Carlo 估计 log-likelihood 的核心。
        
        Diffusion LLM 的训练目标是：给定被 mask 的序列，预测原始 token。
        因此，我们可以通过多次随机 mask、计算预测损失来估计 log-likelihood。
        
        参数：
            batch: (B, L) 原始序列 [prefix + target]
            prompt_index: (L,) bool tensor，True 表示该位置是 prompt（不 mask）
            
        返回：
            noisy_batch: (B, L) 加噪后的序列（target 部分被随机 mask）
            mask_ratio: (B, L) 每个样本的 mask 比例（用于加权 loss）
            
        算法流程：
            1. 对于 batch 中的每个样本，随机选择要 mask 的 token 数量 k
            2. 在 target 部分随机选择 k 个位置进行 mask
            3. prompt 部分永远不 mask
            
        数学原理：
            设 mask 比例为 t = k / target_len
            则 loss / t 是 log-likelihood 的无偏估计
            通过多次采样取平均，得到最终估计值
        """
        b, l = batch.shape  # B = batch_size, L = sequence_length

        # target_len = 序列中 target 的长度（不包括 prompt）
        target_len = (l - prompt_index.sum()).item()
        
        # 随机选择一个 k（要 mask 的 token 数量），范围 [1, target_len]
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        # 为 batch 中每个样本分配不同的 mask 数量（保证覆盖不同的 t 值）
        # x[i] 表示第 i 个样本要 mask 的 token 数量
        # 使用 linspace 确保 batch 内的 mask 比例分布均匀
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1  # 确保 x 在 [1, target_len] 范围内
        assert x.min() >= 1 and x.max() <= target_len

        # 构建 mask 矩阵
        # indices[i, j] = j，表示 target 中的位置索引
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        
        # is_mask[i, j] = True 表示第 i 个样本的第 j 个 target 位置要被 mask
        # 初始时：前 x[i] 个位置为 True
        is_mask = indices < x.unsqueeze(1)

        # 随机打乱每个样本的 mask 位置（不是按顺序 mask 前 k 个，而是随机选 k 个）
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        # 在前面拼接 prompt 部分（全 False，不 mask）
        # is_mask 形状：(B, prompt_len + target_len) = (B, L)
        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        # 应用 mask：被 mask 的位置替换为 mask_id
        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        # 计算 mask 比例 t = x / target_len（用于后续 loss 加权）
        # 返回形状：(B, L)，每个位置的值都是该样本的 mask 比例
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    # ==================== 模型前向 ====================
    
    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        """
        获取模型输出的 logits。
        
        支持 Classifier-Free Guidance (CFG)，但当前 self.cfg 未初始化，
        所以这个功能实际上没用到。
        
        参数：
            batch: (B, L) 输入序列
            prompt_index: (L,) bool tensor，标记 prompt 位置
            
        返回：
            logits: (B, L, vocab_size) 模型预测的 logits
        """
        # CFG 逻辑（当前未启用）
        if self.cfg > 0.:
            # CFG 需要同时计算有条件和无条件的 logits
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id  # 无条件版本：把 prompt 也 mask 掉
            batch = torch.cat([batch, un_batch])  # 拼接：[有条件, 无条件]

        # 模型前向
        logits = self.model(batch).logits

        if self.cfg > 0.:
            # CFG 公式：logits = un_logits + (cfg + 1) * (logits - un_logits)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    # ==================== Monte Carlo 估计 Log-Likelihood ====================
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        """
        使用 Monte Carlo 方法估计 log P(target | prefix)。
        
        这是 Diffusion LLM 评估的核心函数！
        
        原理：
            对于 Diffusion LLM，无法直接计算 log P(x)。
            但我们知道训练目标是最小化去噪损失：
                L(x, t) = E[CE(model(mask(x, t)), x)]
            
            可以证明，log P(x) 与 E[L(x, t) / t] 成正比。
            因此，我们通过多次随机 mask、计算加权 loss 来估计 log-likelihood。
        
        参数：
            prefix: (prefix_len,) prompt 的 token ids
            target: (target_len,) 目标序列的 token ids
            
        返回：
            log_likelihood: float，估计的 log P(target | prefix)
            
        算法：
            1. 拼接 seq = [prefix, target]
            2. 重复 mc_num 次：
               a. 随机 mask target 的一部分
               b. 计算模型预测的 cross-entropy loss
               c. 除以 mask 比例得到加权 loss
            3. 返回负的平均加权 loss（负号因为 loss 越小越好）
        """
        # 拼接 prefix 和 target
        seq = torch.concatenate([prefix, target])[None, :]  # (1, L)
        
        # 复制 batch_size 份（并行计算多个 MC 样本）
        seq = seq.repeat((self.batch_size, 1)).to(self.device)  # (B, L)

        # 标记哪些位置是 prompt（不参与 loss 计算）
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []  # 累积每次 MC 采样的 loss
        
        # Monte Carlo 采样循环
        # 总共采样 mc_num 次，每次处理 batch_size 个样本
        for _ in range(self.mc_num // self.batch_size):
            # 1. 前向扩散：随机 mask target 的一部分
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            # perturbed_seq: 加噪后的序列
            # p_mask: mask 比例（用于加权）

            # 2. 找出被 mask 的位置
            mask_indices = perturbed_seq == self.mask_id

            # 3. 模型前向，获取 logits
            logits = self.get_logits(perturbed_seq, prompt_index)

            # 4. 计算 cross-entropy loss，并除以 mask 比例
            # 这是 log-likelihood 的无偏估计
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size  # 平均
            loss_acc.append(loss.item())

        # 返回负的平均 loss（因为 likelihood 越大越好，loss 越小越好）
        return - sum(loss_acc) / len(loss_acc)

    # ==================== Greedy 验证（用于 LAMBADA 等任务） ====================
    
    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        """
        验证模型是否能通过贪婪解码生成目标序列。
        
        某些评估任务（如 LAMBADA）需要这个功能：
            不是计算 P(target | prefix)，
            而是验证 greedy_decode(prefix) == target。
        
        参数：
            prefix: (prefix_len,) prompt 的 token ids
            target: (target_len,) 目标序列的 token ids
            
        返回：
            correct: bool，True 表示贪婪解码结果与 target 完全一致
            
        算法（Diffusion LLM 的贪婪解码）：
            1. 初始化：seq = [prefix, MASK, MASK, ..., MASK]
            2. 每步选择置信度最高的一个 MASK 位置 unmask
            3. 重复直到所有 MASK 都被 unmask
            4. 检查生成结果是否与 target 一致
            
        注意：
            这个函数很慢！每个 target token 需要一次前向传播。
            建议设置 is_check_greedy=False 来跳过这个验证。
        """
        # 如果禁用了 greedy 检查，直接返回 False
        if not self.is_check_greedy:
            return False

        # 初始化序列：[prefix] + [MASK, MASK, ..., MASK]
        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix  # 填入 prefix

        # 逐步 unmask，每步只 unmask 一个 token（最高置信度的）
        for i in range(len(target)):
            # 找出当前所有 MASK 位置
            mask_index = (seq == self.mask_id)
            
            # 获取 MASK 位置的 logits
            logits = self.get_logits(seq, prompt_index)[mask_index]
            
            # 贪婪选择：取 argmax
            x0 = torch.argmax(logits, dim=-1)

            # 计算每个预测的置信度（softmax 概率）
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            
            # 只保留置信度最高的一个，其他的重新设为 MASK
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id  # 除了最高置信度的，其他都保持 MASK
            
            # 更新序列
            seq[mask_index] = x0.clone()
        
        # 检查生成结果是否与 target 完全一致
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    # ==================== Tokenization 工具 ====================
    
    def _encode_pair(self, context, continuation):
        """
        编码 (context, continuation) 对。
        
        处理 context 末尾空格的问题：
            tokenizer("hello ") 可能不等于 tokenizer("hello") + tokenizer(" ")
            所以需要特殊处理，确保 tokenization 的一致性。
        
        参数：
            context: str，上下文文本
            continuation: str，续写文本
            
        返回：
            context_enc: List[int]，context 的 token ids
            continuation_enc: List[int]，continuation 的 token ids
        """
        # 处理 context 末尾的空格
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            # 把末尾空格移到 continuation 的开头
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # 编码完整序列
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        # 编码 context
        context_enc = self.tokenizer(context)["input_ids"]

        # continuation 的 token ids = 完整序列 - context
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    # ==================== lm-eval 接口：loglikelihood ====================
    
    def loglikelihood(self, requests):
        """
        lm-evaluation-harness 要求实现的接口。
        
        计算每个 (prefix, target) 对的 log P(target | prefix)。
        
        参数：
            requests: List[Instance]，每个 Instance 包含 (prefix, target) 对
            
        返回：
            List[(log_likelihood, is_greedy)]
            - log_likelihood: float，log P(target | prefix)
            - is_greedy: float，0.0 或 1.0，表示是否通过 greedy 验证
            
        用途：
            - HellaSwag、ARC 等选择题任务：选择 log-likelihood 最高的选项
            - LAMBADA：需要 is_greedy 来判断准确率
        """
        # 内部 tokenization 函数
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        # 构建数据集
        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)  # tokenize
        ds = ds.with_format("torch")  # 转为 torch tensor
        
        # 检查序列长度
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                # 计算 log-likelihood（Monte Carlo 估计）
                ll = self.get_loglikelihood(prefix, target)

                # 检查 greedy 解码是否正确（可选）
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        
        torch.cuda.empty_cache()  # 清理显存
        return out

    def loglikelihood_rolling(self, requests):
        """滚动计算 log-likelihood，未实现"""
        raise NotImplementedError
    
    
    # ==================== lm-eval 接口：generate_until ====================
    
    def generate_until(self, requests):
        """
        lm-evaluation-harness 要求实现的接口。
        
        生成文本直到遇到停止符。用于生成类任务（GSM8K、HumanEval、MBPP 等）。
        
        参数：
            requests: List[Instance]，每个 Instance 包含：
                - args[0]: str，prompt 文本
                - args[1]: dict，包含 'until' 字段（停止符列表）
                
        返回：
            List[str]，生成的文本列表
            
        特性：
            - 支持批量生成（batch_size > 1）
            - 支持断点续跑（通过 save_dir 参数）
            - 支持三种生成模式：baseline / prefix_cache / dual_cache
        """
        output = []  # 存储生成结果
        num_tokens = 0  # 统计生成的 token 数（用于速度计算）
        num_nfe = 0  # 统计 NFE（模型前向次数）
        processed_count = 0  # 已处理数量（断点续跑用）
        
        # ==================== 断点续跑：加载已有结果 ====================
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank  # 当前进程编号
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            
            # 如果已有保存文件，加载之前的结果
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        
        # ==================== 构建 batch ====================
        # 把 requests 分成多个 batch
        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue  # 跳过已处理的
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])  # 开始新 batch
        
        # 移除最后一个空 batch
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        # ==================== 主生成循环 ====================
        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0  # batch 中最长的 prompt 长度
            pad_len = []  # 每个样本需要 pad 多少
            
            # ---------- Tokenize 每个 prompt ----------
            for req in batch:
                question = req.args[0]  # 问题文本
                
                # 根据模型类型（Instruct 或 Base）构造 prompt
                if self.is_instruct:
                    # Instruct 模型：使用 chat template
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = self.tokenizer(user_input)['input_ids']
                else:
                    # Base 模型：直接用原始问题
                    user_input = question
                    input_ids = self.tokenizer(user_input)['input_ids']
                
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            
            # ---------- 左填充（Left Padding）到相同长度 ----------
            # 注意：LLM 通常使用左填充，以保持生成位置一致
            batched_input_ids = [
                torch.cat([
                    torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device),
                    torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                ], dim=1) 
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)  # (B, max_len)
            batched_input_ids = batched_input_ids.to(self.device)
            
            # ---------- 构建 attention mask（处理 padding）----------
            if self.batch_size == 1:
                attention_mask = None  # 单样本不需要 mask
            else:
                # 构建 attention mask：padding 位置为 False
                attention_mask = torch.zeros(
                    (batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), 
                    device=self.device, dtype=torch.bool
                )
                for i in range(len(pad_len)):
                    # 只有非 padding 位置可以互相 attend
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            # ---------- 获取停止符 ----------
            stop_tokens = req.args[1]['until']  # 例如 ["\n\n", "<|endoftext|>"]
            
            # ---------- 调用生成函数 ----------
            input_ids = batched_input_ids
            
            if self.use_cache:
                if self.dual_cache and self.token_skip:
                    # 使用 Dual Cache + Token Skip 生成（新版：基于最终 hidden state 判定）
                    generated_answer, nfe = generate_with_dual_cache_tokenskip(
                        self.model, input_ids, 
                        steps=self.steps, 
                        gen_length=self.gen_length, 
                        block_length=self.block_length, 
                        temperature=0,  # 贪婪解码
                        remasking=self.remasking, 
                        mask_id=self.mask_id, 
                        threshold=self.threshold, 
                        factor=self.factor,
                        skip_threshold=self.skip_threshold,
                        force_full_every_k=self.force_full_every_k,
                    )
                elif self.dual_cache:
                    # 使用 Dual Cache 生成（不带 Token Skip）
                    generated_answer, nfe = generate_with_dual_cache(
                        self.model, input_ids, 
                        steps=self.steps, 
                        gen_length=self.gen_length, 
                        block_length=self.block_length, 
                        temperature=0,  # 贪婪解码
                        remasking=self.remasking, 
                        mask_id=self.mask_id, 
                        threshold=self.threshold, 
                        factor=self.factor
                    )
                else:
                    # 使用 Prefix Cache 生成
                    generated_answer, nfe = generate_with_prefix_cache(
                        self.model, input_ids, 
                        steps=self.steps, 
                        gen_length=self.gen_length, 
                        block_length=self.block_length, 
                        temperature=0, 
                        remasking=self.remasking, 
                        mask_id=self.mask_id, 
                        threshold=self.threshold, 
                        factor=self.factor
                    )
            else:
                # 使用基础生成（最慢）
                generated_answer, nfe = generate(
                    self.model, input_ids, 
                    steps=self.steps, 
                    gen_length=self.gen_length, 
                    block_length=self.block_length, 
                    temperature=0, 
                    remasking=self.remasking, 
                    mask_id=self.mask_id, 
                    threshold=self.threshold, 
                    factor=self.factor
                )

            # ---------- 后处理生成结果 ----------
            # HumanEval 任务需要特殊处理（保留代码格式）
            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                # 提取生成部分（去掉 prompt）
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    # 统计有效 token（排除 pad token 126081）
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                # 解码
                batched_generated_answer = [
                    self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) 
                    for i in range(len(generated_answer_ids))
                ]
            else:
                # 普通任务：需要按停止符截断
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    # 解码生成部分（保留特殊 token 以便检测停止符）
                    generated_answer_i = self.tokenizer.decode(
                        generated_answer[i][input_ids.shape[1]:], 
                        skip_special_tokens=False
                    )
                    
                    # 按停止符截断
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    
                    # 重新 tokenize 以统计 token 数
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    
                    # 最终解码（去掉特殊 token）
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # 添加到输出列表
            output.extend(batched_generated_answer)

            # ---------- 增量保存（断点续跑）----------
            if self.save_dir is not None:
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            # ---------- 打印进度 ----------
            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')
        
        end_time = time.time()
        
        # ==================== 速度统计 ====================
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
            
        return output


# ================================================================================
# 主入口：使用 lm-evaluation-harness 的 CLI
# ================================================================================

if __name__ == "__main__":
    # 调用 lm-eval 的命令行工具
    # 使用方法：
    #   python eval_llada.py --model llada_dist \
    #       --model_args model_path=GSAI-ML/LLaDA-8B-Base,steps=128,gen_length=256 \
    #       --tasks gsm8k \
    #       --batch_size 1
    cli_evaluate()
    
