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

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import (generate, generate_with_prefix_cache, generate_with_dual_cache,
                      generate_with_dual_cache_expand, generate_with_streaming_cache,
                      generate_with_dual_cache_saber, generate_with_expand_saber,
                      generate_with_expand_saber_dynamic)
from model.modeling_llada import LLaDAModelLM
import json
import time
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        use_cache=False,
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        mid_block_expand=False,
        mid_trigger_ratio=0.5,
        rewarm_on_expand=True,
        front_block_fallback_only=False,
        hfb=None,
        step_records_dir=None,
        seed=None,
        suffix_mode=None,
        window_blocks=2,
        adaptive_alpha=None,
        streaming_cache=False,
        expand_trigger_mode=None,
        vel_threshold=0.5,
        conf_threshold=0.7,
        saber_dual_cache=False,
        saber_expand=False,
        saber_n=2,
        saber_mu=8,
        berm_mode='cross_step',
        saber_global_aadu=False,
        saber_mtr=0.8,
        saber_post_global_berm_rounds=0,
        saber_post_global_berm_window_mul=1,
        saber_fullstage_berm=False,
        saber_fullstage_berm_on_rewarm=True,
        saber_berm_scope='window',
        saber_dynamic=False,
        saber_n_hi=8,
        saber_n_lo=3,
        saber_mu_lo=4,
        saber_mu_hi=16,
        saber_gamma_n=1.0,
        saber_gamma_mu=1.0,
        saber_gamma_floor=1.5,
        saber_remask_ratio_hi=0.35,
        saber_floor_lo=0,
        eos_penalty=0.0,
        eos_token_id=None,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        if seed is not None:
            set_seed(int(seed))
            print(f"Seed set to {seed}")

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.mid_block_expand = mid_block_expand
        self.mid_trigger_ratio = float(mid_trigger_ratio)
        self.rewarm_on_expand = rewarm_on_expand
        self.front_block_fallback_only = front_block_fallback_only
        self.hfb = float(hfb) if hfb is not None else None
        self.step_records_dir = step_records_dir
        self._all_step_records = []  # collected when step_records_dir is set
        self.suffix_mode = suffix_mode if suffix_mode not in (None, 'None', 'none', 'full') else None
        self.window_blocks = int(window_blocks)
        self.adaptive_alpha = float(adaptive_alpha) if adaptive_alpha is not None else None
        self.streaming_cache = (str(streaming_cache).lower() in ('true', '1', 'yes'))
        self.expand_trigger_mode = expand_trigger_mode if expand_trigger_mode not in (None, 'None', 'none') else 'ratio'
        self.vel_threshold = float(vel_threshold)
        self.conf_threshold = float(conf_threshold)
        self.saber_dual_cache = (str(saber_dual_cache).lower() in ('true', '1', 'yes'))
        self.saber_expand = (str(saber_expand).lower() in ('true', '1', 'yes'))
        self.saber_n = int(saber_n)
        self.saber_mu = int(saber_mu)
        self.berm_mode = str(berm_mode)
        self.saber_global_aadu = (str(saber_global_aadu).lower() in ('true', '1', 'yes'))
        self.saber_mtr = float(saber_mtr)
        self.saber_post_global_berm_rounds = int(saber_post_global_berm_rounds)
        self.saber_post_global_berm_window_mul = int(saber_post_global_berm_window_mul)
        self.saber_fullstage_berm = (str(saber_fullstage_berm).lower() in ('true', '1', 'yes'))
        self.saber_fullstage_berm_on_rewarm = (str(saber_fullstage_berm_on_rewarm).lower() in ('true', '1', 'yes'))
        self.saber_berm_scope = str(saber_berm_scope)
        self.saber_dynamic = (str(saber_dynamic).lower() in ('true', '1', 'yes'))
        self.saber_n_hi = int(saber_n_hi)
        self.saber_n_lo = int(saber_n_lo)
        self.saber_mu_lo = int(saber_mu_lo)
        self.saber_mu_hi = int(saber_mu_hi)
        self.saber_gamma_n = float(saber_gamma_n)
        self.saber_gamma_mu = float(saber_gamma_mu)
        self.saber_gamma_floor = float(saber_gamma_floor)
        self.saber_remask_ratio_hi = float(saber_remask_ratio_hi)
        self.saber_floor_lo = int(saber_floor_lo)
        self.eos_penalty = float(eos_penalty)
        self.eos_token_id = int(eos_token_id) if eos_token_id is not None else None

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        
        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = self.tokenizer(user_input)['input_ids']
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)['input_ids']
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros((batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), device=self.device, dtype=torch.bool)
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            stop_tokens = req.args[1]['until']
            input_ids = batched_input_ids
            step_records = None
            if self.saber_dynamic:
                generated_answer, nfe = generate_with_expand_saber_dynamic(
                    self.model, input_ids, steps=self.steps,
                    gen_length=self.gen_length, block_length=self.block_length,
                    temperature=0, remasking=self.remasking, mask_id=self.mask_id,
                    mid_trigger_ratio=self.saber_mtr,
                    global_aadu=self.saber_global_aadu,
                    berm_mode=self.berm_mode,
                    n_hi=self.saber_n_hi, n_lo=self.saber_n_lo,
                    mu_lo=self.saber_mu_lo, mu_hi=self.saber_mu_hi,
                    gamma_n=self.saber_gamma_n, gamma_mu=self.saber_gamma_mu,
                    gamma_floor=self.saber_gamma_floor,
                    remask_ratio_hi=self.saber_remask_ratio_hi,
                    floor_lo=self.saber_floor_lo,
                    berm_scope=self.saber_berm_scope,
                    eos_penalty=self.eos_penalty,
                    eos_token_id=self.eos_token_id,
                )
            elif self.saber_expand:
                generated_answer, nfe = generate_with_expand_saber(
                    self.model, input_ids, steps=self.steps,
                    gen_length=self.gen_length, block_length=self.block_length,
                    temperature=0, remasking=self.remasking, mask_id=self.mask_id,
                    mid_trigger_ratio=self.saber_mtr,
                    saber_n=self.saber_n, saber_mu=self.saber_mu,
                    berm_mode=self.berm_mode, global_aadu=self.saber_global_aadu,
                    post_global_berm_rounds=self.saber_post_global_berm_rounds,
                    post_global_berm_window_mul=self.saber_post_global_berm_window_mul,
                    fullstage_berm=self.saber_fullstage_berm,
                    fullstage_berm_on_rewarm=self.saber_fullstage_berm_on_rewarm,
                    berm_scope=self.saber_berm_scope,
                )
            elif self.saber_dual_cache:
                generated_answer, nfe = generate_with_dual_cache_saber(
                    self.model, input_ids, steps=self.steps,
                    gen_length=self.gen_length, block_length=self.block_length,
                    temperature=0, remasking=self.remasking, mask_id=self.mask_id,
                    saber_n=self.saber_n, saber_mu=self.saber_mu,
                    berm_mode=self.berm_mode, global_aadu=self.saber_global_aadu,
                )
            elif self.use_cache:
                if self.streaming_cache and self.mid_block_expand:
                    _record = (self.step_records_dir is not None)
                    ret = generate_with_dual_cache_expand(
                        self.model, input_ids, steps=self.steps,
                        gen_length=self.gen_length, block_length=self.block_length,
                        temperature=0, remasking=self.remasking,
                        mask_id=self.mask_id, threshold=self.threshold, factor=self.factor,
                        mid_trigger_ratio=self.mid_trigger_ratio,
                        rewarm_on_expand=self.rewarm_on_expand,
                        front_block_fallback_only=self.front_block_fallback_only,
                        hfb=self.hfb,
                        record_steps=_record,
                        suffix_mode=self.suffix_mode,
                        window_blocks=self.window_blocks,
                        adaptive_alpha=self.adaptive_alpha,
                        streaming_refine=True,
                        expand_trigger_mode=self.expand_trigger_mode,
                        vel_threshold=self.vel_threshold,
                        conf_threshold=self.conf_threshold,
                    )
                    if _record:
                        generated_answer, nfe, step_records = ret
                    else:
                        generated_answer, nfe = ret
                elif self.streaming_cache:
                    generated_answer, nfe = generate_with_streaming_cache(
                        self.model, input_ids, steps=self.steps,
                        gen_length=self.gen_length, block_length=self.block_length,
                        temperature=0, remasking=self.remasking,
                        mask_id=self.mask_id, threshold=self.threshold, factor=self.factor,
                        window_blocks=self.window_blocks,
                        adaptive_alpha=self.adaptive_alpha,
                    )
                elif self.mid_block_expand:
                    # Mid-block chain expansion variant
                    _record = (self.step_records_dir is not None)
                    ret = generate_with_dual_cache_expand(
                        self.model, input_ids, steps=self.steps,
                        gen_length=self.gen_length, block_length=self.block_length,
                        temperature=0, remasking=self.remasking,
                        mask_id=self.mask_id, threshold=self.threshold, factor=self.factor,
                        mid_trigger_ratio=self.mid_trigger_ratio,
                        rewarm_on_expand=self.rewarm_on_expand,
                        front_block_fallback_only=self.front_block_fallback_only,
                        hfb=self.hfb,
                        record_steps=_record,
                        suffix_mode=self.suffix_mode,
                        window_blocks=self.window_blocks,
                        adaptive_alpha=self.adaptive_alpha,
                        expand_trigger_mode=self.expand_trigger_mode,
                        vel_threshold=self.vel_threshold,
                        conf_threshold=self.conf_threshold,
                    )
                    if _record:
                        generated_answer, nfe, step_records = ret
                    else:
                        generated_answer, nfe = ret
                elif self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor,
                                        suffix_mode=self.suffix_mode, window_blocks=self.window_blocks, adaptive_alpha=self.adaptive_alpha)
                else:
                    generated_answer, nfe = generate_with_prefix_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
            
            if step_records is not None:
                self._all_step_records.append(step_records)

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                # print('question: ', question)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')
            # self.accelerator.wait_for_everyone()
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")

        # Save step records to JSON if enabled
        if self.step_records_dir and self._all_step_records:
            os.makedirs(self.step_records_dir, exist_ok=True)
            records_path = os.path.join(self.step_records_dir, 'step_records.json')
            with open(records_path, 'w', encoding='utf-8') as f:
                json.dump(self._all_step_records, f)
            print(f"Step records saved to {records_path} ({len(self._all_step_records)} samples)")

        return output


if __name__ == "__main__":
    cli_evaluate()
    
