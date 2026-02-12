#!/usr/bin/env python3
"""
Runner script for multipass comparison. Launch one per GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_multipass.py --method original  --limit 150 --output results_original.json
    CUDA_VISIBLE_DEVICES=1 python run_multipass.py --method plan_a    --limit 150 --output results_plan_a.json
    CUDA_VISIBLE_DEVICES=2 python run_multipass.py --method adaptive  --limit 150 --output results_adaptive.json
    CUDA_VISIBLE_DEVICES=3 python run_multipass.py --method factor    --limit 150 --output results_factor.json
"""

import argparse
import json
import time
import os

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache
from generate_multipass import generate_multipass, generate_adaptive, generate_factor_multipass


# ============ 5-shot prompt ============
FEW_SHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = <<12/60=0.2>>$0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = <<0.2*50=10>>$10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Maila read 12 x 2 = <<12*2=24>>24 pages today. So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday. There are 120 - 36 = <<120-36=84>>84 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages. #### 42

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: He writes each friend 3*2=<<3*2=6>>6 pages a week. So he writes 6*2=<<6*2=12>>12 pages every week. That means he writes 12*52=<<12*52=624>>624 pages a year. #### 624"""


def make_prompt(tokenizer, question):
    full = FEW_SHOT.strip() + f"\n\nQuestion: {question}\nAnswer:"
    m = [{"role": "user", "content": full}]
    return tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)


def run_method(method, model, tokenizer, device, ds, args):
    results = []
    start = time.time()

    for idx in range(args.limit):
        question = ds[idx]['question']
        prompt_text = make_prompt(tokenizer, question)
        input_ids = torch.tensor(tokenizer(prompt_text)['input_ids']).to(device).unsqueeze(0)
        plen = input_ids.shape[1]

        gen_kwargs = dict(
            steps=args.steps, gen_length=args.gen_length,
            block_length=args.block_length, temperature=0.,
        )

        if method == 'original':
            x, nfe = generate_with_dual_cache(
                model, input_ids, **gen_kwargs, threshold=args.threshold)
            stats = {'total_nfe': nfe}

        elif method == 'plan_a':
            x, nfe, stats = generate_multipass(
                model, input_ids, **gen_kwargs, threshold=args.threshold)

        elif method == 'adaptive':
            x, nfe, stats = generate_adaptive(
                model, input_ids, **gen_kwargs,
                initial_threshold=args.threshold, target_ratio=0.25)

        elif method == 'factor':
            x, nfe, stats = generate_factor_multipass(
                model, input_ids, **gen_kwargs, factor=args.factor)
        else:
            raise ValueError(f"Unknown method: {method}")

        answer = tokenizer.decode(x[0, plen:], skip_special_tokens=True)

        result = {
            'idx': idx,
            'nfe': nfe,
            'answer': answer,
            'stats': stats,
        }
        results.append(result)

        elapsed = time.time() - start
        eta = elapsed / (idx + 1) * (args.limit - idx - 1)
        extra = ""
        if 'passes' in stats:
            extra = f" passes={stats.get('num_passes', len(stats['passes']))}"
        print(f"[{method}][{idx+1:3d}/{args.limit}] nfe={nfe:3d}{extra}  ETA={eta:.0f}s")

    total_time = time.time() - start
    print(f"\n[{method}] Done! {total_time:.0f}s total ({total_time/args.limit:.1f}s/sample)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run multipass comparison method")
    parser.add_argument('--method', type=str, required=True,
                        choices=['original', 'plan_a', 'adaptive', 'factor'])
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--factor', type=float, default=1.0)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_{args.method}_n{args.limit}.json"

    device = 'cuda'
    print(f"Loading model: {args.model_path}")
    model = LLaDAModelLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    ds = load_dataset("gsm8k", "main", split="test")
    print(f"GSM8K test: {len(ds)} samples, running {args.limit}")

    results = run_method(args.method, model, tokenizer, device, ds, args)

    save_data = {
        'method': args.method,
        'config': vars(args),
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
