"""
Standalone runner for P0 correction experiments.
Auto-generated for experiment_p0_correction.ipynb — GPU pool parallel execution.

Usage:
    python _p0_runner.py --config_name t0.4_flip_blk --threshold 0.4 \
        --correction flip --scope block --limit 100 --output_path results.pkl
"""
import argparse
import os
import sys
import gc
import time
import pickle
import re

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

MASK_ID = 126336
MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"

FEW_SHOT_EXAMPLES = """Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?
Answer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.
Tyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.
A double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.
#### 12

Question: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?
Answer: Mary will spend $20 + $2 = $<<20+2=22>>22.
Elle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.
Elle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.
Elle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.
So, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.
#### 1

Question: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?
Answer: The grill burned 3 * 60 = <<3*60=180>>180 coals.
It takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.
#### 240

Question: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?
Answer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.
It gained 2 * 200 = <<2*200=400>>400 pounds from acorns.
It still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.
Thus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.
Therefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.
#### 200

Question: Brendan can cut 8 yards of grass per day, he bought a lawnmower and it helped him to cut more yards by Fifty percent per day. How many yards will Brendan be able to cut after a week?
Answer: The additional yard Brendan can cut after buying the lawnmower is 8 x 0.50 = <<8*0.50=4>>4 yards.
So, the total yards he can cut with the lawnmower is 8 + 4 = <<8+4=12>>12.
Therefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards.
#### 84"""


# ---------------------------------------------------------------------------
# Generate function (identical to notebook version)
# ---------------------------------------------------------------------------
from generate import get_num_transfer_tokens, get_transfer_index


@torch.no_grad()
def generate_dualcache_corrected(
    model, prompt,
    steps=256, gen_length=256, block_length=32,
    temperature=0., remasking="low_confidence", mask_id=MASK_ID,
    threshold=0.4,
    correction="none",
    scope="block",
    regret_threshold=0.1,
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    total_len = Lp + gen_length

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    commit_conf = torch.zeros(B, total_len, dtype=torch.float64, device=model.device)

    nfe = 0
    corr_log = []

    def _apply_correction(logits, x_scope, cc_scope, check_mask, method, reg_thresh):
        if not check_mask.any() or method == "none":
            return x_scope, cc_scope, 0
        p = F.softmax(logits.to(torch.float64), dim=-1)
        new_argmax = torch.argmax(logits, dim=-1)
        if method == "regret":
            cur_conf = torch.gather(p, -1, x_scope.unsqueeze(-1)).squeeze(-1)
            regret = cc_scope - cur_conf
            should_fix = check_mask & (regret > reg_thresh) & (new_argmax != x_scope)
        else:
            should_fix = check_mask & (new_argmax != x_scope)
        if not should_fix.any():
            return x_scope, cc_scope, 0
        fix_conf = torch.gather(p, -1, new_argmax.unsqueeze(-1)).squeeze(-1)
        x_out = torch.where(should_fix, new_argmax, x_scope)
        cc_out = torch.where(should_fix, fix_conf, cc_scope)
        return x_out, cc_out, int(should_fix.sum().item())

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length
        block_mask = (x[:, s:e] == mask_id)
        num_tt = get_num_transfer_tokens(block_mask, steps_per_block)
        was_committed = (x[:, Lp:] != mask_id)

        # WARM STEP
        out = model(x, use_cache=True)
        past_kv = out.past_key_values
        nfe += 1

        replace_pos = torch.zeros_like(x, dtype=torch.bool)
        replace_pos[:, s:e] = True
        global_mask = (x == mask_id)
        global_mask[:, e:] = False

        quota0 = None if threshold is not None else num_tt[:, 0]
        x0, ti = get_transfer_index(
            out.logits, temperature, remasking, global_mask, x, quota0, threshold,
        )
        p_full = F.softmax(out.logits.to(torch.float64), dim=-1)
        nc_full = torch.gather(p_full, -1, x0.unsqueeze(-1)).squeeze(-1)
        commit_conf = torch.where(ti, nc_full, commit_conf)
        x = torch.where(ti, x0, x)

        n_corr = 0
        if correction != "none":
            if scope == "global" and was_committed.any():
                wc_full = torch.zeros(B, total_len, dtype=torch.bool, device=x.device)
                wc_full[:, Lp:] = was_committed
                x, commit_conf, n_corr = _apply_correction(
                    out.logits, x, commit_conf, wc_full, correction, regret_threshold,
                )
            elif scope == "block":
                wc_blk = was_committed[:, nb * block_length:(nb + 1) * block_length]
                if wc_blk.any():
                    x_blk, cc_blk, n_corr = _apply_correction(
                        out.logits[:, s:e], x[:, s:e], commit_conf[:, s:e],
                        wc_blk, correction, regret_threshold,
                    )
                    x = torch.cat([x[:, :s], x_blk, x[:, e:]], dim=1)
                    commit_conf = torch.cat([commit_conf[:, :s], cc_blk, commit_conf[:, e:]], dim=1)

        corr_log.append({
            "block": nb, "type": "warm",
            "unmasked": int(ti.sum().item()), "corrections": n_corr,
            "remaining": int((x[:, s:e] == mask_id).sum().item()),
        })

        # REFINE STEPS
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            was_committed_blk = (x[:, s:e] != mask_id)
            logits_blk = model(
                x[:, s:e], past_key_values=past_kv,
                use_cache=True, replace_position=replace_pos,
            ).logits
            mask_blk = (x[:, s:e] == mask_id)
            quota_i = None if threshold is not None else num_tt[:, i]
            x0_blk, ti_blk = get_transfer_index(
                logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold,
            )
            p_blk = F.softmax(logits_blk.to(torch.float64), dim=-1)
            nc_blk = torch.gather(p_blk, -1, x0_blk.unsqueeze(-1)).squeeze(-1)
            cc_blk = commit_conf[:, s:e].clone()
            cc_blk = torch.where(ti_blk, nc_blk, cc_blk)
            blk_new = torch.where(ti_blk, x0_blk, x[:, s:e])

            n_corr = 0
            if correction != "none" and was_committed_blk.any():
                blk_new, cc_blk, n_corr = _apply_correction(
                    logits_blk, blk_new, cc_blk,
                    was_committed_blk, correction, regret_threshold,
                )
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            commit_conf = torch.cat([commit_conf[:, :s], cc_blk, commit_conf[:, e:]], dim=1)
            nfe += 1
            corr_log.append({
                "block": nb, "type": "refine",
                "unmasked": int(ti_blk.sum().item()), "corrections": n_corr,
                "remaining": int((x[:, s:e] == mask_id).sum().item()),
            })

    total_corr = sum(r["corrections"] for r in corr_log)
    return x, nfe, {"log": corr_log, "total_corrections": total_corr, "nfe": nfe}


# ---------------------------------------------------------------------------
# Prompt / eval helpers
# ---------------------------------------------------------------------------
def build_prompt(question, tokenizer):
    text = FEW_SHOT_EXAMPLES + f"\n\nQuestion: {question}\nAnswer:"
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(formatted)["input_ids"]
    return torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)


def strict_extract(text):
    """strict-match: only matches #### <number> pattern."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    return m.group(1).replace(",", "").strip() if m else None


def flexible_extract(text):
    """flexible-extract: find the last number in the text (lm_eval style)."""
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    if not matches:
        return None
    ans = matches[-1].replace(",", "").strip()
    if ans == "" or ans == "." or ans == "-":
        return None
    return ans


def _normalize_ans(ans):
    """Normalize answer string for comparison: strip trailing .0, etc."""
    if ans is None:
        return None
    ans = ans.strip()
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


def decode_and_evaluate(x, prompt, ref_answer, tokenizer):
    gen_ids = x[0, prompt.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    num_gen_tokens = int((gen_ids != MASK_ID).sum().item())
    for stop in ["Question:", "\n\nQuestion"]:
        if stop in gen_text:
            gen_text = gen_text.split(stop)[0]

    ref_ans = _normalize_ans(strict_extract(ref_answer))

    strict_ans = _normalize_ans(strict_extract(gen_text))
    strict_ok = strict_ans is not None and ref_ans is not None and strict_ans == ref_ans

    flex_ans = _normalize_ans(flexible_extract(gen_text))
    flex_ok = flex_ans is not None and ref_ans is not None and flex_ans == ref_ans

    return gen_text, num_gen_tokens, {
        "strict_ans": strict_ans, "flex_ans": flex_ans, "ref_ans": ref_ans,
        "strict_ok": strict_ok, "flex_ok": flex_ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="P0 correction experiment runner")
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--correction", default="none")
    parser.add_argument("--scope", default="block")
    parser.add_argument("--regret_threshold", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=100, help="0 = full dataset")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--save_details", action="store_true",
                        help="Save gen_text and corr_log per sample (more disk)")
    args = parser.parse_args()

    print(f"[{args.config_name}] Loading model on {os.environ.get('CUDA_VISIBLE_DEVICES','?')} ...")
    from transformers import AutoTokenizer, AutoConfig
    from model.modeling_llada import LLaDAModelLM

    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config,
    ).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    from datasets import load_dataset
    gsm8k = load_dataset("gsm8k", "main", split="test")
    limit = len(gsm8k) if args.limit == 0 else min(args.limit, len(gsm8k))

    print(f"[{args.config_name}] Building {limit} prompts ...")
    prompts = [build_prompt(gsm8k[i]["question"], tokenizer) for i in range(limit)]
    ref_answers = [gsm8k[i]["answer"] for i in range(limit)]

    print(f"[{args.config_name}] Running (threshold={args.threshold}, "
          f"correction={args.correction}, scope={args.scope}) ...")
    results = []
    total_nfe, total_corr, total_gen_tokens = 0, 0, 0
    t0 = time.time()

    for idx in tqdm(range(limit), desc=args.config_name):
        x, nfe, stats = generate_dualcache_corrected(
            model, prompts[idx],
            steps=args.steps, gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0., remasking="low_confidence", mask_id=MASK_ID,
            threshold=args.threshold,
            correction=args.correction,
            scope=args.scope,
            regret_threshold=args.regret_threshold,
        )
        gen_text, num_gen_tokens, eval_result = decode_and_evaluate(
            x, prompts[idx], ref_answers[idx], tokenizer,
        )
        entry = {
            "idx": idx,
            "strict_ok": eval_result["strict_ok"],
            "flex_ok": eval_result["flex_ok"],
            "strict_ans": eval_result["strict_ans"],
            "flex_ans": eval_result["flex_ans"],
            "ref_ans": eval_result["ref_ans"],
            "nfe": nfe,
            "corrections": stats["total_corrections"],
            "num_gen_tokens": num_gen_tokens,
        }
        if args.save_details:
            entry["gen_text"] = gen_text
            entry["corr_log"] = stats["log"]
        results.append(entry)
        total_nfe += nfe
        total_corr += stats["total_corrections"]
        total_gen_tokens += num_gen_tokens

    elapsed = time.time() - t0
    n = len(results)
    strict_acc = sum(r["strict_ok"] for r in results) / n
    flex_acc = sum(r["flex_ok"] for r in results) / n
    strict_se = (strict_acc * (1 - strict_acc) / n) ** 0.5
    flex_se = (flex_acc * (1 - flex_acc) / n) ** 0.5

    output = {
        "config_name": args.config_name,
        "results": results,
        "strict_acc": strict_acc,
        "flex_acc": flex_acc,
        "strict_se": strict_se,
        "flex_se": flex_se,
        "avg_nfe": total_nfe / n,
        "total_nfe": total_nfe,
        "avg_corrections": total_corr / n,
        "total_gen_tokens": total_gen_tokens,
        "tokens_per_second": total_gen_tokens / elapsed,
        "elapsed": elapsed,
        "n_samples": n,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(output, f)

    # lm_eval 风格输出
    print(f"\nTotal number of tokens generated: {total_gen_tokens}")
    print(f"Total time taken: {elapsed} seconds")
    print(f"Tokens per second: {total_gen_tokens / elapsed}")
    print(f"Total NFE is {total_nfe}")
    print(f"\n|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|")
    print(f"|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|")
    print(f"|gsm8k|      3|flexible-extract|     5|exact_match|↑  |{flex_acc:5.2f}|±  |{flex_se:.4f}|")
    print(f"|     |       |strict-match    |     5|exact_match|↑  |{strict_acc:5.2f}|±  |{strict_se:.4f}|")
    print(f"\n[{args.config_name}] → {args.output_path}")


if __name__ == "__main__":
    main()
