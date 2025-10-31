import torch
from __future__ import annotations
from typing import Tuple, List
import argparse
import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import json
from typing import Any, Dict

file_path = "test_results.json"
with open(file_path, "w") as file:
    file.write("")

def print_log_to_txt(log_message: str):
    with open(file_path, "a") as file:
        file.write(log_message + "\n")

def shortlist_from_logits(
    logits: torch.Tensor,
    top_k: int = 32,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)

    probs = log_probs.exp()
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    if top_p < 1.0:
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumsum <= top_p
        if not torch.any(nucleus_mask):
            nucleus_mask[0] = True
        kept_idx_sorted = sorted_idx[nucleus_mask]
    else:
        kept_idx_sorted = sorted_idx

    if top_k is not None and top_k > 0:
        kept_idx_sorted = kept_idx_sorted[:top_k]

    if eos_token_id is not None:
        if (kept_idx_sorted != eos_token_id).all():
            kept_idx_sorted = torch.cat([kept_idx_sorted, torch.tensor([eos_token_id], device=kept_idx_sorted.device)])

    kept_logp = log_probs.index_select(dim=-1, index=kept_idx_sorted)
    print_log_to_txt("probs:- " str(kept_logp))
    print_log_to_txt("ids:- " str(kept_idx_sorted))

    return kept_idx_sorted, kept_logp

@torch.no_grad()
def critic_values_for_candidates(
    critic: AutoModelForCausalLMWithValueHead,
    prefix_ids: torch.Tensor,
    cand_token_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    K = cand_token_ids.size(0)
    prefix = prefix_ids.repeat(K, 1)
    cand_col = cand_token_ids.view(K, 1)
    seqs = torch.cat([prefix, cand_col], dim=1).to(device)
    attn = torch.ones_like(seqs, dtype=torch.long, device=device)
    out = critic(input_ids=seqs, attention_mask=attn, return_dict=True)
    values = out.logits.squeeze(-1)
    print("Critic logits:- ", values)
    return values

def print_candidates_table(
    tokenizer,
    cand_ids: torch.Tensor,
    base_logp: torch.Tensor,
    values: torch.Tensor,
    combined: torch.Tensor,
    max_rows: int = 10,
):
    K = cand_ids.size(0)
    header = f"{'#':<3}{'TokenId':>8}{'BaseLogP':>12}{'Value':>10}{'Combined':>12}   Token(text)"
    print_log_to_txt(header)
    print_log_to_txt("-" * len(header))
    order = torch.argsort(combined, descending=True)
    rows = min(K, max_rows)
    for r in range(rows):
        i = order[r].item()
        tid = cand_ids[i].item()
        tok_text = repr(tokenizer.decode([tid], skip_special_tokens=True).replace("\n", "\\n"))
        print_log_to_txt(f"{r:<3}{tid:>8}{base_logp[i].item():>12.3f}{values[i].item():>10.3f}{combined[i].item():>12.3f}   {tok_text}")

def decode_loop(
    actor: AutoModelForCausalLM,
    critic: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    max_new_tokens: int = 128,
    top_k: int = 32,
    top_p: float = 1.0,
    lambda_val: float = 0.8,
    zscore_values: bool = True,
    mode: str = "greedy",
    temperature: float = 1.0,
    device: str = "cuda",
) -> str:
    actor.eval()
    critic.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    eos_id = tokenizer.eos_token_id
    total_gen = 0
    step = 0

    while total_gen < max_new_tokens:
        step += 1

        out = actor(input_ids=input_ids, attention_mask=attn, return_dict=True)
        logits = out.logits[:, -1, :]
        logits = logits.squeeze(0)

        cand_ids, base_logp = shortlist_from_logits(
            logits, top_k=top_k, top_p=top_p, eos_token_id=eos_id
        )

        vals = critic_values_for_candidates(
            critic=critic, prefix_ids=input_ids, cand_token_ids=cand_ids, device=input_ids.device
        )

        # base_logp = base_logp.float()
        # vals = vals.float()
        v = vals - vals.mean()
        scale = base_logp.std().clamp_min(1e-6) / v.std().clamp_min(1e-6)
        if zscore_values:
            vmean, vstd = vals.mean(), vals.std()
            vals = (vals - vmean) / (vstd + 1e-6)
        # print("scale:- ", scale)
        combined = base_logp + lambda_val * vals

        print_log_to_txt(f"\n--- Step {step} | K={cand_ids.numel()} | λ={lambda_val} | mode={mode} ---")
        cur_txt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print_log_to_txt("[Prefix]", repr(cur_txt[-160:] if len(cur_txt) > 160 else cur_txt))
        print_candidates_table(tokenizer, cand_ids, base_logp, vals, combined, max_rows=10)

        if mode == "greedy":
            best_idx = torch.argmax(combined).item()
            next_token = cand_ids[best_idx]
        elif mode == "sample":
            comb = combined / max(temperature, 1e-6)
            probs = F.softmax(comb, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_token = cand_ids[next_idx]
        else:
            raise ValueError("mode must be 'greedy' or 'sample'")

        input_ids = torch.cat([input_ids, next_token.view(1, 1).to(device)], dim=1)
        attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
        total_gen += 1

        if eos_id is not None and next_token.item() == eos_id:
            print_log_to_txt("\n[EOS reached] stopping.")
            break

        if total_gen >= max_new_tokens:
            print_log_to_txt("\n[max_new_tokens reached] stopping.")
            break

    final = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print_log_to_txt("\n[Final Output]\n", final)
    return final

def main():
    args = {
        "actor_path": "meta-llama/Llama-3.2-3B-Instruct",
        "critic_path": "samhitha2601/llama-3.2-3b-gsm8k-ppo-verl-step12",
        "max_new_tokens": 512,
        "top_k": 64,
        "top_p": 1.0,
        "lambda_val": 0.8,
        "zscore_values": False,
        "mode": "greedy",
        "temperature": 0.8,
        "dtype": "bfloat16",
    }

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args['dtype']]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args['actor_path'], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    actor = AutoModelForCausalLM.from_pretrained(
        args['actor_path'], torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )

    try:
        actor.config.attn_implementation = "sdpa"
    except Exception:
        pass

    cfg = AutoConfig.from_pretrained(args['critic_path'])
    cfg.num_labels = 1
    cfg.problem_type = "regression"
    cfg.pad_token_id = tok.pad_token_id

    critic = AutoModelForSequenceClassification.from_pretrained(
        args['critic_path'],
        config=cfg,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    try:
        if hasattr(critic, "pretrained_model"):
            critic.pretrained_model.config.attn_implementation = "sdpa"
        else:
            critic.config.attn_implementation = "sdpa"
    except Exception:
        pass

    print_log_to_txt("Loading gsm8k test dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    results = []
    print_log_to_txt(f"Starting benchmark on {len(dataset)} samples...")

    for i, sample in enumerate(dataset):
        prompt = sample['question']
        
        print_log_to_txt(f"Processing sample {i+1}/{len(dataset)}")
        print_log_to_txt(f"Question:- {prompt}")
        generation = decode_loop(
            actor=actor,
            critic=critic,
            tokenizer=tok,
            user_prompt=prompt,
            max_new_tokens=args['max_new_tokens'],
            top_k=args['top_k'],
            top_p=args['top_p'],
            lambda_val=args['lambda_val'],
            zscore_values=args['zscore_values'],
            mode=args['mode'],
            temperature=args['temperature'],
            device=device,
        )
        
        results.append({
            "question": prompt,
            "ground_truth_answer": sample['answer'],
            "generated_answer": generation
        })
        print_log_to_txt(f"Ground Truth Answer:- {sample['answer']}")
        print_log_to_txt(f"Generated Answer:- {generation}")
    output_filename = "gsm8k_results.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print_log_to_txt(f"Benchmark complete. Results saved to {output_filename}")

    del actor, critic
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
