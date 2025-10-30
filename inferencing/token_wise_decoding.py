#!/usr/bin/env python
# tokenwise_vas_verl.py
# Token-wise Value-Augmented Sampling (VAS) with VERL actor/critic HuggingFace checkpoints.
#
# At each decoding step:
#  1) Get base logits over next token from the ACTOR.
#  2) Build a shortlist (top-k / nucleus) of candidate tokens.
#  3) For each candidate token 'a', score V(prefix ⊕ a) using the CRITIC.
#  4) Combine: score(a) = log p_actor(a|prefix) + λ * normalize(V(prefix ⊕ a))
#  5) Pick the next token via greedy argmax or sampling from the combined distribution.
#
# Notes:
#  - Uses chat template for Llama Instruct.
#  - Critic is loaded via TRL's AutoModelForCausalLMWithValueHead (value at last token).
#  - For simplicity/robustness this recomputes the actor forward each step (no KV cache).
#    You can add caching later for speed.

from __future__ import annotations
from typing import Tuple, List
import argparse
import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
from trl import AutoModelForCausalLMWithValueHead

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
    print(out.logits)
    print(out.logits.shape)
    values = out.logits.squeeze(-1)
    print(values)
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
    print(header)
    print("-" * len(header))
    order = torch.argsort(combined, descending=True)
    rows = min(K, max_rows)
    for r in range(rows):
        i = order[r].item()
        tid = cand_ids[i].item()
        tok_text = repr(tokenizer.decode([tid], skip_special_tokens=True).replace("\n", "\\n"))
        print(f"{r:<3}{tid:>8}{base_logp[i].item():>12.3f}{values[i].item():>10.3f}{combined[i].item():>12.3f}   {tok_text}")

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


        # if zscore_values:
        #     vmean, vstd = vals.mean(), vals.std()
        #     vals = (vals - vmean) / (vstd + 1e-6)

        combined = base_logp + lambda_val * vals

        print(f"\n--- Step {step} | K={cand_ids.numel()} | λ={lambda_val} | mode={mode} ---")
        cur_txt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print("[Prefix]", repr(cur_txt[-160:] if len(cur_txt) > 160 else cur_txt))
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
            print("\n[EOS reached] stopping.")
            break

        if total_gen >= max_new_tokens:
            print("\n[max_new_tokens reached] stopping.")
            break

    final = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\n[Final Output]\n", final)
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor_path", required=True, help="Path to VERL actor 'huggingface' folder")
    ap.add_argument("--critic_path", required=True, help="Path to VERL critic 'huggingface' folder")
    ap.add_argument("--prompt", required=True, help="User prompt text")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--lambda_val", type=float, default=0.8)
    ap.add_argument("--zscore_values", action="store_true")
    ap.add_argument("--mode", choices=["greedy","sample"], default="greedy")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dtype", choices=["bfloat16","float16","float32"], default="bfloat16")
    args = ap.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.actor_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    actor = AutoModelForCausalLM.from_pretrained(
        args.actor_path, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )

    try:
        actor.config.attn_implementation = "sdpa"
    except Exception:
        pass
    

    cfg = AutoConfig.from_pretrained(args.critic_path)
    cfg.num_labels = 1
    cfg.problem_type = "regression"
    cfg.pad_token_id = tok.pad_token_id

    critic = AutoModelForSequenceClassification.from_pretrained(
        args.critic_path,
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

    decode_loop(
        actor=actor,
        critic=critic,
        tokenizer=tok,
        user_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        lambda_val=args.lambda_val,
        zscore_values=args.zscore_values,
        mode=args.mode,
        temperature=args.temperature,
        device=device,
    )

if __name__ == "__main__":
    main()
