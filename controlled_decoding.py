#!/usr/bin/env python3
"""
controlled_decoding_blockwise.py

Blockwise Controlled Decoding / Value-Augmented Sampling
 - Base LM: frozen, provides candidate continuations.
 - Value LM: scalar prefix scorer (critic). Here we add a simple linear head.

At each step:
  1) Show current prefix (since prompt).
  2) Sample K blocks (M tokens each) from base LM (batched).
  3) For each block, compute:
       - Base LM block log-prob (sum over the M new tokens)
       - Value score V(prefix + block)
       - Combined score: base_logp + λ * V
  4) Print a neat table and pick the best block by combined score.
  5) Append best block and continue.

This is a demo: the value head is random. Replace ValueLM with your trained scorer.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------
# Value model: LM + scalar head
# ------------------------------
class ValueLM(nn.Module):
    """Wraps an LM with a scalar value head (simulating a trained prefix scorer)."""

    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        hidden_dim = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_dim, 1)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]            # (B, T, H)
        pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        val = self.value_head(pooled)                  # (B, 1)
        return val.squeeze(-1)                         # (B,)


# ------------------------------------------
# Utility: sum base log-prob over M new toks
# ------------------------------------------
@torch.no_grad()
def block_logprobs(
    base_model: AutoModelForCausalLM,
    seqs: torch.Tensor,
    prefix_len: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute sum of log-probs assigned by base_model to the continuation part
    (positions prefix_len .. T-1), teacher-forcing over each candidate sequence.

    Args:
      seqs: (B, T) candidate full sequences (prefix + block)
      prefix_len: length of the shared prefix in tokens
      attention_mask: (B, T) mask (1 for real tokens)
    Returns:
      (B,) tensor of summed log-probs over continuation tokens
    """
    # Shifted LM loss trick:
    # log p(x_{t} | x_{<t}) is from logits at positions t-1
    out = base_model(seqs[:, :-1], attention_mask=attention_mask[:, :-1] if attention_mask is not None else None)
    logits = out.logits  # (B, T-1, V)
    logp = torch.log_softmax(logits, dim=-1)

    # Gather log-probs of the actual next tokens
    next_tokens = seqs[:, 1:]                        # (B, T-1)
    token_logp = logp.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

    # Only sum continuation part: indices prefix_len .. T-1-1 == T-2 (since we shifted)
    cont_mask = torch.zeros_like(token_logp, dtype=torch.bool)
    cont_mask[:, prefix_len-1:] = True  # because token_logp[t] corresponds to x_{t+1} given prefix up to t

    # Respect attention_mask if provided
    if attention_mask is not None:
        # For token_logp position t (predict x_{t+1}), require both t and t+1 to be valid
        am = attention_mask
        valid_mask = (am[:, :-1] * am[:, 1:]).bool()
        cont_mask = cont_mask & valid_mask

    cont_logp_sum = (token_logp * cont_mask).sum(dim=1)  # (B,)
    return cont_logp_sum


# ------------------------------------------
# Pretty printers
# ------------------------------------------
def show_prefix(prefix_text: str, clip: int = 160) -> None:
    trimmed = prefix_text[-clip:] if len(prefix_text) > clip else prefix_text
    print(f"[Current Prefix] …{trimmed!r}" if len(prefix_text) > clip else f"[Current Prefix] {trimmed!r}")

def print_candidates_table(rows: List[Tuple[int, float, float, float, str]], max_cont_preview: int = 80) -> None:
    """
    rows: list of (idx, base_logp, value, combined, continuation_text)
    """
    header = f"{'Cand#':<6}{'BaseLogP':>12}{'Value':>10}{'Combined':>12}   Continuation"
    print(header)
    print("-"*len(header))
    for idx, blp, val, comb, cont in rows:
        preview = cont.replace("\n", "\\n")[:max_cont_preview]
        print(f"{idx:<6}{blp:>12.3f}{val:>10.3f}{comb:>12.3f}   {preview!r}")


# ------------------------------------------
# Blockwise controlled decoding
# ------------------------------------------
@torch.no_grad()
def controlled_decode_blockwise(
    base_model,
    value_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    block_size: int = 4,
    num_candidates: int = 4,
    lambda_val: float = 0.7,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:

    base_model.to(device).eval()
    value_model.to(device).eval()

    enc = tokenizer(prompt, return_tensors="pt")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # safe default for causal LMs

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prefix_len = input_ids.size(1)

    print(f"\n[Prompt] {prompt}\n" + "="*90)

    total_generated = 0
    step = 0

    while total_generated < max_new_tokens:
        step += 1
        # Show current prefix since the prompt start
        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        show_prefix(full_text)

        # Batch-generate K candidate blocks of size M
        in_batch = input_ids.repeat(num_candidates, 1)
        am_batch = attention_mask.repeat(num_candidates, 1)

        cand = base_model.generate(
            input_ids=in_batch,
            attention_mask=am_batch,
            max_new_tokens=block_size,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )  # (K, prefix_len + M)

        # Compute base block log-probs in batch
        am_cand = torch.ones_like(cand, dtype=torch.long, device=device)
        base_logps = block_logprobs(base_model, cand, prefix_len=prefix_len, attention_mask=am_cand)  # (K,)

        # Compute value scores on each full prefix+block sequence
        vals = value_model(cand, attention_mask=am_cand)  # (K,)

        # Combine (log-space): base_logp + λ * value
        combined = base_logps + lambda_val * vals  # (K,)

        # Prepare readable continuations
        cont_texts = []
        for i in range(num_candidates):
            cont = tokenizer.decode(cand[i, prefix_len:], skip_special_tokens=True)
            cont_texts.append(cont)

        # Print table
        rows = [(i, base_logps[i].item(), vals[i].item(), combined[i].item(), cont_texts[i])
                for i in range(num_candidates)]
        print(f"\n--- Step {step}: K={num_candidates}, M={block_size}, λ={lambda_val} ---")
        print_candidates_table(rows)

        # Pick best
        best_idx = torch.argmax(combined).item()
        best_block = cand[best_idx].unsqueeze(0)  # (1, prefix+M)
        best_cont = cont_texts[best_idx]
        print(f"\n[Chosen] cand#{best_idx}  continuation={best_cont!r}")

        # Update running prefix
        new_tokens = best_block[0, prefix_len:]
        n_new = new_tokens.size(0)
        total_generated += n_new
        input_ids = best_block
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        prefix_len = input_ids.size(1)

        # Stop if EOS appeared in the new block
        if (new_tokens == tokenizer.eos_token_id).any():
            print("\n[EOS reached] stopping.")
            break

        if total_generated >= max_new_tokens:
            print("\n[max_new_tokens reached] stopping.")
            break

    # Final text (skip special tokens)
    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\n[Final Output]\n" + final_text + "\n" + "="*90)
    return final_text


# ------------------------------
# Main
# ------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_model_name = "meta-llama/Llama-3.2-3B"   # base LM (policy)
    value_model_name = "meta-llama/Llama-3.2-1B"  # value LM (critic)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    value_model = ValueLM(value_model_name)  # replace with your trained scorer

    prompt = "Explain why the moon appears brighter at night compared to during the day."
    _ = controlled_decode_blockwise(
        base_model=base_model,
        value_model=value_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=32,
        block_size=4,
        num_candidates=3,
        lambda_val=0.6,       # weight of value vs base log-prob
        temperature=0.8,
        top_p=0.9,
        device=device,
    )


if __name__ == "__main__":
    main()
