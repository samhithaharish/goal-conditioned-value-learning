"""
VAS Inference for GSM8K - FIXED VERSION
========================================

Fixed the past_key_values compatibility issue with newer transformers versions.
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from trl.models import AutoModelForCausalLMWithValueHead
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import re
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================================
# VAS Logits Processor - FIXED VERSION
# ============================================================================

class VASLogitsProcessor(LogitsProcessor):
    """Value Augmented Sampling - Beam search with value guidance - FIXED"""
    
    def __init__(
        self,
        value_model: AutoModelForCausalLMWithValueHead,
        beta: float = 0.1,
        topk: int = 20,
        topk_per_device_batch_size: int = 1,
        value_model_tokenizer: Optional[AutoTokenizer] = None,
        policy_tokenizer: Optional[AutoTokenizer] = None,
    ):
        super().__init__()
        self.value_model = value_model
        self.policy_tokenizer = policy_tokenizer
        self.value_model_tokenizer = value_model_tokenizer or policy_tokenizer
        self.beta = beta
        self.topk = topk
        self.topk_per_device_batch_size = topk_per_device_batch_size
        
        assert self.topk > 0, 'topk must be larger than zero'
        
        self.last_input_ids = None
        self.past_key_values = None
    
    def _convert_to_cache_format(self, past_key_values):
        """
        Convert tuple format past_key_values to Cache format if needed.
        This fixes compatibility with newer transformers versions.
        """
        if past_key_values is None:
            return None
        
        # Check if it's already in the new Cache format
        if hasattr(past_key_values, 'get_seq_length'):
            return past_key_values
        
        # If it's a tuple, try to convert to DynamicCache
        if isinstance(past_key_values, tuple):
            try:
                from transformers.cache_utils import DynamicCache
                cache = DynamicCache()
                for layer_past in past_key_values:
                    if isinstance(layer_past, tuple) and len(layer_past) == 2:
                        cache.update(
                            layer_past[0],  # key
                            layer_past[1],  # value
                            layer_idx=len(cache.key_cache)
                        )
                return cache
            except ImportError:
                # If DynamicCache not available, return None to disable KV caching
                return None
        
        return past_key_values
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply VAS decoding to logits - FIXED VERSION"""
        augmented_outputs = torch.clone(scores)
        batch_size = input_ids.shape[0]
        orig_input_ids = input_ids
        
        # Get value predictions
        try:
            if (self.last_input_ids is not None and 
                input_ids[0, :-1].shape == self.last_input_ids.shape and 
                torch.all(input_ids[0, :-1] == self.last_input_ids)):
                # Reuse cached past key values
                # Convert to cache format before passing
                converted_past = self._convert_to_cache_format(self.past_key_values)
                _, _, _, past_key_values = self.value_model(
                    input_ids,
                    past_key_values=converted_past,
                    return_past_key_values=True
                )
            else:
                # Compute fresh
                _, _, _, past_key_values = self.value_model(
                    input_ids,
                    return_past_key_values=True
                )
            
            self.past_key_values = past_key_values
            self.last_input_ids = input_ids[0, :].clone()
        except Exception as e:
            # If there's any error with KV caching, fall back to no caching
            print(f"⚠️  Warning: KV cache error, disabling caching: {e}")
            self.past_key_values = None
            self.last_input_ids = None
        
        # Get top-k token indices
        values = torch.zeros_like(scores, device=scores.device)
        topk_ids = torch.topk(scores, self.topk, dim=-1).indices
        
        # Score top-k tokens in batches
        for i in range(0, topk_ids.shape[1], self.topk_per_device_batch_size):
            curr_topk_ids = topk_ids[:, i:i + self.topk_per_device_batch_size]
            
            # Expand input_ids to include each top-k token
            curr_input_ids = orig_input_ids.unsqueeze(1).repeat(
                1, curr_topk_ids.shape[1], 1
            )
            curr_input_ids = torch.cat(
                [curr_input_ids, curr_topk_ids.unsqueeze(-1)], dim=-1
            )
            curr_input_ids = curr_input_ids.reshape(
                (batch_size * self.topk_per_device_batch_size, -1)
            )
            
            # Prepare past_key_values for this batch
            expanded_past = None
            if self.past_key_values is not None:
                try:
                    # Handle tuple format
                    if isinstance(self.past_key_values, tuple):
                        expanded_past = tuple(
                            (
                                t1.repeat(curr_topk_ids.shape[1], 1, 1, 1),
                                t2.repeat(curr_topk_ids.shape[1], 1, 1, 1)
                            )
                            for t1, t2 in self.past_key_values
                        )
                        # Convert to cache format
                        expanded_past = self._convert_to_cache_format(expanded_past)
                    # Handle Cache format
                    elif hasattr(self.past_key_values, 'get_seq_length'):
                        # For Cache objects, we need to handle differently
                        # For simplicity, we'll just not use past for expanded sequences
                        expanded_past = None
                except Exception as e:
                    print(f"⚠️  Warning: Error expanding past_key_values: {e}")
                    expanded_past = None
            
            # Get values for these tokens
            try:
                _, _, value, _ = self.value_model(
                    curr_input_ids,
                    past_key_values=expanded_past,
                    return_past_key_values=True
                )
            except Exception as e:
                # Fallback: call without past_key_values
                print(f"⚠️  Warning: Value model error, using fresh forward pass: {e}")
                _, _, value, _ = self.value_model(
                    curr_input_ids,
                    past_key_values=None,
                    return_past_key_values=True
                )
            
            # Reshape values back
            value = value.reshape(
                (batch_size, self.topk_per_device_batch_size, -1)
            )[:, :, -1]
            
            # Store values for top-k tokens
            values = values.scatter_(1, curr_topk_ids, value)
        
        # Normalize values (center around mean)
        values = values.scatter_(
            1,
            topk_ids,
            values.gather(1, topk_ids) - values.gather(1, topk_ids).mean(-1, keepdim=True)
        )
        
        # Apply value guidance to logits
        augmented_outputs = augmented_outputs + self.beta * values
        
        return augmented_outputs


# ============================================================================
# Dataset for YOUR Specific Format
# ============================================================================

class GSM8KDataset(Dataset):
    """
    GSM8K dataset that handles YOUR specific format.
    Uses: question, gold_answer (or gold_freeform)
    """
    
    def __init__(self, data_source, tokenizer, max_length=512):
        """
        Args:
            data_source: File path (JSONL) or list of dicts
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if isinstance(data_source, str):
            print(f"📁 Loading from file: {data_source}")
            self.data = self._load_jsonl(data_source)
        else:
            self.data = data_source
        
        print(f"✓ Loaded {len(self.data)} examples")
    
    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            print(f" Error loading JSONL: {e}")
            return []
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract question (always 'question' in your format)
        question = item.get('question', "")
        if not isinstance(question, str):
            question = str(question)
        
        # Extract answer - try gold_answer first, then gold_freeform
        answer = item.get('gold_answer', item.get('gold_freeform', ""))
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Extract numeric answer
        numeric_answer = self._extract_number(answer)
        
        # Create prompt
        prompt = f"""Question: {question}

Let me solve this step by step:
"""
        
        return {
            'prompt': prompt,
            'question': question,
            'full_answer': answer,
            'numeric_answer': numeric_answer,
            'idx': idx
        }
    
    @staticmethod
    def _extract_number(answer_str: str) -> Optional[float]:
        """Extract numeric answer from gold_answer/gold_freeform"""
        if not answer_str:
            return None
        
        # Try to find answer after ####
        if '####' in answer_str:
            answer_str = answer_str.split('####')[-1].strip()
        
        # Find all numbers (including decimals and negatives)
        numbers = re.findall(r'-?\d+\.?\d*', answer_str)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                return None
        return None


# ============================================================================
# Inference Engine
# ============================================================================

class VASInferenceEngine:
    """VAS Inference Engine with fixed past_key_values handling"""
    
    def __init__(
        self,
        actor_path: str,
        critic_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize models"""
        self.device = device
        
        print("\n" + "="*80)
        print("LOADING MODELS")
        print("="*80 + "\n")
        
        # Load tokenizer
        print("[1/3] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(actor_path)
        
        # IMPORTANT: Set padding side to LEFT for decoder-only models
        self.tokenizer.padding_side = "left"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("      ✓ Tokenizer loaded (padding_side='left')\n")
        
        # Load actor
        print("[2/3] Loading actor model...")
        print(f"      Path: {actor_path}")
        self.actor = AutoModelForCausalLM.from_pretrained(
            actor_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.actor.eval()
        print("      ✓ Actor model loaded\n")
        
        # Load critic
        print("[3/3] Loading critic model...")
        print(f"      Path: {critic_path}")
        self.critic = AutoModelForCausalLMWithValueHead.from_pretrained(
            critic_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.critic.eval()
        print("      ✓ Critic model loaded\n")
        
        print("="*80)
        print("✓ ALL MODELS LOADED SUCCESSFULLY")
        print("="*80 + "\n")
    
    def generate_with_vas(
        self,
        prompts: List[str],
        beta: float = 0.1,
        topk: int = 20,
        num_beams: int = 4,
        max_new_tokens: int = 512,
    ) -> List[str]:
        """Generate with VAS"""
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Create VAS processor
        vas_processor = VASLogitsProcessor(
            value_model=self.critic,
            beta=beta,
            topk=topk,
            topk_per_device_batch_size=1,
            value_model_tokenizer=self.tokenizer,
            policy_tokenizer=self.tokenizer,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.actor.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                logits_processor=LogitsProcessorList([vas_processor]),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove prompt
            prompt_length = inputs['input_ids'][i].shape[0]
            generated = output[prompt_length:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def run_inference(
        self,
        dataset: GSM8KDataset,
        batch_size: int = 8,
        beta: float = 0.1,
        topk: int = 20,
        num_beams: int = 4,
        max_new_tokens: int = 512,
    ) -> List[Dict]:
        """Run inference on dataset"""
        print("\n" + "="*80)
        print("RUNNING INFERENCE")
        print("="*80)
        print(f"Total examples: {len(dataset)}")
        print(f"Batch size:     {batch_size}")
        print(f"Num beams:      {num_beams}")
        print(f"Beta:           {beta}")
        print(f"Top-K:          {topk}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
        
        all_results = []
        
        for batch in tqdm(dataloader, desc="Inference Progress"):
            # Generate
            generated_text = self.generate_with_vas(
                prompts=batch['prompt'],
                beta=beta,
                topk=topk,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )
            
            # Collect results
            for i, text in enumerate(generated_text):
                result = {
                    'prompt': batch['prompt'][i],
                    'generated_text': text,
                    'question': batch['question'][i],
                    'numeric_answer': batch['numeric_answer'][i],
                    'original_answer': batch['full_answer'][i],
                }
                all_results.append(result)
        
        return all_results
    
    @staticmethod
    def _collate_fn(batch):
        """Custom collate function"""
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            result[key] = values
        return result


# ============================================================================
# Evaluation
# ============================================================================

class AnswerEvaluator:
    """Evaluate generated answers"""
    
    @staticmethod
    def extract_answer(text: str) -> Optional[float]:
        """Extract numeric answer from text"""
        if not text:
            return None
        
        text = text.lower()
        
        patterns = [
            r'(?:answer|result|equals?|is|gives?|total)\s*(?:is\s+)?(?:=|:)?\s*(-?\d+\.?\d*)',
            r'(-?\d+\.?\d*)\s*(?:is the answer|is correct)',
            r'therefore[,.]?\s*(-?\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        
        # Fallback: last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass
        
        return None
    
    @staticmethod
    def is_correct(
        generated_answer: Optional[float],
        ground_truth: Optional[float],
        tolerance: float = 1e-3
    ) -> bool:
        """Check if answer is correct"""
        if generated_answer is None or ground_truth is None:
            return False
        return abs(generated_answer - ground_truth) < tolerance
    
    @staticmethod
    def evaluate(results: List[Dict]) -> Dict:
        """Evaluate results"""
        correct = 0
        total = len(results)
        
        for result in results:
            generated_answer = AnswerEvaluator.extract_answer(result['generated_text'])
            ground_truth = result['numeric_answer']
            
            if AnswerEvaluator.is_correct(generated_answer, ground_truth):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }


# ============================================================================
# Main
# ============================================================================

def main(
    actor_path: str = "samhitha2601/llama-3.2-3b-gsm8k-ppo-verl-step12",
    critic_path: str = "samhitha2601/llama3-gsm8k-critic",
    data_path: str = "data/gsm8k/test.jsonl",
    batch_size: int = 8,
    num_beams: int = 4,
    beta: float = 0.1,
    topk: int = 20,
    output_dir: str = "gsm8k_results",
):
    """Main inference pipeline"""
    
    print("\n" + "="*80)
    print("🎬 STARTING GSM8K VAS INFERENCE - FIXED VERSION")
    print("="*80)
    
    print("\nConfiguration:")
    print(f"  Actor:      {actor_path}")
    print(f"  Critic:     {critic_path}")
    print(f"  Data:       {data_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num beams:  {num_beams}")
    print(f"  Beta:       {beta}")
    print(f"  Top-K:      {topk}")
    print(f"  Output:     {output_dir}")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"\nError: Data file not found at {data_path}")
        print("Please ensure the path is correct")
        return
    
    # Initialize engine
    engine = VASInferenceEngine(
        actor_path=actor_path,
        critic_path=critic_path,
    )
    
    # Load dataset
    print("\n📁 Loading dataset...")
    dataset = GSM8KDataset(
        data_source=data_path,
        tokenizer=engine.tokenizer,
        max_length=512,
    )
    
    # Run inference
    results = engine.run_inference(
        dataset=dataset,
        batch_size=batch_size,
        beta=beta,
        topk=topk,
        num_beams=num_beams,
    )
    
    # Evaluate
    print("\nEvaluating results...")
    metrics = AnswerEvaluator.evaluate(results)
    
    # Print results
    print("\n" + "="*80)
    print("Greedy Decoding RESULTS")
    print("="*80)
    print(f"✓ Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print("="*80)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "vas_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'config': {
                'actor_path': actor_path,
                'critic_path': critic_path,
                'batch_size': batch_size,
                'num_beams': num_beams,
                'beta': beta,
                'topk': topk,
            },
            'sample_results': results[:10],
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("✓ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VAS Inference for GSM8K - FIXED")
    parser.add_argument("--actor_path", type=str, default="samhitha2601/llama-3.2-3b-gsm8k-ppo-verl-step12")
    parser.add_argument("--critic_path", type=str, default="samhitha2601/llama3-gsm8k-critic")
    parser.add_argument("--data_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="gsm8k_results")
    
    args = parser.parse_args()
    
    main(
        actor_path=args.actor_path,
        critic_path=args.critic_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        beta=args.beta,
        topk=args.topk,
        output_dir=args.output_dir,
    )