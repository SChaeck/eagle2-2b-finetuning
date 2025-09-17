import io
import os
import copy
import time
import torch
import requests
import random
import numpy as np
import wandb
import gc
import time
import re
import wandb
from trl import GRPOConfig
from eagle2_trl_grpo_trainer import Eagle2TRLGRPOTrainer
from eagle2_data_collator import Eagle2DataCollator

from PIL import Image
import torch.distributed as dist
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset, Dataset
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from trl import SFTConfig, SFTTrainer
from eagle2_trl_sft_trainer import Eagle2TRLSFTTrainer
from eagle2_data_collator import Eagle2DataCollator
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional

# NOTE: 
# For GPU memory optimization, before starting, go to the Hugging Face cache and set "max_dynamic_tiles" from 12 to 1 in the config.json and preprocessor_config.json file.

existing_processed_datasets = False

seed = 7777
test_size = 0.1

newline_between_blocks = True # This for the newline between blocks 

dataset_path = "/home/compu/test_suchae/eagle2-2b-finetuning/no_task2_dataset.jsonl"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        
def ealge_format_multiview_data(sample):
    prompt_blocks = sample["prompt_blocks"]
    # Change the value of 'type' from 'image_url' to 'image' in dicts
    for block in prompt_blocks:
        if isinstance(block, dict) and block.get("type") == "image_url":
            block["type"] = "image"    
    
    conversation = [
        {
            "role": "system",
            "content":  [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": prompt_blocks,
        },
    ]
    return {
        "prompt": conversation,
        "solution": sample["ground_truth_answer"]
    }
    

def main():
    set_seed(seed)
      
    if existing_processed_datasets:
        train_dataset = Dataset.load_from_disk("./Multiview_processed_datasets/train_dataset")
        eval_dataset = Dataset.load_from_disk("./Multiview_processed_datasets/eval_dataset")

    else:
        # Load JSONL as Hugging Face dataset
        dataset = load_dataset("json", data_files=dataset_path)

        # Split dataset into train and eval
        dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        train_dataset = dataset["train"] 
        eval_dataset = dataset["test"]
        
        train_dataset = [ealge_format_multiview_data(sample) for sample in train_dataset]
        train_dataset = Dataset.from_list(train_dataset)

        eval_dataset = Dataset.from_list([ealge_format_multiview_data(sample) for sample in eval_dataset])
        
        train_dataset.save_to_disk("./Multiview_processed_datasets/train_dataset")
        eval_dataset.save_to_disk("./Multiview_processed_datasets/eval_dataset")
            
    model_id = "nvidia/Eagle2-2B"

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map={'': torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"

    # Configure LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
        use_dora=True,
        init_lora_weights="gaussian",
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    def to_text(completion):
        if isinstance(completion, str):
            return completion
        # completion == list of messages [{'role':..., 'content':...}]
        parts = []
        for msg in completion:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # multimodal: [{'type':'text','text':...}, {'type':'image',...}]
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                        parts.append(item["text"])
        return "".join(parts)

    def format_reward(completions, **kwargs):
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        texts = [to_text(c) for c in completions]
        # print(f"[DEBUG] texts: {texts}")
        matches = [re.match(pattern, t, re.DOTALL | re.MULTILINE) for t in texts]
        return [1.0 if m else 0.0 for m in matches]
    
    def accuracy_reward(completions, **kwargs):
        import re
        golds = kwargs.get("solution", [])
        # print(f'[_accuracy_reward] golds: {golds}')
        texts = [to_text(c) for c in completions]
        rewards = []
        for t, g in zip(texts, golds):
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", t, re.DOTALL | re.IGNORECASE)
            pred = (m.group(1).strip() if m else "")
            rewards.append(float(pred.lower() == str(g).lower()))
        return rewards    


    # Initialize wandb with dongguk university team
    wandb.init(
        entity="schaeck-dongguk-university",  # Use dongguk university team
        project="eagle2-2b-trl-grpo-finetuning"
    )

    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        output_dir="eagle2-2b-trl-grpo-Multitask",  # Directory to save the model
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        learning_rate=1e-5,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward
        num_train_epochs=10,
        per_device_train_batch_size=2,  # Batch size for training (reduced for memory)
        per_device_eval_batch_size=2,  # Batch size for evaluation (reduced for memory)
        gradient_accumulation_steps=1,  # Must be divisible by num_generations for GRPO
        bf16=True,
        # Parameters that control the data preprocessing
        max_completion_length=1024,  # default: 256
        num_generations=2,  # default: 8
        max_prompt_length=3060,
        # Parameters related to reporting and saving
        report_to=["wandb"],
        logging_steps=5,
        push_to_hub=True,
        eval_steps=60,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",
        save_steps=60,
        # scale_rewards=False
    )
    

    processor.tokenizer.pad_token = "<|endoftext|>"
    processor.pad_token = "<|endoftext|>"
    processor.tokenizer.pad_token_id = 151643
    processor.pad_token_id = 151643
    processor.bos_token_id = processor.tokenizer.bos_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    # eagle2_data_collator = Eagle2DataCollator(processor.tokenizer)

    trainer = Eagle2TRLGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    # trainer.push_to_hub(dataset_name=dataset_id)
    
if __name__ == "__main__":
    main()