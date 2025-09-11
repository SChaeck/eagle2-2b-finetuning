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

from PIL import Image
import torch.distributed as dist
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset, Dataset
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from trl import SFTConfig, SFTTrainer
from eagle2_trl_sft_trainer import Eagle2TRLSFTTrainer
from eagle2_data_collator import Eagle2DataCollator

# NOTE: 
# For GPU memory optimization, before starting, go to the Hugging Face cache and set "max_dynamic_tiles" from 12 to 1 in the config.json and preprocessor_config.json file.

existing_processed_datasets = False

seed = 7777
test_size = 0.05

newline_between_blocks = True # This for the newline between blocks 

dataset_path = "/home/compu/test_suchae/eagle2-2b-finetuning/multitask_dataset.jsonl"


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
            
    answer = sample["ground_truth_answer"]
    
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt_blocks
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }

def main():
    set_seed(seed)
      
    # # If distributed environment variables are not set, manually configure for single process
    # if 'RANK' not in os.environ:
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'  # Use any available port
    #     os.environ['RANK'] = '0'
    #     os.environ['WORLD_SIZE'] = '1'
        
    #     # Initialize backend ('nccl' if GPU is available, otherwise 'gloo')
    #     backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    #     dist.init_process_group(backend=backend, init_method='env://')
    
    if existing_processed_datasets:
        train_dataset = Dataset.load_from_disk("./Multiview_processed_datasets/train_dataset")
        eval_dataset = Dataset.load_from_disk("./Multiview_processed_datasets/eval_dataset")

    else:
        system_message = """You are a Vision Language Model designed to interpret and reason over multiple related chart images (multi-view).
    You will be provided with a set of chart images that together represent different perspectives, time points, or facets of the same data context.
    Your task is to analyze all provided images collectively and answer the user's query by integrating information across the views.
    Focus on delivering concise, accurate answers (typically a word, number, or short phrase) based on the combined visual information.
    Do not provide extra explanation unless specifically requested. Assume the user expects you to synthesize insights from all images."""

        # Load JSONL as Hugging Face dataset
        dataset = load_dataset("json", data_files=dataset_path)

        # Split dataset into train and eval
        dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        train_dataset = dataset["train"] 
        eval_dataset = dataset["test"]
        
        t0 = time.time()
        train_dataset = [ealge_format_multiview_data(sample) for sample in train_dataset]
        t1 = time.time()
        print("time taken (train_dataset to list) : ", t1 - t0)
        train_dataset = Dataset.from_list(train_dataset)
        t2 = time.time()
        print("train_dataset length: ", len(train_dataset))
        print("time taken (train_dataset to Dataset) : ", t2 - t1)

        eval_dataset = Dataset.from_list([ealge_format_multiview_data(sample) for sample in eval_dataset])
        t3 = time.time()
        print("eval_dataset length: ", len(eval_dataset))
        print("time taken (eval_dataset to Dataset) : ", t3 - t2)
        
        train_dataset.save_to_disk("./Multiview_processed_datasets/train_dataset")
        eval_dataset.save_to_disk("./Multiview_processed_datasets/eval_dataset")

    model_id = "nvidia/Eagle2-2B"

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )

    processor = AutoProcessor.from_pretrained(
        "nvidia/Eagle2-2B", 
        trust_remote_code=True, 
        use_fast=True
    )
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

    # Initialize wandb with dongguk university team
    wandb.init(
        entity="schaeck-dongguk-university",  # Use dongguk university team
        project="eagle2-2b-finetuning"
    )

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="eagle2-2b-trl-sft-Multitask",  # Directory to save the model
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=64,  # Steps to accumulate gradients
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        label_names=["labels"],
        max_length=None,
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        lr_scheduler_type="cosine",
        learning_rate=2e-4,  # Learning rate for training
        # Logging and evaluation
        logging_steps=5,  # Steps interval for logging
        eval_steps=50,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=100,  # Steps interval for saving
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        # max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        remove_unused_columns=False,  # Whether to remove unused columns
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        use_legacy_prediction_loop=True,
        report_to="wandb",  # Use Weights & Biases for logging
    )

    processor.tokenizer.pad_token = "<|endoftext|>"
    processor.tokenizer.pad_token_id = 151643

    eagle2_data_collator = Eagle2DataCollator(processor.tokenizer)

    trainer = Eagle2TRLSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=eagle2_data_collator
    )

    trainer.train()
    
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()