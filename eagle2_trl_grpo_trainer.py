from trl import GRPOTrainer
from typing import Union, Any
import torch
import warnings
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import pad
from PIL import Image
import requests
import os
import io
import numpy as np
import time
from trl.extras.profiling import profiling_context
from trl.trainer.utils import selective_log_softmax
from accelerate.utils import gather_object, broadcast_object_list
from contextlib import nullcontext
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.models.utils import unwrap_model_for_generation
from trl.data_utils import is_conversational, apply_chat_template
from accelerate.utils import gather
import functools

# vLLM imports - only import when needed
try:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    SamplingParams = None
    GuidedDecodingParams = None


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def profiling_decorator(func: callable) -> callable:
    """
    Decorator to profile a function and log execution time using [`extras.profiling.profiling_context`].

    Args:
        func (`callable`):
            Function to be profiled.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_decorator

    class MyTrainer(Trainer):
        @profiling_decorator
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            # Code to profile: simulate a computationally expensive operation
            result = A @ B
    ```
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

class Eagle2TRLGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalize_content_items(self, content_items):
        """
        Apply content normalization logic from SFT to GRPO
        - Remove keys with None values
        - Handle image loading
        """
        normalized_items = []
        for item in content_items:
            if not isinstance(item, dict):
                continue
                
            # Remove keys with None values (same as SFT logic)
            normalized_item = {k: v for k, v in item.items() if v is not None}
            
            # Image processing logic
            if normalized_item.get("type") == "image" and "image_url" in normalized_item:
                image_url_dict = normalized_item["image_url"]
                if isinstance(image_url_dict, dict) and "url" in image_url_dict:
                    try:
                        image_path = image_url_dict["url"]
                        if image_path.startswith(("http://", "https://")):
                            # Load image from web URL
                            response = requests.get(image_path)
                            response.raise_for_status()
                            image = Image.open(io.BytesIO(response.content))
                        else:
                            # Load image from local file
                            if os.path.exists(image_path):
                                image = Image.open(image_path)
                            else:
                                print(f"Warning: Image file not found: {image_path}")
                                continue
                        
                        # Add image directly to item
                        normalized_item["image"] = image
                    except Exception as e:
                        print(f"Warning: Failed to load image {image_path}: {e}")
                        continue
            
            normalized_items.append(normalized_item)
        
        return normalized_items

    def _preprocess_conversation_data(self, inputs):
        """
        Preprocess conversational data - apply SFT's _prepare_dataset logic to GRPO
        """
        if not isinstance(inputs, list):
            print(f"[WARNING] inputs is not a list, got {type(inputs)}")
            return inputs
            
        processed_inputs = []
        
        for input_item in inputs:
            if not isinstance(input_item, dict):
                print(f"[WARNING] input_item is not a dict, got {type(input_item)}")
                processed_inputs.append(input_item)
                continue
                
            if "prompt" not in input_item:
                processed_inputs.append(input_item)
                continue
                
            if not isinstance(input_item["prompt"], list):
                processed_inputs.append(input_item)
                continue
                
            # Normalize conversation messages
            processed_messages = []
            for message in input_item["prompt"]:
                if not isinstance(message, dict) or "content" not in message:
                    continue
                    
                # When content is a list (multimodal)
                if isinstance(message["content"], list):
                    normalized_content = self._normalize_content_items(message["content"])
                    if normalized_content:  # Exclude empty content
                        processed_message = message.copy()
                        processed_message["content"] = normalized_content
                        processed_messages.append(processed_message)
                else:
                    # Text-only case
                    processed_messages.append(message)
            
            # Update with preprocessed messages
            processed_item = input_item.copy()
            processed_item["prompt"] = processed_messages
            processed_inputs.append(processed_item)
        
        return processed_inputs

    def _process_prompts_with_images(self, prompts):
        """
        Process images from prompts in the same way as SFT
        """
        prompts_text = []
        images_by_prompt = []

        for prompt in prompts:
            # Apply chat template for each prompt
            try:
                prompt_text = self.processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                prompts_text.append(prompt_text)
            except Exception as e:
                print(f"Error applying chat template: {e}")
                text_parts = []
                for message in prompt:
                    if isinstance(message.get("content"), list):
                        for content_item in message["content"]:
                            if content_item.get("type") == "text" and "text" in content_item:
                                text_parts.append(content_item["text"])
                prompts_text.append(" ".join(text_parts))

            # Collect images per prompt
            try:
                image_inputs_per_prompt, _ = self.processing_class.process_vision_info(prompt)
            except Exception as e:
                print(f"Error processing vision info: {e}")
                image_inputs_per_prompt = None

            if image_inputs_per_prompt is None:
                images_by_prompt.append(None)
            else:
                images_by_prompt.append(image_inputs_per_prompt)

        return {
            "prompts_text": prompts_text,
            "images_by_prompt": images_by_prompt,
        }
    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Debug: check original inputs
        
        # BatchFeature check - skip processing in this case (also problematic in original GRPO)
        from transformers.feature_extraction_utils import BatchFeature
        if isinstance(inputs, BatchFeature):
            # Try to extract actual list from BatchFeature
            if hasattr(inputs, 'data') and isinstance(inputs.data, dict):
                # Check BatchFeature internal data structure
                # Generally this case should not occur, so raise error
                raise ValueError("BatchFeature input not expected in _generate_and_score_completions")
        
        if len(inputs) > 0:
            try:
                _ = inputs[0]
            except (KeyError, TypeError) as e:
                return super()._generate_and_score_completions(inputs)

        # Apply preprocessing only for lists with dictionary elements
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
            try:
                processed_inputs = self._preprocess_conversation_data(inputs)
            except Exception as e:
                print(f"[WARNING] Preprocessing failed: {e}, using original inputs")
                processed_inputs = inputs
        else:
            processed_inputs = inputs
        
        # Debug: check data after preprocessing

        prompts = [x["prompt"] for x in processed_inputs]
        
        # Apply same image processing logic as SFT
        processed_prompts_and_images = self._process_prompts_with_images(prompts)
        prompts_text = processed_prompts_and_images["prompts_text"]
        images_by_prompt = processed_prompts_and_images.get("images_by_prompt", None)
        
        has_images = (
            isinstance(images_by_prompt, list)
            and any((imgs is not None and len(imgs) > 0) for imgs in images_by_prompt)
        ) or (images_by_prompt is not None and not isinstance(images_by_prompt, list))
        
        # Per-sample processing, then batch-collate (SFT-style)
        per_sample_outputs = []
        for prompt, images in zip(prompts_text, images_by_prompt):
            sample_out = self.processing_class(
                text=[prompt],
                images=images,
                videos=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            per_sample_outputs.append(sample_out)

        # Pad input_ids/attention_mask across samples
        text_features = []
        for o in per_sample_outputs:
            # squeeze batch dim 1 -> [L]
            text_features.append({
                "input_ids": o["input_ids"].squeeze(0),
                "attention_mask": o["attention_mask"].squeeze(0),
            })
        padded = self.processing_class.tokenizer.pad(text_features, padding=True, return_tensors="pt")

        prompt_inputs = {
            "input_ids": padded["input_ids"].to(device),
            "attention_mask": padded["attention_mask"].to(device),
        }

        # Concatenate vision tensors if present
        pixel_values_list = [o["pixel_values"] for o in per_sample_outputs if "pixel_values" in o]
        if len(pixel_values_list) > 0:
            prompt_inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0).to(device)
            
            # Generate image_flags based on pixel_values shape (similar to SFT trainer)
            total_pixel_values = prompt_inputs["pixel_values"]
            if total_pixel_values.dim() == 4:  # [B*N, C, H, W] format
                # For GRPO, we need to figure out how many views per sample
                # Assuming each sample has the same number of views
                num_samples = len(per_sample_outputs)
                total_views = total_pixel_values.size(0)
                views_per_sample = total_views // num_samples if num_samples > 0 else 1
                
                # Create image_flags: True for real images, False for padding
                image_flags_list = []
                for o in per_sample_outputs:
                    if "pixel_values" in o:
                        pv = o["pixel_values"]
                        if pv.dim() == 4:  # [N, C, H, W]
                            num_views = pv.size(0)
                            # True for actual images, False for any padding
                            flags = torch.ones(num_views, dtype=torch.bool)
                        elif pv.dim() == 3:  # [C, H, W]
                            flags = torch.ones(1, dtype=torch.bool)
                        else:
                            flags = torch.zeros(views_per_sample, dtype=torch.bool)
                    else:
                        flags = torch.zeros(views_per_sample, dtype=torch.bool)
                    image_flags_list.append(flags)
                
                prompt_inputs["image_flags"] = torch.cat(image_flags_list, dim=0).to(device)
            
        image_sizes_list = [o["image_sizes"] for o in per_sample_outputs if "image_sizes" in o]
        if len(image_sizes_list) > 0:
            prompt_inputs["image_sizes"] = torch.cat(image_sizes_list, dim=0).to(device)

        # Logging (optional)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    # Check if prompt_ids is a simple tensor
                    if isinstance(prompt_ids, dict):
                        raise ValueError("prompt_ids should be a tensor, not a dict")
                    
                    # Check if pixel_values is passed to model
                    generate_kwargs = {"attention_mask": prompt_mask}
                    if "pixel_values" in prompt_inputs:
                        generate_kwargs["pixel_values"] = prompt_inputs["pixel_values"]
                    # Note: image_flags is NOT needed for generation, only for logit computation
                    if "image_sizes" in prompt_inputs:
                        generate_kwargs["image_sizes"] = prompt_inputs["image_sizes"]
                    
                    # Measure generation
                    gen_prompt_len = prompt_ids.size(1)
                    gen_start = time.time()
                    prompt_completion_ids = unwrapped_model.generate(
                        input_ids=prompt_ids,  # explicitly pass as input_ids
                        generation_config=self.generation_config,
                        **generate_kwargs
                    )
                    gen_dur = time.time() - gen_start


            # Compute prompt length and robustly extract completion ids
            prompt_length = prompt_ids.size(1)
            generated_length = prompt_completion_ids.size(1)

            # Check whether generated sequences include the prompt as a prefix
            includes_prompt_prefix = False
            if prompt_completion_ids.dim() == 2 and generated_length >= prompt_length:
                try:
                    includes_prompt_prefix = (prompt_completion_ids[:, :prompt_length] == prompt_ids).all().item()
                except Exception as e:
                    includes_prompt_prefix = False

            if includes_prompt_prefix:
                completion_ids = prompt_completion_ids[:, prompt_length:]
                # Keep prompt_ids as-is for clarity; slice below is redundant but harmless
                prompt_ids = prompt_completion_ids[:, :prompt_length]
            else:
                # Many multimodal/chat models return completions-only. Treat full output as completion.
                completion_ids = prompt_completion_ids
                # Reconstruct prompt_completion_ids for downstream logit computation
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token (safe for zero-length completions)
        if completion_ids.size(1) == 0:
            completion_mask = torch.zeros((completion_ids.size(0), 0), dtype=torch.int, device=device)
        else:
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            has_eos = is_eos.any(dim=1)
            if has_eos.any():
                eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        # Prepare multimodal inputs for logit computation
        model_kwargs = {}
        if "pixel_values" in prompt_inputs:
            # Expand pixel_values to match prompt_completion_ids batch size if needed
            pixel_values = prompt_inputs["pixel_values"]
            if pixel_values.size(0) != prompt_completion_ids.size(0):
                # Repeat pixel_values for each generation if needed
                repeat_factor = prompt_completion_ids.size(0) // pixel_values.size(0)
                if repeat_factor > 1:
                    pixel_values = pixel_values.repeat(repeat_factor, 1, 1, 1)
            model_kwargs["pixel_values"] = pixel_values
        
        if "image_flags" in prompt_inputs:
            # Expand image_flags to match prompt_completion_ids batch size if needed
            image_flags = prompt_inputs["image_flags"]
            if image_flags.size(0) != prompt_completion_ids.size(0):
                repeat_factor = prompt_completion_ids.size(0) // image_flags.size(0)
                if repeat_factor > 1:
                    image_flags = image_flags.repeat(repeat_factor, 1)
            model_kwargs["image_flags"] = image_flags
            

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size, **model_kwargs
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Ensure consistent sample count across prompts/completions/ids
        n_prompts = len(prompts)
        n_compl_text = len(completions_text)
        n_ids = len(completion_ids_list)
        n = min(n_prompts, n_compl_text, n_ids)
        prompts = prompts[:n]
        completions_text = completions_text[:n]
        completion_ids_list = completion_ids_list[:n]

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                # Do not mutate prompt; just read the last assistant message content if present
                bootstrap = prompt[-1]["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Initialize rewards tensor with consistent n
        rewards_per_func = torch.zeros(n, len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs][:n] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    # Defensive: coerce reward outputs to length n with padding/trimming and rich diagnostics
                    try:
                        coerced = np.asarray(output_reward_func, dtype=np.float32)
                    except Exception:
                        if np.isscalar(output_reward_func):
                            coerced = np.full((n,), float(output_reward_func), dtype=np.float32)
                        else:
                            coerced = np.full((0,), np.nan, dtype=np.float32)

                    if coerced.ndim == 0:
                        coerced = np.full((n,), float(coerced), dtype=np.float32)

                    if coerced.size == 0:
                        coerced = np.full((n,), np.nan, dtype=np.float32)
                    elif coerced.size != n:
                        if coerced.size > n:
                            coerced = coerced[:n]
                        else:
                            coerced = np.pad(coerced, (0, n - coerced.size), constant_values=np.nan)

                    rewards_per_func[:, i] = torch.tensor(coerced, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "pixel_values": model_kwargs.get("pixel_values"),
            "image_flags": model_kwargs.get("image_flags"),
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }
        
    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, pixel_values, image_flags, batch_size=None, **model_kwargs) -> torch.Tensor:
        # Force batch size to 1 for testing (remove later)
        batch_size = 1
        # batch_size = batch_size or input_ids.size(0)  # original code
        all_logps = []
        
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # Slice multimodal data to match batch
            # pixel_values and image_flags are in [B*N, ...] format, so special handling needed
            if pixel_values is not None:
                total_samples = input_ids.size(0)  # total number of samples
                total_views = pixel_values.size(0)  # total number of views
                num_views_per_sample = total_views // total_samples
                
                pixel_start_idx = i * num_views_per_sample  
                pixel_end_idx = (i + batch_size) * num_views_per_sample
                pixel_values_batch = pixel_values[pixel_start_idx:pixel_end_idx]
            else:
                pixel_values_batch = None
                
            if image_flags is not None:
                total_samples = input_ids.size(0)  # total number of samples
                total_flags = image_flags.size(0)  # total number of flags
                num_views_per_sample = total_flags // total_samples
                
                flags_start_idx = i * num_views_per_sample
                flags_end_idx = (i + batch_size) * num_views_per_sample  
                image_flags_batch = image_flags[flags_start_idx:flags_end_idx]
            else:
                image_flags_batch = None

            # Prepare model kwargs for this batch
            batch_model_kwargs = {}
            for key, value in model_kwargs.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    # Slice tensor inputs to match current batch
                    batch_model_kwargs[key] = value[i : i + batch_size]
                else:
                    # Keep scalar or non-tensor values as-is
                    batch_model_kwargs[key] = value
            
            # print(f"!![DEBUG] input_ids_batch.shape: {input_ids_batch.shape}")
            # print(f"!![DEBUG] input_ids_batch: {input_ids_batch}")
            # print(f"!![DEBUG] attention_mask_batch.shape: {attention_mask_batch.shape}")
            # # print(f"!![DEBUG] attention_mask_batch: {attention_mask_batch}")
            # print(f"!![DEBUG] logits_to_keep: {logits_to_keep}")
            # print(f"!![DEBUG] pixel_values_batch.shape: {pixel_values_batch.shape}")
            # # print(f"!![DEBUG] pixel_values_batch: {pixel_values_batch}")
            # print(f"!![DEBUG] image_flags_batch.shape: {image_flags_batch.shape}")
            # print(f"!![DEBUG] image_flags_batch: {image_flags_batch}")
            # print(f"!![DEBUG] batch_model_kwargs: {batch_model_kwargs}")
            
            # Try to use logits_to_keep if supported, otherwise fallback to manual slicing
            try:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                logits = model(
                            input_ids=input_ids_batch, 
                            attention_mask=attention_mask_batch, 
                            pixel_values=pixel_values_batch,
                            image_flags=image_flags_batch,
                            logits_to_keep=logits_to_keep + 1,
                            **batch_model_kwargs
                ).logits
                logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            except TypeError:
                # For models that don't support logits_to_keep argument
                # See https://github.com/huggingface/trl/issues/2770
                logits = model(
                    input_ids=input_ids_batch, 
                    attention_mask=attention_mask_batch,
                    pixel_values=pixel_values_batch,
                    image_flags=image_flags_batch,
                    **batch_model_kwargs
                ).logits
                logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
                # Manually slice to keep only the last logits_to_keep tokens
                logits = logits[:, -logits_to_keep:]
            
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        
        return torch.cat(all_logps, dim=0)
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask, pixel_values, image_flags = inputs["prompt_ids"], inputs["prompt_mask"], inputs["pixel_values"], inputs["image_flags"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, pixel_values, image_flags)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep, pixel_values, image_flags
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep, pixel_values, image_flags
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss
    
    def shuffle_multimodal_tensor_dict(self, tensor_dict: dict[str, torch.Tensor], num_views_per_sample: int = None) -> dict[str, torch.Tensor]:
        """
        Shuffles a dictionary of tensors while preserving multimodal data structure.
        For Eagle2, pixel_values and image_flags have shape [B*N, ...] where B is batch size and N is views per sample.
        """
        # Find a text tensor to determine the actual batch size
        text_tensors = ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages", "old_per_token_logps"]
        actual_batch_size = None
        
        for key in text_tensors:
            if key in tensor_dict and tensor_dict[key] is not None:
                actual_batch_size = tensor_dict[key].shape[0]
                break
        
        if actual_batch_size is None:
            # Fallback: assume all tensors have the same batch size
            first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
            actual_batch_size = first_tensor.shape[0]
        
        # If we have multimodal data, calculate views per sample
        if num_views_per_sample is None and "pixel_values" in tensor_dict and tensor_dict["pixel_values"] is not None:
            total_views = tensor_dict["pixel_values"].shape[0]
            num_views_per_sample = total_views // actual_batch_size
        
        
        # Generate permutation for the actual batch
        permutation = torch.randperm(actual_batch_size)
        
        shuffled_dict = {}
        for key, tensor in tensor_dict.items():
            if tensor is None:
                shuffled_dict[key] = None
                continue
                
            if key in ["pixel_values", "image_flags"] and num_views_per_sample and num_views_per_sample > 1:
                # Reshape to [B, N, ...], shuffle along B, then reshape back to [B*N, ...]
                original_shape = tensor.shape
                if key == "pixel_values":  # [B*N, C, H, W] -> [B, N, C, H, W]
                    tensor_reshaped = tensor.view(actual_batch_size, num_views_per_sample, *original_shape[1:])
                elif key == "image_flags":  # [B*N] -> [B, N]
                    tensor_reshaped = tensor.view(actual_batch_size, num_views_per_sample)
                
                # Shuffle along batch dimension
                tensor_shuffled = tensor_reshaped[permutation]
                
                # Reshape back to original format
                shuffled_dict[key] = tensor_shuffled.view(*original_shape)
            else:
                # Standard shuffle for text tensors
                shuffled_dict[key] = tensor[permutation]
        
        return shuffled_dict
    
    def split_multimodal_tensor_dict(self, tensor_dict: dict[str, torch.Tensor], num_splits: int) -> list[dict[str, torch.Tensor]]:
        """
        Split a dictionary of tensors into multiple chunks while preserving multimodal data structure.
        """
        # Find a text tensor to determine the actual batch size
        text_tensors = ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages", "old_per_token_logps"]
        actual_batch_size = None
        
        for key in text_tensors:
            if key in tensor_dict and tensor_dict[key] is not None:
                actual_batch_size = tensor_dict[key].shape[0]
                break
        
        if actual_batch_size is None:
            # Fallback: assume all tensors have the same batch size
            first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
            actual_batch_size = first_tensor.shape[0]
        
        # Calculate views per sample for multimodal data
        num_views_per_sample = None
        if "pixel_values" in tensor_dict and tensor_dict["pixel_values"] is not None:
            total_views = tensor_dict["pixel_values"].shape[0]
            num_views_per_sample = total_views // actual_batch_size
        
        
        # Calculate chunk size for text tensors
        chunk_size = actual_batch_size // num_splits
        
        split_dicts = []
        for i in range(num_splits):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_splits - 1 else actual_batch_size
            
            split_dict = {}
            for key, tensor in tensor_dict.items():
                if tensor is None:
                    split_dict[key] = None
                    continue
                
                if key in ["pixel_values", "image_flags"] and num_views_per_sample and num_views_per_sample > 1:
                    # For multimodal tensors: [B*N, ...] format
                    start_view_idx = start_idx * num_views_per_sample
                    end_view_idx = end_idx * num_views_per_sample
                    split_dict[key] = tensor[start_view_idx:end_view_idx]
                else:
                    # For text tensors: normal split
                    split_dict[key] = tensor[start_idx:end_idx]
            
            split_dicts.append(split_dict)
        
        return split_dicts
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """Override to use our multimodal-aware shuffle and split functions."""
        # Import the original functions
        from trl.trainer.grpo_trainer import shuffle_tensor_dict, split_tensor_dict
        
        # Monkey patch both functions temporarily
        original_shuffle = shuffle_tensor_dict
        original_split = split_tensor_dict
        
        def multimodal_shuffle_wrapper(tensor_dict):
            """Wrapper that uses our multimodal shuffle function."""
            return self.shuffle_multimodal_tensor_dict(tensor_dict)
        
        def multimodal_split_wrapper(tensor_dict, num_splits):
            """Wrapper that uses our multimodal split function."""
            return self.split_multimodal_tensor_dict(tensor_dict, num_splits)
        
        # Replace both functions
        import trl.trainer.grpo_trainer
        trl.trainer.grpo_trainer.shuffle_tensor_dict = multimodal_shuffle_wrapper
        trl.trainer.grpo_trainer.split_tensor_dict = multimodal_split_wrapper
        
        try:
            # Call the original training loop with all arguments
            result = super()._inner_training_loop(
                batch_size=batch_size, 
                args=args, 
                resume_from_checkpoint=resume_from_checkpoint, 
                trial=trial, 
                ignore_keys_for_eval=ignore_keys_for_eval
            )
        finally:
            # Restore the original functions
            trl.trainer.grpo_trainer.shuffle_tensor_dict = original_shuffle
            trl.trainer.grpo_trainer.split_tensor_dict = original_split
        
        return result
