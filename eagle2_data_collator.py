import torch

class Eagle2DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        try:
            self.image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        except KeyError:
            self.image_token_id = 151667
            print(f"Warning: '<image>' token not found. Using hardcoded ID: {self.image_token_id}")

    def __call__(self, features):
        # Extract labels first (always present)
        labels_list = [feature['labels'] for feature in features]
        
        # Prepare only input_ids and attention_mask for tokenizer.pad
        text_features_for_padding = [
            {
                "input_ids": feature['input_ids'],
                "attention_mask": feature['attention_mask']
            } 
            for feature in features
        ]

        padded_text = self.tokenizer.pad(
            text_features_for_padding,
            padding='longest',
            return_tensors='pt',
        )

        # Manually pad labels to match input_ids length
        max_len = padded_text['input_ids'].shape[1]
        
        padded_labels = []
        for label in labels_list:
            pad_amount = max_len - len(label)
            padded_labels.append(
                torch.nn.functional.pad(label, (pad_amount, 0), value=-100)
            )

        input_ids = padded_text['input_ids']
        batch = {
            'input_ids': input_ids,
            'attention_mask': padded_text['attention_mask'],
            'labels': torch.stack(padded_labels),
        }

        # As a safeguard, mask out any image tokens in labels at batch time
        if isinstance(self.image_token_id, int):
            image_mask = (batch['input_ids'] == self.image_token_id)
            batch['labels'][image_mask] = -100

        # Handle pixel_values and image_flags for zero, single, or multi-view per sample
        # Determine the maximum number of views in the batch
        max_num_views = 0
        for feature in features:
            if 'pixel_values' in feature:
                pv = feature['pixel_values']
                if isinstance(pv, torch.Tensor):
                    if pv.dim() == 4:  # [N, C, H, W]
                        max_num_views = max(max_num_views, pv.shape[0])
                    elif pv.dim() == 3:  # [C, H, W]
                        max_num_views = max(max_num_views, 1)
        
        if max_num_views > 0:
            # Prepare padded pixel_values: [B, N, C, H, W] and image_flags: [B, N]
            padded_pixel_values = []
            padded_image_flags = []

            for feature in features:
                # Prepare per-sample pixel_values tensor of shape [n, C, H, W]
                if 'pixel_values' in feature and isinstance(feature['pixel_values'], torch.Tensor):
                    pv = feature['pixel_values']
                    if pv.dim() == 4 and pv.shape[0] >= 1:
                        per_sample_pv = pv
                    elif pv.dim() == 3:
                        per_sample_pv = pv.unsqueeze(0)
                    else:
                        per_sample_pv = torch.zeros(1, 3, 448, 448)
                else:
                    per_sample_pv = torch.zeros(1, 3, 448, 448)

                n, c, h, w = per_sample_pv.shape
                pad_n = max_num_views - n
                if pad_n > 0:
                    pad_tensor = torch.zeros(pad_n, c, h, w)
                    per_sample_pv = torch.cat([per_sample_pv, pad_tensor], dim=0)
                padded_pixel_values.append(per_sample_pv)

                # image_flags per sample: [n]
                if 'image_flags' in feature and isinstance(feature['image_flags'], torch.Tensor):
                    flags = feature['image_flags']
                    if flags.dim() == 0:
                        flags = flags.unsqueeze(0)
                else:
                    flags = torch.zeros(n, dtype=torch.bool)
                if flags.numel() < max_num_views:
                    flags = torch.cat([flags, torch.zeros(max_num_views - flags.numel(), dtype=torch.bool)], dim=0)
                elif flags.numel() > max_num_views:
                    flags = flags[:max_num_views]
                padded_image_flags.append(flags)

            # Reshape pixel_values from [B, N, C, H, W] to [B*N, C, H, W] for the model
            stacked_pixel_values = torch.stack(padded_pixel_values)  # [B, N, C, H, W]
            batch_size, num_views, c, h, w = stacked_pixel_values.shape
            reshaped_pixel_values = stacked_pixel_values.view(batch_size * num_views, c, h, w)  # [B*N, C, H, W]
            
            # Reshape image_flags from [B, N] to [B*N] to match pixel_values
            stacked_image_flags = torch.stack(padded_image_flags)  # [B, N]
            reshaped_image_flags = stacked_image_flags.view(batch_size * num_views)  # [B*N]
            
            batch.update({
                'pixel_values': reshaped_pixel_values,              # [B*N, C, H, W]
                'image_flags': reshaped_image_flags,                # [B*N]
            })
        
        return batch