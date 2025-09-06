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

        # Handle pixel_values and image_flags if present
        # Check if any feature has pixel_values (indicates image data is present)
        has_images = any('pixel_values' in feature for feature in features)
        
        if has_images:
            # Extract image-related data (with fallback for missing keys)
            pixel_values_list = []
            image_flags_list = []
            image_sizes_list = []
            
            for feature in features:
                if 'pixel_values' in feature:
                    pixel_values_list.append(feature['pixel_values'])
                else:
                    # Create dummy pixel_values for consistency
                    pixel_values_list.append(torch.zeros(3, 448, 448))  # Adjust size as needed
                    
                if 'image_flags' in feature:
                    image_flags_list.append(feature['image_flags'])
                else:
                    # Create dummy image_flags (False = no real image)
                    image_flags_list.append(torch.tensor(False, dtype=torch.bool))

                if 'image_sizes' in feature:
                    image_sizes_list.append(feature['image_sizes'])
                else:
                    image_sizes_list.append(torch.tensor([448, 448]))

            # Stack and reshape pixel_values
            pixel_values_stacked = torch.stack(pixel_values_list)
            b, c, h, w = pixel_values_stacked.shape
            pixel_values_reshaped = pixel_values_stacked.view(b, c, h, w)

            # Stack and reshape image_flags
            image_flags_stacked = torch.stack(image_flags_list)  # shape: [b] or [b, t]
            if image_flags_stacked.dim() == 1:
                image_flags_reshaped = image_flags_stacked  # shape: [b]
            else:
                image_flags_reshaped = image_flags_stacked.view(b)  # shape: [b]

            batch.update({
                'pixel_values': pixel_values_reshaped,
                'image_flags': image_flags_reshaped,
            })

            # Attach image_sizes if gathered
            if len(image_sizes_list) > 0:
                batch['image_sizes'] = torch.stack(image_sizes_list)
            
            # # Handle image_sizes if present
            # if 'image_sizes' in features[0]:
            #     image_sizes_list = [feature.get('image_sizes', torch.tensor([[448, 448]])) for feature in features]
            #     batch['image_sizes'] = torch.stack(image_sizes_list)
        
        return batch