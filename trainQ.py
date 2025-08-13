import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os

# Load processor globally
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)

class CachedIAMDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples = []

        print("Preprocessing dataset into memory...")

        for i in range(len(df)):
            image_bytes = df['image'][i]['bytes']
            text = df['text'][i]

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()

            labels = processor.tokenizer(
                text,
                padding="max_length",
                max_length=max_target_length
            ).input_ids

            labels = [
                label if label != processor.tokenizer.pad_token_id else -100
                for label in labels
            ]

            self.samples.append({
                "pixel_values": pixel_values,
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Quantum-inspired optimizer approximation (Classical)
class QuantumInspiredOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta=0.9, gamma=0.01):
        defaults = dict(lr=lr, beta=beta, gamma=gamma)
        super(QuantumInspiredOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta = group['beta']
                gamma = group['gamma']

                state['step'] += 1
                # Quantum-inspired update (simplified approximation)
                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
                p.data.add_(exp_avg, alpha=-group['lr'] * (1 + gamma * state['step'])) # Added step

        return loss

def main():
    # Load data
    df = pd.read_parquet('DATA/0000.parquet')
    print(df.head())

    # Create dataset and dataloader
    dataset = CachedIAMDataset(df, processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Model config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Optimizer
    # optimizer = AdamW(model.parameters(), lr=1e-5) # Replace with our optimizer
    optimizer = QuantumInspiredOptimizer(model.parameters(), lr=1e-5, beta=0.9, gamma=0.01)

    # Training loop
    for epoch in range(1):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            # Update tqdm progress bar with current batch loss
            progress_bar.set_postfix(batch_loss=loss.item())

        print(f"Epoch {epoch} Avg Loss: {train_loss / len(dataloader):.4f}")

    # Save the model
    output_dir = "trocr_handwritten_quantum_inspired"  # You can change this
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)  # Also save the processor
    print(f"Model saved to {output_dir}")


if __name__ == '__main__':
    main()
