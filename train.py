import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Flickr8kDataset
from model import VLM


#hyperparameters
CAPTIONS_FILE = "data/captions.txt"
IMAGES_DIR    = "data/images"
BATCH_SIZE    = 32
EPOCHS        = 10
LR            = 1e-4
SAVE_PATH     = "vlm_checkpoint.pt"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Data ---
    dataset = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Model ---
    model = VLM().to(device)

    # Only pass params that actually need gradients (ResNet is frozen)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    # ignore_index=-100 tells the loss to skip padding positions
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, input_ids, attention_mask) in enumerate(dataloader):
            images         = images.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # logits: [B, seq_len+1, vocab_size]
            logits = model(images, input_ids, attention_mask)

            # Shift logits: position 0 (visual token) predicts input_ids[:, 0],
            # position 1 predicts input_ids[:, 1], ..., drop the last logit as it has nothing to predict.
            shift_logits = logits[:, :-1, :]   # [B, seq_len, vocab_size]

            # Labels are the raw caption token ids.
            # Mask padding positions so they don't contribute to loss.
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # [B, seq_len]

            # CrossEntropyLoss expects [N, vocab] and [N], so flatten batch+seq dims
            loss = criterion(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}  step {batch_idx+1}/{len(dataloader)}  loss {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done — avg loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Checkpoint saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
