import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Flickr8kDataset
from model import VLM
from splits import get_split


#hyperparameters
CAPTIONS_FILE = "data/captions.txt"
IMAGES_DIR    = "data/images"
BATCH_SIZE    = 32
EPOCHS        = 10
PROJECTION_LR = 1e-4
GPT2_LR       = 1e-5
WARMUP_STEPS  = 200

SPATIAL_VISUAL_TOKENS = False  # flip to True for the 49-token ablation run

_SUFFIX = "49tok" if SPATIAL_VISUAL_TOKENS else "1tok"
SAVE_PATH      = f"vlm_checkpoint_{_SUFFIX}.pt"       # latest, every epoch
BEST_SAVE_PATH = f"vlm_checkpoint_{_SUFFIX}_best.pt"   # only overwritten on val-loss improvement


def is_new_best(val_loss, best_so_far):
    return best_so_far is None or val_loss < best_so_far


def evaluate_val_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, input_ids, attention_mask in dataloader:
            images         = images.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(images, input_ids, attention_mask)
            shift_logits = logits[:, :-1, :]

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            loss = criterion(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                labels.reshape(-1),
            )
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)


def build_param_groups(model, projection_lr, gpt2_lr):
    return [
        {"params": list(model.projection.parameters()), "lr": projection_lr},
        {"params": list(model.gpt2.parameters()), "lr": gpt2_lr},
    ]


def build_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Data ---
    split = get_split(IMAGES_DIR, CAPTIONS_FILE)
    train_dataset = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR, split_filenames=split["train"])
    val_dataset   = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR, split_filenames=split["val"])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model ---
    model = VLM(spatial_visual_tokens=SPATIAL_VISUAL_TOKENS).to(device)

    # Only pass params that actually need gradients (ResNet is frozen).
    # Projection layer (from scratch) gets a higher LR than GPT-2 (fine-tuned,
    # already knows language — a flat LR risks catastrophic forgetting there).
    param_groups = build_param_groups(model, PROJECTION_LR, GPT2_LR)
    optimizer = torch.optim.Adam(param_groups)

    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(WARMUP_STEPS, total_steps))

    # ignore_index=-100 tells the loss to skip padding positions
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- Loop ---
    best_val_loss = None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, input_ids, attention_mask) in enumerate(train_loader):
            images         = images.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # logits: [B, seq_len+N, vocab_size]  (N visual tokens)
            logits = model(images, input_ids, attention_mask)

            # Shift logits: visual-token position(s) predict input_ids[:, 0],
            # position 1 predicts input_ids[:, 1], ..., drop the last logit as it has nothing to predict.
            shift_logits = logits[:, :-1, :]

            # Labels are the raw caption token ids.
            # Mask padding positions so they don't contribute to loss.
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # CrossEntropyLoss expects [N, vocab] and [N], so flatten batch+seq dims
            loss = criterion(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}  step {batch_idx+1}/{len(train_loader)}  loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate_val_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} done — avg train loss: {avg_loss:.4f}  val loss: {val_loss:.4f}")

        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Checkpoint saved to {SAVE_PATH}")

        if is_new_best(val_loss, best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_SAVE_PATH)
            print(f"New best val loss {val_loss:.4f} — saved to {BEST_SAVE_PATH}")


if __name__ == "__main__":
    train()
