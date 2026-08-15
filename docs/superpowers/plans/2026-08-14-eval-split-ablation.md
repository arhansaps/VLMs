# Evaluation, Data Split & Visual-Token Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give this VLM a held-out train/val/test split, automated BLEU/CIDEr evaluation, best-checkpoint selection by validation loss, an LR warmup+decay schedule, and a single-token vs. multi-token visual encoding ablation — so captioning quality can be measured and compared instead of eyeballed.

**Architecture:** Extend the existing flat-script structure rather than restructuring it. Add `splits.py` (deterministic image-level train/val/test assignment) and `evaluate.py` (loads a checkpoint, generates captions over a split, scores BLEU-1/4 + CIDEr). Wire a validation loop, best-checkpoint saving, and an LR scheduler into `train.py`. Add an optional spatial (multi-token) mode to `VisionEncoder`/`VLM` so `train.py` can be run once with 1 visual token and once with 49, and the two runs compared with `evaluate.py`.

**Tech Stack:** existing stack (PyTorch 2.5, torchvision ResNet-50, HF Transformers GPT-2) + `pytest` (new tests) + `nltk` (BLEU) + `pycocoevalcap` (CIDEr — pure-Python `Cider` scorer only; METEOR/SPICE from that package need Java and are intentionally not used here).

**Spec:** captured inline below (no separate spec doc — requirements came from conversation, not a written spec file).

## Global Constraints

- This Flickr8k copy has 8,091 images (not the original paper's 8,000) — split proportionally to approximate the standard Karpathy split (6000/1000/1000 ≈ 90/5/5): **90% train / 5% val / 5% test**, split **by image filename**, never by caption row, so no image's 5 captions leak across splits.
- Do not change ResNet-50's architecture or its `requires_grad=False` freezing.
- Keep the existing `CrossEntropyLoss(ignore_index=-100)` padding convention.
- Keep the existing `torch.device("cuda" if torch.cuda.is_available() else "cpu")` auto-detect pattern in every new/modified entrypoint.
- Match existing code style: flat hyperparameter constants at the top of scripts, no config framework/YAML, argparse only where `inference.py` already sets the precedent (and now `evaluate.py`).
- `vlm_checkpoint.pt` and any new checkpoint filenames stay gitignored (already covered by the existing `.gitignore` pattern — verify new filenames match or extend it).

---

### Task 1: Deterministic image-level train/val/test split

**Files:**
- Create: `splits.py`
- Modify: `dataset.py` (`Flickr8kDataset.__init__`)
- Test: `test_splits.py`

**Interfaces:**
- Produces: `get_split(images_dir, captions_file, seed=42, val_frac=0.05, test_frac=0.05) -> dict` returning `{"train": [filenames], "val": [filenames], "test": [filenames]}`.
- Produces: `Flickr8kDataset(captions_file, images_dir, max_caption_len=40, split_filenames=None)` — when `split_filenames` is given (an iterable of filenames), only caption rows for those filenames are kept.

- [ ] **Step 1: Write the failing tests for `get_split`**

```python
# test_splits.py
from splits import get_split
from dataset import load_captions

CAPTIONS_FILE = "data/captions.txt"
IMAGES_DIR = "data/images"


def test_split_is_deterministic():
    split_a = get_split(IMAGES_DIR, CAPTIONS_FILE, seed=42)
    split_b = get_split(IMAGES_DIR, CAPTIONS_FILE, seed=42)
    assert split_a == split_b


def test_split_has_no_overlap():
    split = get_split(IMAGES_DIR, CAPTIONS_FILE, seed=42)
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_split_covers_all_images():
    pairs = load_captions(CAPTIONS_FILE)
    all_filenames = {filename for filename, _ in pairs}
    split = get_split(IMAGES_DIR, CAPTIONS_FILE, seed=42)
    covered = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert covered == all_filenames


def test_split_proportions_are_approximately_90_5_5():
    split = get_split(IMAGES_DIR, CAPTIONS_FILE, seed=42)
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert 0.88 <= len(split["train"]) / total <= 0.92
    assert 0.03 <= len(split["val"]) / total <= 0.07
    assert 0.03 <= len(split["test"]) / total <= 0.07
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pip install pytest && pytest test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'splits'`

- [ ] **Step 3: Implement `splits.py`**

```python
# splits.py
import random

from dataset import load_captions


def get_split(images_dir, captions_file, seed=42, val_frac=0.05, test_frac=0.05):
    """
    Deterministic image-level train/val/test split.
    Splits by image filename (not by caption row) so a given image's
    captions never appear in more than one split.
    """
    pairs = load_captions(captions_file)
    filenames = sorted({filename for filename, _ in pairs})

    rng = random.Random(seed)
    rng.shuffle(filenames)

    n = len(filenames)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    return {
        "train": filenames[:n_train],
        "val": filenames[n_train:n_train + n_val],
        "test": filenames[n_train + n_val:],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_splits.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing test for dataset filtering**

```python
# append to test_splits.py
from dataset import Flickr8kDataset


def test_dataset_filters_by_split_filenames():
    split = get_split(IMAGES_DIR, CAPTIONS_FILE, seed=42)
    val_dataset = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR, split_filenames=split["val"])
    filenames_in_dataset = {fn for fn, _ in val_dataset.pairs}
    assert filenames_in_dataset <= set(split["val"])
    assert len(val_dataset) > 0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest test_splits.py::test_dataset_filters_by_split_filenames -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'split_filenames'`

- [ ] **Step 7: Modify `dataset.py`**

```python
# dataset.py — replace Flickr8kDataset.__init__
class Flickr8kDataset(Dataset):
    def __init__(self, captions_file, images_dir, max_caption_len=40, split_filenames=None):
        self.images_dir = images_dir
        pairs = load_captions(captions_file)

        if split_filenames is not None:
            allowed = set(split_filenames)
            pairs = [(filename, caption) for filename, caption in pairs if filename in allowed]

        self.pairs = pairs

        # GPT-2 tokenizer — we use it to turn caption strings into token id tensors.
        # The model will predict these token ids one by one during training.

        #downloaded from hugging face
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # GPT-2 has no padding token by default. Set it to eos_token so the
        # tokenizer can pad shorter captions to a fixed length.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_caption_len = max_caption_len
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest test_splits.py -v`
Expected: PASS (5 tests)

- [ ] **Step 9: Commit**

```bash
git add splits.py dataset.py test_splits.py
git commit -m "add deterministic train/val/test split"
```

---

### Task 2: Evaluation script (BLEU-1/4 + CIDEr)

**Files:**
- Create: `evaluate.py`
- Test: `test_evaluate.py`

**Interfaces:**
- Consumes: `get_split` from Task 1, `load_captions` and `IMAGE_TRANSFORMS` from `dataset.py`, `VLM` from `model.py`.
- Produces: `compute_bleu(references, hypotheses) -> dict` and `compute_cider(references, hypotheses) -> (float, dict)`, both keyed by image filename (`references[filename]` = list of ref strings, `hypotheses[filename]` = one generated string). Later tasks (Task 6 runbook) call the `evaluate.py` CLI directly, not these functions.

- [ ] **Step 1: Install dependencies**

Run: `pip install nltk pycocoevalcap`

- [ ] **Step 2: Write the failing tests**

```python
# test_evaluate.py
from evaluate import compute_bleu, compute_cider


def test_bleu_is_high_for_identical_caption():
    references = {"img1": ["a dog runs on the beach"]}
    hypotheses = {"img1": "a dog runs on the beach"}
    scores = compute_bleu(references, hypotheses)
    assert scores["bleu1"] > 95
    assert scores["bleu4"] > 90


def test_bleu_is_low_for_unrelated_caption():
    references = {"img1": ["a dog runs on the beach"]}
    hypotheses = {"img1": "completely unrelated text about cars"}
    scores = compute_bleu(references, hypotheses)
    assert scores["bleu1"] < 30


def test_cider_is_higher_for_closer_match():
    references = {
        "img1": ["a dog runs on the beach", "a brown dog runs along the shore"],
    }
    close_hyp = {"img1": "a dog runs on the beach"}
    far_hyp = {"img1": "a cat sleeps on a couch"}

    close_score, _ = compute_cider(references, close_hyp)
    far_score, _ = compute_cider(references, far_hyp)
    assert close_score > far_score
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 4: Implement the scoring functions in `evaluate.py`**

```python
# evaluate.py
import argparse
from collections import defaultdict

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from PIL import Image
from transformers import GPT2Tokenizer

from dataset import load_captions, IMAGE_TRANSFORMS
from splits import get_split
from model import VLM


def compute_bleu(references, hypotheses):
    """
    references: dict[filename] -> list[str] (multiple reference captions)
    hypotheses: dict[filename] -> str (one generated caption)
    Returns corpus-level BLEU-1..4 on a 0-100 scale.
    """
    smoothing = SmoothingFunction().method1
    filenames = list(hypotheses.keys())
    list_of_references = [[ref.split() for ref in references[fn]] for fn in filenames]
    list_of_hypotheses = [hypotheses[fn].split() for fn in filenames]

    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n for _ in range(n))
        scores[f"bleu{n}"] = 100 * corpus_bleu(
            list_of_references, list_of_hypotheses, weights=weights, smoothing_function=smoothing
        )
    return scores


def compute_cider(references, hypotheses):
    """
    references / hypotheses keyed by filename, in pycocoevalcap's expected format
    (gts: filename -> list[str], res: filename -> list[str] of length 1).
    Returns (mean_score, per_image_scores).
    """
    cider_scorer = Cider()
    gts = {fn: references[fn] for fn in hypotheses}
    res = {fn: [hypotheses[fn]] for fn in hypotheses}
    mean_score, per_image_scores = cider_scorer.compute_score(gts, res)
    return mean_score, per_image_scores
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest test_evaluate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Add the CLI entrypoint to `evaluate.py`**

```python
# evaluate.py — append below the functions from Step 4

def load_references(captions_file, filenames):
    pairs = load_captions(captions_file)
    allowed = set(filenames)
    refs = defaultdict(list)
    for filename, caption in pairs:
        if filename in allowed:
            refs[filename].append(caption)
    return dict(refs)


def generate_captions(model, tokenizer, images_dir, filenames, device, max_new_tokens=40):
    """Greedy decoding (not sampling) so evaluation runs are reproducible."""
    hypotheses = {}
    model.eval()
    with torch.no_grad():
        for filename in filenames:
            image = Image.open(f"{images_dir}/{filename}").convert("RGB")
            image_tensor = IMAGE_TRANSFORMS(image).unsqueeze(0).to(device)

            visual_features = model.encoder(image_tensor)
            visual_tokens = model.projection(visual_features)
            if visual_tokens.dim() == 2:
                visual_tokens = visual_tokens.unsqueeze(1)
            num_visual_tokens = visual_tokens.size(1)
            attention_mask = torch.ones(1, num_visual_tokens, device=device)

            output_ids = model.gpt2.generate(
                inputs_embeds=visual_tokens,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            hypotheses[filename] = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return hypotheses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="vlm_checkpoint.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--spatial", action="store_true", help="model was trained with 49 spatial visual tokens")
    parser.add_argument("--captions-file", default="data/captions.txt")
    parser.add_argument("--images-dir", default="data/images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLM(spatial_visual_tokens=args.spatial).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    split = get_split(args.images_dir, args.captions_file)
    filenames = split[args.split]

    references = load_references(args.captions_file, filenames)
    hypotheses = generate_captions(model, tokenizer, args.images_dir, filenames, device)

    bleu_scores = compute_bleu(references, hypotheses)
    cider_score, _ = compute_cider(references, hypotheses)

    print(f"Evaluated {len(filenames)} images from the '{args.split}' split (checkpoint: {args.checkpoint})")
    for name, score in bleu_scores.items():
        print(f"{name}: {score:.2f}")
    print(f"CIDEr: {cider_score:.4f}")
```

`--spatial` here depends on the `spatial_visual_tokens` argument added to `VLM` in Task 5 — this CLI won't run until Task 5 is done, but the scoring functions (Steps 1-5) are independently testable now.

- [ ] **Step 7: Commit**

```bash
git add evaluate.py test_evaluate.py
git commit -m "add BLEU/CIDEr evaluation script"
```

---

### Task 3: Validation loss + best-checkpoint selection

**Files:**
- Modify: `train.py`
- Test: `test_train.py`

**Interfaces:**
- Consumes: `get_split` (Task 1), `Flickr8kDataset(..., split_filenames=...)` (Task 1).
- Produces: `is_new_best(val_loss, best_so_far) -> bool` and `evaluate_val_loss(model, dataloader, criterion, device) -> float`, both used inside `train()`.

- [ ] **Step 1: Write the failing test for `is_new_best`**

```python
# test_train.py
from train import is_new_best


def test_is_new_best_when_no_prior_best():
    assert is_new_best(1.5, None) is True


def test_is_new_best_when_improved():
    assert is_new_best(1.0, 1.5) is True


def test_is_new_best_when_not_improved():
    assert is_new_best(2.0, 1.5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_new_best' from 'train'`

- [ ] **Step 3: Add `is_new_best` and `evaluate_val_loss` to `train.py`**

```python
# train.py — add near the top, after imports
def is_new_best(val_loss, best_so_far):
    return best_so_far is None or val_loss < best_so_far


def evaluate_val_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, input_ids, attention_mask in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_train.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire the split, val loop, and best-checkpoint saving into `train()`**

```python
# train.py — replace the hyperparameters block and train() body
from splits import get_split

CAPTIONS_FILE = "data/captions.txt"
IMAGES_DIR    = "data/images"
BATCH_SIZE    = 32
EPOCHS        = 10
LR            = 1e-4
SAVE_PATH      = "vlm_checkpoint.pt"       # latest, every epoch
BEST_SAVE_PATH = "vlm_checkpoint_best.pt"  # only overwritten on val-loss improvement


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
    model = VLM().to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
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

            logits = model(images, input_ids, attention_mask)
            shift_logits = logits[:, :-1, :]

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            loss = criterion(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
```

This is an integration change (real training loop), not something to unit test directly — Task 1's split tests and this task's `is_new_best`/`evaluate_val_loss`-shape tests are the coverage; correctness of the full loop is confirmed by actually running `python train.py` for a couple of epochs after Task 5 is also in place (see Task 6 runbook).

- [ ] **Step 6: Extend `.gitignore` for the new checkpoint file**

```
# .gitignore
/data
vlm_checkpoint.pt
vlm_checkpoint_best.pt
```

- [ ] **Step 7: Commit**

```bash
git add train.py test_train.py .gitignore
git commit -m "add validation loss and best-checkpoint selection"
```

---

### Task 4: LR warmup + cosine decay, differential LR for projection vs. GPT-2

**Files:**
- Modify: `train.py`
- Test: `test_train.py` (append)

**Interfaces:**
- Produces: `build_param_groups(model, projection_lr, gpt2_lr) -> list[dict]` and `build_lr_lambda(warmup_steps, total_steps) -> Callable[[int], float]`, both consumed inside `train()`.

**Why differential LR:** the projection layer is randomly initialized and needs to move fast; GPT-2 is already a good language model and full fine-tuning it at the same `1e-4` used for the from-scratch layer risks catastrophic forgetting of its language ability. Giving GPT-2 a smaller LR (`1e-5`) than the projection layer (`1e-4`) is the standard fix.

- [ ] **Step 1: Write the failing tests**

```python
# test_train.py — append
from model import VLM
from train import build_param_groups, build_lr_lambda


def test_build_param_groups_separates_projection_and_gpt2():
    model = VLM()
    groups = build_param_groups(model, projection_lr=1e-4, gpt2_lr=1e-5)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-5

    projection_param_ids = {id(p) for p in model.projection.parameters()}
    group0_ids = {id(p) for p in groups[0]["params"]}
    assert projection_param_ids == group0_ids


def test_lr_lambda_warms_up_linearly():
    lr_lambda = build_lr_lambda(warmup_steps=10, total_steps=100)
    assert lr_lambda(0) == 0.0
    assert lr_lambda(5) == 0.5
    assert lr_lambda(10) == 1.0


def test_lr_lambda_decays_after_warmup():
    lr_lambda = build_lr_lambda(warmup_steps=10, total_steps=100)
    assert lr_lambda(100) < 0.01
    assert lr_lambda(55) < lr_lambda(10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_param_groups' from 'train'`

- [ ] **Step 3: Implement both functions in `train.py`**

```python
# train.py — add near is_new_best/evaluate_val_loss
import math


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_train.py -v`
Expected: PASS (6 tests total in this file)

- [ ] **Step 5: Wire param groups + scheduler into `train()`**

```python
# train.py — inside train(), replace the optimizer/criterion block
PROJECTION_LR = 1e-4
GPT2_LR       = 1e-5
WARMUP_STEPS  = 200

    # --- Model ---
    model = VLM().to(device)

    param_groups = build_param_groups(model, PROJECTION_LR, GPT2_LR)
    optimizer = torch.optim.Adam(param_groups)

    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(WARMUP_STEPS, total_steps))

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
```

And inside the batch loop, right after `optimizer.step()`:

```python
            optimizer.step()
            scheduler.step()
```

(`PROJECTION_LR`/`GPT2_LR`/`WARMUP_STEPS` replace the old flat `LR = 1e-4` constant — remove `LR` since it's no longer used.)

- [ ] **Step 6: Commit**

```bash
git add train.py test_train.py
git commit -m "add LR warmup+cosine schedule with differential projection/GPT-2 rates"
```

---

### Task 5: Multi-token (spatial) visual encoding ablation

**Files:**
- Modify: `encoder.py`, `model.py`, `train.py`, `inference.py`
- Test: `test_encoder.py`, `test_model.py`

**Interfaces:**
- Produces: `VisionEncoder(spatial=False)` — unchanged default behavior, `[B, 2048]`; `VisionEncoder(spatial=True)` — `[B, 49, 2048]`.
- Produces: `VLM(spatial_visual_tokens=False)` — unchanged default, 1 visual token; `VLM(spatial_visual_tokens=True)` — 49 visual tokens. `VLM.forward` signature is unchanged either way.
- `ProjectionLayer` needs no code change — `nn.Linear` already broadcasts over the extra `[B, 49, 2048] -> [B, 49, 768]` leading dimension.

- [ ] **Step 1: Write the failing tests for `VisionEncoder`**

```python
# test_encoder.py
import torch
from encoder import VisionEncoder


def test_vision_encoder_default_output_shape():
    encoder = VisionEncoder()
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 2048)


def test_vision_encoder_spatial_output_shape():
    encoder = VisionEncoder(spatial=True)
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 49, 2048)
```

- [ ] **Step 2: Run tests to verify the spatial one fails**

Run: `pytest test_encoder.py -v`
Expected: `test_vision_encoder_default_output_shape` PASSES already (no regression); `test_vision_encoder_spatial_output_shape` FAILS with `TypeError: __init__() got an unexpected keyword argument 'spatial'`

- [ ] **Step 3: Modify `encoder.py`**

```python
# encoder.py
import torch
import torch.nn as nn
from torchvision import models


class VisionEncoder(nn.Module):
    def __init__(self, spatial=False):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if spatial:
            # Drop avgpool + fc: keep the [B, 2048, 7, 7] spatial feature map
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        else:
            # Drop only fc: avgpool gives [B, 2048, 1, 1]
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.spatial = spatial

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.encoder(x)

        if self.spatial:
            b, c, h, w = features.shape
            return features.view(b, c, h * w).permute(0, 2, 1)  # [B, 49, 2048]

        features = features.squeeze(-1).squeeze(-1)  # [B, 2048]
        return features
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_encoder.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing tests for `VLM`**

```python
# test_model.py
import torch
from model import VLM


def test_vlm_forward_single_token_shape():
    model = VLM(spatial_visual_tokens=False)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (2, 10))
    attention_mask = torch.ones(2, 10)
    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (2, 11, 50257)


def test_vlm_forward_spatial_tokens_shape():
    model = VLM(spatial_visual_tokens=True)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (2, 10))
    attention_mask = torch.ones(2, 10)
    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (2, 59, 50257)  # 49 visual tokens + 10 caption tokens
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest test_model.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'spatial_visual_tokens'`

- [ ] **Step 7: Modify `model.py`**

```python
# model.py
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from encoder import VisionEncoder
from projection import ProjectionLayer


class VLM(nn.Module):
    def __init__(self, spatial_visual_tokens=False):
        super().__init__()

        self.encoder = VisionEncoder(spatial=spatial_visual_tokens)
        self.projection = ProjectionLayer()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, images, input_ids, attention_mask):
        # images:         [B, 3, 224, 224]
        # input_ids:      [B, seq_len]
        # attention_mask: [B, seq_len]

        visual_features = self.encoder(images)          # [B, 2048] or [B, 49, 2048]
        visual_tokens = self.projection(visual_features)  # [B, 768] or [B, 49, 768]
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(1)   # [B, 1, 768]
        num_visual_tokens = visual_tokens.size(1)

        token_embeddings = self.gpt2.transformer.wte(input_ids)  # [B, seq_len, 768]

        inputs_embeds = torch.cat([visual_tokens, token_embeddings], dim=1)  # [B, seq_len+N, 768]

        visual_mask = torch.ones(images.size(0), num_visual_tokens, device=attention_mask.device)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)  # [B, seq_len+N]

        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        return outputs.logits  # [B, seq_len+N, vocab_size]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest test_model.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Update `train.py` to select the mode and keep the two ablation runs' checkpoints separate**

```python
# train.py — add near the top-of-file constants
SPATIAL_VISUAL_TOKENS = False  # flip to True for the 49-token ablation run

_SUFFIX = "49tok" if SPATIAL_VISUAL_TOKENS else "1tok"
SAVE_PATH      = f"vlm_checkpoint_{_SUFFIX}.pt"
BEST_SAVE_PATH = f"vlm_checkpoint_{_SUFFIX}_best.pt"
```

```python
# train.py — inside train(), where the model is constructed
    model = VLM(spatial_visual_tokens=SPATIAL_VISUAL_TOKENS).to(device)
```

- [ ] **Step 10: Update `.gitignore` to cover both ablation checkpoints**

```
# .gitignore
/data
vlm_checkpoint*.pt
```

- [ ] **Step 11: Update `inference.py` to generalize past the single-token assumption**

```python
# inference.py
import argparse
import torch
from PIL import Image
from transformers import GPT2Tokenizer

from dataset import IMAGE_TRANSFORMS
from model import VLM


def generate_caption(image_path, checkpoint, spatial_visual_tokens=False, max_new_tokens=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLM(spatial_visual_tokens=spatial_visual_tokens).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORMS(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        visual_features = model.encoder(image_tensor)
        visual_tokens = model.projection(visual_features)
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(1)
        num_visual_tokens = visual_tokens.size(1)

        attention_mask = torch.ones(1, num_visual_tokens, device=device)

        output_ids = model.gpt2.generate(
            inputs_embeds=visual_tokens,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--checkpoint", default="vlm_checkpoint_1tok_best.pt")
    parser.add_argument("--spatial", action="store_true", help="checkpoint was trained with 49 spatial visual tokens")
    args = parser.parse_args()

    caption = generate_caption(args.image, args.checkpoint, spatial_visual_tokens=args.spatial)
    print(f"Caption: {caption}")
```

- [ ] **Step 12: Commit**

```bash
git add encoder.py model.py train.py inference.py test_encoder.py test_model.py .gitignore
git commit -m "add spatial multi-token visual encoding mode for ablation"
```

---

### Task 6: Run the ablation and record results (runbook, no new code)

This task is where the previous five tasks actually produce a finding. No tests — it's executing the pipeline twice and writing down what it measured.

- [ ] **Step 1: Train the 1-token baseline**

Confirm `SPATIAL_VISUAL_TOKENS = False` in `train.py`, then:

Run: `python train.py`
Produces: `vlm_checkpoint_1tok.pt`, `vlm_checkpoint_1tok_best.pt`

- [ ] **Step 2: Train the 49-token ablation**

Set `SPATIAL_VISUAL_TOKENS = True` in `train.py`, then:

Run: `python train.py`
Produces: `vlm_checkpoint_49tok.pt`, `vlm_checkpoint_49tok_best.pt`

- [ ] **Step 3: Evaluate both on the held-out test split**

Run:
```bash
python evaluate.py --checkpoint vlm_checkpoint_1tok_best.pt --split test
python evaluate.py --checkpoint vlm_checkpoint_49tok_best.pt --spatial --split test
```

- [ ] **Step 4: Record the real numbers in `README.md`**

Add a "Results" section to `README.md` with a table of the two runs' BLEU-1/4 and CIDEr scores from Step 3's actual output — use the real printed numbers, not placeholders. Note whichever direction the result goes (spatial tokens help, hurt, or are a wash) as the actual finding, since either outcome is informative.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "record 1-token vs 49-token visual encoding ablation results"
```

---

## Self-Review Notes

- **Spec coverage:** eval (BLEU+CIDEr) → Task 2; train/val/test split → Task 1; ablation → Task 5+6; checkpoint selection by val loss → Task 3; LR scheduling → Task 4; inference `model.eval()`/`no_grad()` → already present in the existing `inference.py` (confirmed during audit, no task needed).
- **Not in scope, deliberately:** caption text normalization (lowercasing/punctuation), gradient clipping, mixed precision, experiment tracking (wandb/tensorboard) — flagged as options during the audit but not requested; add as a follow-up plan if wanted.
- **Type/name consistency check:** `get_split` (Task 1) is reused verbatim in Task 3 and Task 2/6; `is_new_best`/`evaluate_val_loss` (Task 3) are only used inside `train.py`; `build_param_groups`/`build_lr_lambda` (Task 4) likewise; `VLM(spatial_visual_tokens=...)` and `VisionEncoder(spatial=...)` naming is consistent across Tasks 5 and 6 and the `evaluate.py`/`inference.py` `--spatial` flags added in Tasks 2 and 5.
