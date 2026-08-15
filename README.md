# Vision-Language Model from Scratch (not really)

A minimal, from-first-principles Vision-Language Model (VLM) that generates natural-language captions for images. It exists to answer one question directly: *how do you get a language model to condition its output on a visual input?* Most production VLMs (LLaVA, Flamingo, PaliGemma) are too large to inspect end-to-end; this project reproduces the same core mechanism — projecting image features into a language model's embedding space — at a scale that fits on a single GPU and can be read top to bottom.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10 |
| Deep learning framework | [PyTorch](https://pytorch.org/) 2.5.1 (+cu124, CUDA-enabled build) |
| Vision backbone | `torchvision.models.resnet50` — ImageNet-pretrained, frozen |
| Language model | `GPT2LMHeadModel` via [Hugging Face Transformers](https://huggingface.co/gpt2) 4.48.0 |
| Tokenizer | `GPT2Tokenizer` (Hugging Face), EOS token reused as PAD token |
| Image I/O / preprocessing | Pillow (PIL) 12.2.0 + `torchvision.transforms` |
| Data loading | `torch.utils.data.Dataset` / `DataLoader` |
| Dataset | [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,091 images × 5 captions each (40,456 caption rows) |
| Hardware target | CUDA GPU if available, falls back to CPU (`torch.cuda.is_available()`) |

No web framework, API layer, or UI — this is a pure training/inference research script, run from the command line.

## Architecture

```
Input Image (224x224x3)
       |
ResNet-50 (frozen, ImageNet-pretrained; final FC layer stripped off)
       |
[B, 2048] pooled visual feature vector (output of avgpool)
       |
Linear(2048 -> 768) + LayerNorm      <-- Projection Layer (the only module trained from scratch)
       |
[B, 768] "visual token" living in GPT-2's embedding space
       |
Prepended to GPT-2 token embeddings as position 0, with an all-ones entry
added to the attention mask so GPT-2 always attends to it
       |
GPT-2 (small, 124M) LM head, fully fine-tuned
       |
Generated caption, token by token
```

**How the three pieces fit together (`model.py`):**
1. `VisionEncoder` (`encoder.py`) runs the image through ResNet-50 up to (and including) the average-pool layer, producing a `[B, 2048]` feature vector. The classification head is discarded — this project never needs ImageNet class predictions, only the pooled feature representation. All ResNet parameters have `requires_grad=False`; it is used purely as a fixed feature extractor.
2. `ProjectionLayer` (`projection.py`) is a single `Linear(2048 → 768)` followed by `LayerNorm`. It is the entire "bridge" between the vision and language spaces, and the only newly-initialized module in the whole system — everything else is a pretrained checkpoint.
3. The projected `[B, 768]` vector is unsqueezed to `[B, 1, 768]` and treated as **one token** ("the image is one token"). It's concatenated in front of the caption's GPT-2 token embeddings (`gpt2.transformer.wte(input_ids)`), and the attention mask is extended by one column of `1`s so GPT-2 always attends to it. GPT-2 then runs normally over `inputs_embeds` (rather than `input_ids`, since the sequence starts with a non-token embedding), producing `[B, seq_len+1, vocab_size]` logits.

**What actually gets trained:** the projection layer (from scratch) and every GPT-2 weight (full fine-tune, not frozen/LoRA) — `train.py` only excludes ResNet params from the optimizer via `filter(lambda p: p.requires_grad, ...)`. ResNet-50 stays frozen throughout.

## File-by-file breakdown

| File | Purpose |
|---|---|
| [`dataset.py`](dataset.py) | `Flickr8kDataset`: parses `data/captions.txt` (`filename,caption` CSV, header skipped) into `(filename, caption)` pairs. `__getitem__` loads the image, converts to RGB, resizes to 224×224, and normalizes with ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) — required because ResNet-50 was trained on data normalized this way. Captions are tokenized with the GPT-2 tokenizer, padded/truncated to `max_caption_len=40`, with EOS reused as the pad token (GPT-2 ships with no PAD token by default). Returns `(image_tensor [3,224,224], input_ids [40], attention_mask [40])`. Each of the 5 captions per image is treated as an independent sample (~40k total samples). |
| [`encoder.py`](encoder.py) | `VisionEncoder`: wraps a pretrained, frozen ResNet-50 with its final FC layer removed, exposing the 2048-d pooled feature vector per image. |
| [`projection.py`](projection.py) | `ProjectionLayer`: `Linear(2048→768) + LayerNorm`, mapping ResNet features into GPT-2's embedding dimensionality. The only module trained from a random init. |
| [`model.py`](model.py) | `VLM`: composes `VisionEncoder` + `ProjectionLayer` + `GPT2LMHeadModel` into the full forward pass described above, returning next-token logits over the visual-token-prefixed sequence. |
| [`train.py`](train.py) | Training loop. Hyperparameters: `BATCH_SIZE=32`, `EPOCHS=10`, `LR=1e-4`, Adam optimizer, `CrossEntropyLoss(ignore_index=-100)`. Labels are the raw caption token ids with padding positions masked to `-100` so they don't contribute to loss; logits are shifted so position *i* predicts `input_ids[i]` (position 0's prediction, from the visual token, predicts the first caption token). Logs loss every 50 steps and prints an epoch average. Saves/overwrites a full model checkpoint (`vlm_checkpoint.pt`) to disk after every epoch. Runs on CUDA if available, else CPU. |
| [`inference.py`](inference.py) | CLI script (`--image <path>`) that loads `vlm_checkpoint.pt`, runs an image through the encoder + projection to get a single visual token, then calls `gpt2.generate()` seeded with that token as the entire prompt (`inputs_embeds`, no text prompt). Uses sampling decoding: `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.3`, `do_sample=True` — chosen over greedy decoding specifically to avoid repetition loops. Prints the decoded caption. |
| [`test_dataset.py`](test_dataset.py) | Standalone sanity-check script (not a pytest suite) — instantiates `Flickr8kDataset`, prints total sample count and the shapes of a single item and of one `DataLoader` batch, to confirm the dataset/dataloader plumbing works before spending time on a full training run. |
| [`initial.md`](initial.md) | The original project pitch/README draft — architecture diagram, dataset description, training objective, setup instructions, references to the "Show and Tell" and CLIP papers. Superseded in detail by this README but kept as the original design note. |
| [`data/captions.txt`](data/captions.txt) | Flickr8k captions, CSV format `image,caption`, header row + 40,456 data rows (8,091 unique images × 5 captions each). |
| [`data/images/`](data/images/) | The 8,091 Flickr8k JPEGs referenced by `captions.txt`. |
| [`vlm_checkpoint.pt`](vlm_checkpoint.pt) | Saved model weights (`model.state_dict()`) from the most recent training run (~598 MB — full GPT-2 + projection layer weights; ResNet weights are not saved since they're just the stock pretrained ones). Gitignored. |
| [`.gitignore`](.gitignore) | Ignores `/data` (the dataset) and `vlm_checkpoint.pt` (the trained weights) — both are regenerated/downloaded locally rather than committed. |

## Data flow summary

```
data/captions.txt + data/images/
        |  (Flickr8kDataset)
        v
(image_tensor, input_ids, attention_mask)   -- batched by DataLoader
        |  (VLM.forward, train.py)
        v
next-token logits  -->  CrossEntropyLoss vs. shifted caption tokens (padding masked out)
        |
        v
backprop through GPT-2 + ProjectionLayer only (ResNet frozen)
        |
        v
vlm_checkpoint.pt saved after every epoch
        |  (inference.py)
        v
image  -->  encoder -> projection -> visual token -> gpt2.generate()  -->  caption string
```

## Setup

```bash
pip install torch torchvision transformers pillow
```

Download [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place images in `data/images/` and captions in `data/captions.txt` (both are gitignored, so this step is required after cloning).

```bash
python train.py                      # trains for 10 epochs on a 90/5/5 train/val/test split
                                      # saves vlm_checkpoint_1tok.pt (latest) and vlm_checkpoint_1tok_best.pt (best val loss)
python evaluate.py --checkpoint vlm_checkpoint_1tok_best.pt --split test   # BLEU-1..4 + CIDEr on held-out test images
python inference.py --image data/images/<some_image>.jpg --checkpoint vlm_checkpoint_1tok_best.pt
```

Training and inference both auto-detect CUDA and fall back to CPU. See [BUGFIXES.md](BUGFIXES.md) for issues hit along the way and how they were diagnosed/fixed.

## Results

Test-split evaluation (404 held-out images, 1 visual token, greedy decoding with repetition guards) after 10 epochs:

| Metric | Score |
|---|---|
| BLEU-1 | 60.97 |
| BLEU-2 | 39.05 |
| BLEU-3 | 25.86 |
| BLEU-4 | 17.32 |
| CIDEr  | 0.4623 |

For reference, Show-and-Tell/NeuralTalk-era captioning baselines at a similar scale typically land around BLEU-4 ~20-27 and CIDEr ~0.6-0.9 on Flickr8k/30k — these numbers are in that neighborhood, not state-of-the-art but a functioning captioner. (An earlier run scored CIDEr 0.0002 with a non-trivial BLEU-1 — that mismatch was diagnostic of two real bugs, not just a weak model; see [BUGFIXES.md](BUGFIXES.md).)

## Design notes / why things are the way they are

- **ResNet-50 is frozen entirely.** It's already trained on ~1.1M ImageNet images; the project's job is only to learn the *bridge* between its feature space and GPT-2's, not to re-learn vision.
- **The image becomes exactly one token** by default (`SPATIAL_VISUAL_TOKENS = False` in `train.py`). No patch grid, no cross-attention — the simplest possible fusion mechanism, comparable to "Show and Tell" (2014) rather than modern patch/cross-attention VLMs. A `spatial_visual_tokens=True` mode exists on `VisionEncoder`/`VLM` for an ablation using ResNet's 7×7 = 49 spatial features as 49 tokens instead.
- **GPT-2 is fully fine-tuned, not frozen**, but at a lower learning rate (1e-5) than the from-scratch projection layer (1e-4) — full LR parity between a random-init layer and an already-competent language model risks catastrophic forgetting in GPT-2.
- **Sampling decoding at inference, greedy + repetition guards at evaluation.** Greedy decoding on a small fine-tuned GPT-2 degenerates into repetition loops without help; `inference.py` uses `temperature=0.7` + `top_p=0.9` + `repetition_penalty=1.3` (sampling), `evaluate.py` uses greedy decoding with `repetition_penalty=1.3` + `no_repeat_ngram_size=3` so evaluation runs stay reproducible.
- **The model is explicitly supervised to predict EOS.** Padding beyond the caption is masked out of the loss via `ignore_index=-100`, but the *first* padding position (which already holds the EOS token, since `pad_token == eos_token`) is deliberately left unmasked — otherwise the model never learns when a caption should end (see [BUGFIXES.md](BUGFIXES.md)).

## What this is not

Not a state-of-the-art model — captions will sometimes be wrong, and this isn't chasing benchmark leaderboards. The goal is understanding the mechanism of visual conditioning in a language model.

## References

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Google, 2014) — the paper this is most directly based on
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (OpenAI, 2021)
- [GPT-2](https://huggingface.co/gpt2) via Hugging Face Transformers
