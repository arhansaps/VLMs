# Vision-Language Model from Scratch(not really)

A minimal VLM built to understand how vision and language representations can be fused. Specifically, how visual features extracted from images can be projected into the embedding space of a language model to generate natural language captions.

Most modern VLMs (LLaVA, Flamingo, PaliGemma) are too large to study meaningfully from the outside. This project builds the same core idea at a scale you can actually train, inspect, and reason about.

## Architecture

```
Input Image (224x224)
       |
ResNet-50 (frozen, pretrained on ImageNet)
       |
[B, 2048] visual feature vector
       |
Linear(2048 -> 768) + LayerNorm   <- projection layer (only part trained from scratch)
       |
[B, 768] visual token in GPT-2 embedding space
       |
GPT-2 LM Head (fine-tuned)
       |
Generated caption
```

- A frozen ResNet-50 acts as the vision encoder, extracting a 2048-dimensional feature vector from any input image
- A learned linear projection layer maps that 2048-dim visual feature into the 768-dim embedding space that GPT-2 expects
- GPT-2 takes the projected visual token as a prefix and generates a caption token by token

The projection layer is the central idea. It is the bridge between two entirely different representation spaces and learning it is what the training process is actually doing.

## Dataset

Flickr8k: 8,000 images each paired with 5 human-written captions. Small enough to train on a single GPU in a few hours, diverse enough to be a real test.

## Training

- Objective: cross-entropy loss over next-token prediction (standard language modelling)
- Teacher forcing: ground-truth tokens fed as input during training, not the model's own predictions
- What gets trained: only the projection layer and GPT-2 weights, the ResNet encoder stays frozen
- Optimizer: Adam, lr=1e-4

## Setup

```bash
pip install torch torchvision transformers pillow
```

Download Flickr8k from Kaggle and place images in `data/images/` and captions in `data/captions.txt`.

```bash
python train.py
python inference.py --image data/images/sample.jpg
```

## Sample Outputs

| Image | Generated Caption |
|-------|------------------|
| Dog running on beach | "a dog runs along the shoreline" |
| Two children on a swing | "two children play on a swing in a park" |

*(Updated after training)*

## What this is not

This is not a state-of-the-art model. BLEU scores will be modest and captions will sometimes be wrong. The goal is to understand the mechanism, specifically what it takes to get a language model to condition its output on a visual input, not to beat benchmarks.

## References

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Google, 2014) -- the paper this is most directly based on
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (OpenAI, 2021)
- [GPT-2](https://huggingface.co/gpt2) via Hugging Face Transformers
