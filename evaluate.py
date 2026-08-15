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
