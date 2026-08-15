import argparse
import torch
from PIL import Image
from transformers import GPT2Tokenizer

from dataset import IMAGE_TRANSFORMS
from model import VLM


def generate_caption(image_path, checkpoint, spatial_visual_tokens=False, max_new_tokens=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model = VLM(spatial_visual_tokens=spatial_visual_tokens).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # --- Preprocess image ---
    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORMS(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        # Get visual token(s) — same first two steps as training forward pass
        visual_features = model.encoder(image_tensor)          # [1, 2048] or [1, 49, 2048]
        visual_tokens = model.projection(visual_features)      # [1, 768] or [1, 49, 768]
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(1)          # [1, 1, 768]
        num_visual_tokens = visual_tokens.size(1)

        # Attention mask for the visual token(s)
        attention_mask = torch.ones(1, num_visual_tokens, device=device)

        # Use GPT-2's built-in generate(), seeded with the visual token(s) as the prompt
        output_ids = model.gpt2.generate(
            inputs_embeds=visual_tokens,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,           # sampling avoids greedy repetition loops
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
