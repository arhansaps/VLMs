import argparse
import torch
from PIL import Image
from transformers import GPT2Tokenizer

from dataset import IMAGE_TRANSFORMS
from model import VLM

CHECKPOINT = "vlm_checkpoint.pt"


def generate_caption(image_path, max_new_tokens=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model = VLM().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # --- Preprocess image ---
    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORMS(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        # Get visual token — same first two steps as training forward pass
        visual_features = model.encoder(image_tensor)          # [1, 2048]
        visual_token = model.projection(visual_features)       # [1, 768]
        visual_token = visual_token.unsqueeze(1)               # [1, 1, 768]

        # Attention mask for the single visual token
        attention_mask = torch.ones(1, 1, device=device)

        # Use GPT-2's built-in generate(), seeded with the visual token as the prompt
        output_ids = model.gpt2.generate(
            inputs_embeds=visual_token,
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
    args = parser.parse_args()

    caption = generate_caption(args.image)
    print(f"Caption: {caption}")
