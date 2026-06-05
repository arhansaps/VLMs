import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2Tokenizer

# These are the exact mean/std ImageNet was normalized with during ResNet-50 training.
# We must use the same values because we're using a pretrained ResNet — it learned
# to see the world through this normalization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),       # ResNet-50 expects 224x224
    transforms.ToTensor(),               # PIL image [0,255] → float tensor [0,1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_captions(captions_file):
    """
    Parses captions.txt into a list of (filename, caption) tuples.
    Flickr8k format:  <filename>,<caption>
    """
    pairs = []
    with open(captions_file, "r") as f:
        next(f)  # skip 1st row lmao
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, caption = line.split(",", 1)
            pairs.append((filename.strip(), caption.strip()))
    return pairs


class Flickr8kDataset(Dataset):
    def __init__(self, captions_file, images_dir, max_caption_len=40):
        self.images_dir = images_dir
        self.pairs = load_captions(captions_file)

        # GPT-2 tokenizer — we use it to turn caption strings into token id tensors.
        # The model will predict these token ids one by one during training.

        #downloaded from hugging face
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # GPT-2 has no padding token by default. Set it to eos_token so the
        # tokenizer can pad shorter captions to a fixed length.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]

        # --- Image ---
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert("RGB")  # ensure 3 channels
        image_tensor = IMAGE_TRANSFORMS(image)          # shape: [3, 224, 224]

        # --- Caption ---
        # Tokenize, pad/truncate to max_caption_len, return pytorch tensors
        encoded = self.tokenizer(
            caption,
            max_length=self.max_caption_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # encoded["input_ids"]      shape: [1, max_caption_len]  — the token ids
        # encoded["attention_mask"] shape: [1, max_caption_len]  — 1 = real token, 0 = padding

        # squeeze removes the batch dim the tokenizer adds, so shapes become [max_caption_len]
        input_ids      = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return image_tensor, input_ids, attention_mask
