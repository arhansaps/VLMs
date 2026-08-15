from dataset import Flickr8kDataset

CAPTIONS_FILE = "data/captions.txt"
IMAGES_DIR = "data/images"


def test_attention_mask_extends_one_past_caption_for_eos_supervision():
    dataset = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR, max_caption_len=40)
    tokenizer = dataset.tokenizer
    filename, caption = dataset.pairs[0]

    raw_encoded = tokenizer(
        caption, max_length=40, padding="max_length", truncation=True, return_tensors="pt"
    )
    raw_seq_len = int(raw_encoded["attention_mask"].squeeze(0).sum().item())

    _, input_ids, attention_mask = dataset[0]

    assert int(attention_mask.sum().item()) == raw_seq_len + 1
    assert input_ids[raw_seq_len].item() == tokenizer.eos_token_id
    assert attention_mask[raw_seq_len].item() == 1
