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
