from dataset import Flickr8kDataset, load_captions
from splits import get_split

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


def test_dataset_filters_by_split_filenames():
    split = get_split(IMAGES_DIR, CAPTIONS_FILE, seed=42)
    val_dataset = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR, split_filenames=split["val"])
    filenames_in_dataset = {fn for fn, _ in val_dataset.pairs}
    assert filenames_in_dataset <= set(split["val"])
    assert len(val_dataset) > 0
