from torch.utils.data import DataLoader
from dataset import Flickr8kDataset

dataset = Flickr8kDataset(
    captions_file="data/captions.txt",
    images_dir="data/images",
)

print(f"Total samples: {len(dataset)}")  # should be ~40,000 as each (caption,image) pair is treated as one sample so 5 x 8000 = 40,000

# Grab the first item and print shapes
image, input_ids, attention_mask = dataset[0]
print(f"Image tensor:    {image.shape}")         # [3, 224, 224]
print(f"Input IDs:       {input_ids.shape}")     # [40]
print(f"Attention mask:  {attention_mask.shape}") # [40]

# Test the dataloader with a batch
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"\nBatch image shape:  {batch[0].shape}")  # [8, 3, 224, 224]
print(f"Batch token shape:  {batch[1].shape}")    # [8, 40]
print("Dataset OK")
