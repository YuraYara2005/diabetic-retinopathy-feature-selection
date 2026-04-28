import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Local imports
from dataloader import create_dataloaders
from feature_extractor import ResNetFeatureExtractor

def extract_and_save(loader, model, split_name, output_dir):
    all_features = []
    all_labels = []

    print(f"\n--- Starting extraction for {split_name} split ---")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Processing {split_name}"):
            # images are moved to GPU inside the model's forward()
            features = model(images)
            
            # features: [Batch, 2048] on GPU -> move to CPU for numpy
            all_features.append(features.cpu().numpy())
            # labels are already on CPU from the dataloader
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Use Path for robust Windows path joining
    feat_path = output_dir / f"{split_name}_features.npy"
    label_path = output_dir / f"{split_name}_labels.npy"
    
    np.save(feat_path, X)
    np.save(label_path, y)

    print(f"✅ Saved {split_name}: Features {X.shape}, Labels {y.shape}")

def main():
    # 1. Configuration - Use absolute paths to avoid FileNotFoundError
    # This points to the project root regardless of where you run the script
    ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Increased Batch Size: Your hardware can likely handle 64 or 128 for ResNet50
    BATCH_SIZE = 64 
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Initialize
    extractor = ResNetFeatureExtractor()
    # Pass string version of path for compatibility with os.path in dataloader
    loaders = create_dataloaders(str(DATA_DIR), batch_size=BATCH_SIZE)

    # 3. Run Pipeline
    # Only process splits that actually have labels (usually train and val)
    for split in ['train', 'val']:
        if loaders[split] is not None:
            extract_and_save(loaders[split], extractor, split, PROCESSED_DIR)

    print("\n💎 Deep Feature Extraction Pipeline Complete!")
    print(f"Files are located in: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()