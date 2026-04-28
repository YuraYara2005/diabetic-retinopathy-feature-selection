import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from dataloader import create_dataloaders
from feature_extractor import ResNetFeatureExtractor

def extract_and_save(loader, model, split_name, output_dir):
    """
    Passes data through the model and saves features and labels as .npy files.
    """
    all_features = []
    all_labels = []

    print(f"\n--- Starting extraction for {split_name} split ---")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Extracting {split_name}"):
            # Extract features (the model handles GPU transfer internally)
            features = model(images)
            
            # Move to CPU and convert to numpy
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate all batches
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Save as .npy (Faster and more memory-efficient than CSV for 2048 dims)
    feat_path = os.path.join(output_dir, f"{split_name}_features.npy")
    label_path = os.path.join(output_dir, f"{split_name}_labels.npy")
    
    np.save(feat_path, X)
    np.save(label_path, y)

    print(f"Saved {split_name} features: {X.shape} to {feat_path}")
    print(f"Saved {split_name} labels: {y.shape} to {label_path}")

def main():
    # 1. Configuration
    DATA_DIR = "data"
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    BATCH_SIZE = 32
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 2. Initialize Model and Loaders
    extractor = ResNetFeatureExtractor()
    loaders = create_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # 3. Run Pipeline for each split
    for split in ['train', 'val', 'test']:
        if loaders[split] is not None:
            extract_and_save(loaders[split], extractor, split, PROCESSED_DIR)

    print("\n✅ Deep Feature Extraction Pipeline Complete!")

if __name__ == "__main__":
    main()