import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class APTOSDataset(Dataset):
    """
    Custom Dataset for APTOS 2019 images.
    Modified to return Binary Labels (0: No DR, 1: DR).
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        self.image_col = self.df.columns[0]
        self.label_col = self.df.columns[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx][self.image_col]
        img_name = f"{img_id}.png" if not str(img_id).endswith('.png') else img_id
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        # --- Updated Binary Logic ---
        original_label = self.df.iloc[idx][self.label_col]
        # 0 remains 0 (No DR), while 1, 2, 3, 4 all become 1 (DR)
        label = 1 if original_label > 0 else 0
        # ----------------------------

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    transform = get_transforms()
    
    # Building paths
    train_csv = os.path.join(data_dir, "raw", "train_1.csv")
    val_csv   = os.path.join(data_dir, "raw", "valid.csv")
    test_csv  = os.path.join(data_dir, "raw", "test.csv")
    
    train_dir = os.path.join(data_dir, "raw", "train_images")
    val_dir   = os.path.join(data_dir, "raw", "val_images")
    test_dir  = os.path.join(data_dir, "raw", "test_images")

    train_ds = APTOSDataset(train_csv, train_dir, transform=transform)
    val_ds   = APTOSDataset(val_csv, val_dir, transform=transform)
    test_ds  = APTOSDataset(test_csv, test_dir, transform=transform)

    loaders = {
        # pin_memory=True speeds up data transfer to GPU
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True),
        'val':   DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    }
    
    return loaders