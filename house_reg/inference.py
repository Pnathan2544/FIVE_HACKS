"""Inference-only script — loads saved model and generates submission.csv"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

base_dir = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Config
img_size = 224
batch_size = 32
num_workers = 0  # Windows doesn't support multiprocessing workers well

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["id"] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


# Load model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
model.load_state_dict(torch.load(os.path.join(base_dir, "best_model_fold0.pth"), weights_only=True))
model.to(device)
model.eval()

# Load test data
test_df = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
test_ds = TestDataset(test_df, os.path.join(base_dir, "test", "test"), transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Predict
all_preds = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)

preds = np.concatenate(all_preds)
test_df["answer"] = (preds > 0.5).astype(int)
test_df[["id", "answer"]].to_csv(os.path.join(base_dir, "submission.csv"), index=False)

print(f"\nSubmission saved! Distribution:\n{test_df['answer'].value_counts()}")
print("Done!")