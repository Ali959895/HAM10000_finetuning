import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CocoClassificationDataset(Dataset):
    def __init__(self, vis_root, ann_file, transform=None):
        self.vis_root = vis_root
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.num_classes = 80

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.vis_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        label = torch.zeros(self.num_classes)
        for ann in anns:
            label[ann["category_id"] - 1] = 1  # COCO is 1-indexed

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label
        }
