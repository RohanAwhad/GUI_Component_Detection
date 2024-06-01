# Copied from: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb

import os
import pytorch_lightning as pl
import torch
import torchvision

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor


# ===
# Settings
# ===
ROOT_DIR = "."

LR = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
N_GPUS = 1
MODEL_PATH = os.path.join(ROOT_DIR, "custom-model")

CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

# ds settings
DATA_DIR = os.path.join(ROOT_DIR, "data")
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(DATA_DIR, "train")
VAL_DIRECTORY = os.path.join(DATA_DIR, "valid")
TEST_DIRECTORY = os.path.join(DATA_DIR, "test")
NUM_WORKERS = 4


# ===
# Data
# ===
class CocoDetection(torchvision.datasets.CocoDetection):
  def __init__(
    self,
    image_directory_path: str,
    image_processor,
    train: bool = True
  ):
    annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
    super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
    self.image_processor = image_processor

  def __getitem__(self, idx):
    images, annotations = super(CocoDetection, self).__getitem__(idx)
    image_id = self.ids[idx]
    annotations = {'image_id': image_id, 'annotations': annotations}
    encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
    pixel_values = encoding["pixel_values"].squeeze()
    target = encoding["labels"][0]

    return pixel_values, target


# Dataloader
def collate_fn(batch):
  # DETR authors employ various image sizes during training, making it not possible
  # to directly batch together images. Hence they pad the images to the biggest
  # resolution in a given batch, and create a corresponding binary pixel_mask
  # which indicates which pixels are real/which are padding
  pixel_values = [item[0] for item in batch]
  encoding = image_processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  return {
      'pixel_values': encoding['pixel_values'],
      'pixel_mask': encoding['pixel_mask'],
      'labels': labels
  }

# ===
# Model
# ===
class Detr(pl.LightningModule):
  def __init__(self, lr, lr_backbone, weight_decay):
    super().__init__()
    self.model = DetrForObjectDetection.from_pretrained(
      pretrained_model_name_or_path=CHECKPOINT,
      num_labels=len(id2label),
      ignore_mismatched_sizes=True
    )

    self.lr = lr
    self.lr_backbone = lr_backbone
    self.weight_decay = weight_decay

  def forward(self, pixel_values, pixel_mask): return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

  def common_step(self, batch, batch_idx):
    pixel_values = batch["pixel_values"]
    pixel_mask = batch["pixel_mask"]
    labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

    loss = outputs.loss
    loss_dict = outputs.loss_dict

    return loss, loss_dict

  def training_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)
    # logs metrics for each training_step, and the average across the epoch
    self.log("training_loss", loss)
    for k,v in loss_dict.items(): self.log("train_" + k, v.item())
    return loss

  def validation_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)
    self.log("validation/loss", loss)
    for k, v in loss_dict.items(): self.log("validation_" + k, v.item())
    return loss

  def configure_optimizers(self):
    # DETR authors decided to use different learning rate for backbone
    # you can learn more about it here:
    # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
    # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
    param_dicts = [
      {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad], "lr": self.lr},
      {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
    ]
    return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

  def train_dataloader(self): return TRAIN_DATALOADER
  def val_dataloader(self): return VAL_DATALOADER


if __name__ == '__main__':

  # dataset loading
  image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
  TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
  VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
  TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

  print("Number of training examples:", len(TRAIN_DATASET))
  print("Number of validation examples:", len(VAL_DATASET))
  print("Number of test examples:", len(TEST_DATASET))

  TRAIN_DATALOADER = DataLoader(
    dataset=TRAIN_DATASET,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True
  )

  VAL_DATALOADER = DataLoader(
    dataset=VAL_DATASET,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True
  )

  TEST_DATALOADER = DataLoader(
    dataset=TEST_DATASET,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True
  )

  # model
  model = Detr(lr=LR, lr_backbone=LR_BACKBONE, weight_decay=WEIGHT_DECAY)

  # sanity check
  batch = next(iter(TRAIN_DATALOADER))
  outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])


  # settings
  trainer = Trainer(
    devices=N_GPUS,
    accelerator="gpu",
    max_epochs=NUM_EPOCHS,
    gradient_clip_val=0.1,
    accumulate_grad_batches=1,
    log_every_n_steps=5,
    precision="16-mixed",
  )
  trainer.fit(model)

  # save the model
  model.model.save_pretrained(MODEL_PATH)
