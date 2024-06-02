# Copied from: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb

# Specifically for ampere GPUs
import torch
# Faster, but less precise
torch.set_float32_matmul_precision("high")

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
ROOT_DIR = "/home/rawhad/personal_jobs/GUI_Detection/GUI_Component_Detection"

LR = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100

N_GPUS = torch.cuda.device_count()
BATCH_PER_GPU = 8
FINAL_BATCH_SIZE = 32
BATCH_SIZE = BATCH_PER_GPU * N_GPUS
GRAD_ACCUMULATION = int(FINAL_BATCH_SIZE // BATCH_SIZE)
print(f'Batch Size:                   {BATCH_SIZE:3d}')
print(f'Gradient Accumulation Steps:  {GRAD_ACCUMULATION:3d}')

MODEL_PATH = os.path.join(ROOT_DIR, "custom-model")

CHECKPOINT = 'facebook/detr-resnet-101'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

# ds settings
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(DATA_DIR, "train")
VAL_DIRECTORY = os.path.join(DATA_DIR, "valid")
TEST_DIRECTORY = os.path.join(DATA_DIR, "test")

NUM_WORKERS = min(2, (os.cpu_count() - 2) // 3)


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
  def __init__(self, lr, lr_backbone, weight_decay, id2label):
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

# ===
# Custom Callback to save HF model
# ===
class SaveModelCallback(pl.Callback):
  def __init__(self, model_path): self.model_path = model_path
  def on_validation_epoch_end(self, trainer, pl_module): pl_module.model.save_pretrained(self.model_path)


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

  categories = TRAIN_DATASET.coco.cats
  id2label = {k: v['name'] for k,v in categories.items()}
  # model
  model = Detr(lr=LR, lr_backbone=LR_BACKBONE, weight_decay=WEIGHT_DECAY, id2label=id2label)

  # sanity check
  batch = next(iter(TRAIN_DATALOADER))
  outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])


  # settings
  trainer = Trainer(
    devices=N_GPUS,
    accelerator="gpu",
    max_epochs=NUM_EPOCHS,
    gradient_clip_val=0.1,
    accumulate_grad_batches=GRAD_ACCUMULATION,
    log_every_n_steps=5,
    precision="bf16-mixed",
    benchmark=True,  # cuDNN benchmark to speed training for constant input sizes
    callbacks=[SaveModelCallback(model_path=MODEL_PATH)],
    # following flags are set for understanding speed
    #callbacks=[pl.callbacks.DeviceStatsMonitor(cpu_stats=False)],
    #profiler='simple',
    #fast_dev_run=10,
  )
  trainer.fit(model)
