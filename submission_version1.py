# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
manipes_ibm_flood_data_path = kagglehub.dataset_download('manipes/ibm-flood-data')

print('Data source import complete.')

!pip install terratorch -q
!pip install numpy==2.0.0

import sys
import os
import sys
import numpy as np
import torch

import terratorch
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels

import albumentations
import albumentations as A
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, Resize
)
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import rasterio

"""## Dataset Preparation"""

dataset_path = '/kaggle/input/datasets/manipes/ibm-flood-data/data/'

import albumentations as A

class AddDerivedBands(A.ImageOnlyTransform):
    """
    Appends derived channels to the image at transform time.

    Args:
        band_fns: list of callables. Each receives the image array (H, W, C)
                  in original channel order and must return a (H, W) array
                  for the new channel.

    Channel order for this dataset: [HH, HV, Green, Red, NIR, SWIR]
                                      idx:  0   1     2    3    4     5
    Example band_fns:
        lambda img: img[..., 0] / (img[..., 1] + 1e-6)          # HH/HV ratio
        lambda img: np.log1p(np.clip(img[..., 0], 0, None))      # log(1+HH)
        lambda img: (img[..., 4] - img[..., 3]) /
                    (img[..., 4] + img[..., 3] + 1e-6)           # NDVI proxy
    """
    def __init__(self, band_fns, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.band_fns = band_fns

    def apply(self, img, **params):
        extras = []
        for fn in self.band_fns:
            result = fn(img)
            if result.ndim == 2:
                result = result[..., np.newaxis]
            extras.append(result)
        return np.concatenate([img] + extras, axis=-1).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ()


# --- Define derived bands here. Add / remove callables freely. ---

# Channel indices: HH=0, HV=1, Green=2, Red=3, NIR=4, SWIR=5
DERIVED_BAND_FNS = [
    lambda img: img[..., 0] / (img[..., 1] + 1e-6),          # HH/HV ratio
    # lambda img: np.log1p(np.clip(img[..., 0], 0, None)),      # log(1 + HH)
]

# Approximate mean/std for each derived band (computed from training distribution).
# Re-compute from your training set for best results.
DERIVED_MEANS = [2.27]  #, 6.74]   # HH/HV ratio, log(1+HH)
DERIVED_STDS  = [1.20] #, 0.56]   # HH/HV ratio, log(1+HH)

derived_transform = AddDerivedBands(DERIVED_BAND_FNS)
datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
    batch_size=2,
    num_workers=2,
    num_classes=2,

    # Define dataset paths
    train_data_root=dataset_path+'image',
    train_label_data_root=dataset_path+'label',
    val_data_root=dataset_path+'image',
    val_label_data_root=dataset_path+'label',
    test_data_root=dataset_path+'image',
    test_label_data_root=dataset_path+'label',

    # Define splits
    train_split=dataset_path+'split/train.txt',
    val_split=dataset_path+'split/val.txt',
    test_split=dataset_path+'split/test.txt',

    img_grep='*image.tif',
    label_grep='*label.tif',

    train_transform=[
        derived_transform,
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        Resize(512, 512),
        ToTensorV2(),
    ],
    val_transform=[
        derived_transform,
        Resize(512, 512),
        ToTensorV2(),
    ],
    test_transform=[
        derived_transform,
        Resize(512, 512),
        ToTensorV2(),
    ],

    # Means/stds: original 6 bands + one entry per derived band
    means = [841.1257, 371.6175, 1734.1789, 1588.3142, 1742.8452, 1218.5551] + DERIVED_MEANS,
    stds  = [473.7090, 170.3611,  623.0474,  612.8465,  564.5835,  528.0894] + DERIVED_STDS,
    no_data_replace=0,
    no_label_replace=-1,
    # We use all six bands of the data, so we don't need to define dataset_bands and output_bands.
    predict_data_root = dataset_path + '/prediction/image'
)

# Setup train and val datasets
datamodule.setup("fit")

# checking datasets train split size
train_dataset = datamodule.train_dataset
print(len(train_dataset))

# checking datasets validation split size
val_dataset = datamodule.val_dataset
print(len(val_dataset))

# plotting a few samples
val_dataset.plot(val_dataset[0])
val_dataset.plot(val_dataset[1])

"""## Fine-tuning the IBM Prithvi Model

In this section, we fine-tune the IBM Prithvi geospatial foundation model for flood detection over India. The model, pretrained on global Earth observation data, is adapted to multi-sensor inputs including SAR and optical imagery, and trained to segment flooded areas accurately.

The fine-tuning process leverages high-resolution patches and flood labels derived from Bhuvan inundation maps and water body datasets. During training, the model learns India-specific flood patterns, improving its ability to identify inundated regions under diverse land cover and weather conditions.

Monitoring and checkpointing ensure the best-performing model is saved based on validation metrics.
"""

#Hyperparameters
EPOCHS = 60
BATCH_SIZE = 6
LR = 7.46e-04
WEIGHT_DECAY = 1.69e-05
HEAD_DROPOUT=0.1
FREEZE_BACKBONE = True

BANDS = list(range(1, 6 + len(DERIVED_BAND_FNS) + 1))
NUM_FRAMES = 1
CLASS_WEIGHTS =[0.25, 1.30]
SEED = 0
OUT_DIR='/kaggle/working/'

pl.seed_everything(SEED, workers=True)

# Logger
logger = TensorBoardLogger(
    save_dir=OUT_DIR,
    name="test",
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val/mIoU",
    mode="max",
    dirpath=os.path.join(OUT_DIR, "test", "checkpoints"),
    filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
)

early_stopping_callback = EarlyStopping(
    monitor="val/mIoU",
    patience=20,
    mode="max",
    verbose=True,
)

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    precision="bf16-mixed",
    num_nodes=1,
    logger=logger,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback, early_stopping_callback],
    num_sanity_val_steps=0,
    # limit_predict_batches=1,  # predict only in the first batch for generating plots
)

# DataModule
data_module = datamodule


# Model

decoder_args = dict(
    decoder="UperNetDecoder",
    decoder_channels=256,
    decoder_scale_modules=True,
)

necks = [
    # dict(
    #        name="SelectIndices",
    #        # indices=[2, 5, 8, 11]    # indices for prithvi_vit_100
    #       # indices=[5, 11, 17, 23],   # indices for prithvi_eo_v2_300
    #        # indices=[7, 15, 23, 31]  # indices for prithvi_eo_v2_600
    #    ),
    dict(
            name="ReshapeTokensToImage",
            effective_time_dim=NUM_FRAMES,
        )
    ]

backbone_args = dict(
    backbone_pretrained=True,
    backbone="prithvi_eo_v2_300_tl", # other models are availble like prithvi_eo_v2_300, prithvi_eo_v2_tiny_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
    #backbone_coords_encoding=["time", "location"],
    backbone_bands=BANDS,
    backbone_num_frames=1, # 1 is the default value,
    # backbone_pretrained_cfg_overlay=None
    )

model_args = dict(
    **backbone_args,
    **decoder_args,
    num_classes=2,
    head_dropout=HEAD_DROPOUT,
    necks=necks,
    rescale=True,
)



model = SemanticSegmentationTask(
    model_args=model_args,
    plot_on_val=False,
    class_weights=CLASS_WEIGHTS,
    loss="dice",
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    scheduler="StepLR",
    scheduler_hparams={"step_size": 16, "gamma": 0.909027866016032},
    ignore_index=-1,
    freeze_backbone=FREEZE_BACKBONE,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

trainer.fit(model, datamodule=data_module)

"""## Testing

This section evaluates the fine-tuned Prithvi model on unseen flood events.
"""

datamodule.setup("test")
best_ckpt_path = checkpoint_callback.best_model_path
print(f"Best checkpoint: {best_ckpt_path}")

test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path)

"""## Prediction

This section runs inference using the fine-tuned model and generates flood predictions for unseen data. The predicted flood masks are saved as georeferenced GeoTIFF files, preserving the spatial metadata of the input imagery for downstream analysis and visualization.

"""

datamodule.setup("predict")

predictions = trainer.predict(
    model,
    datamodule=datamodule,
    ckpt_path=best_ckpt_path
)

output_dir = "/kaggle/working/prediction"
os.makedirs(output_dir, exist_ok=True)

for batch_idx, (preds, file_paths) in enumerate(predictions):

    if isinstance(preds, tuple):
        preds = preds[0]

    if preds.ndim == 4:               # [B, C, H, W]
        preds = preds.argmax(dim=1)   # [B, H, W]

    preds = preds.cpu().numpy().astype("int16")

    for i in range(preds.shape[0]):
        arr = preds[i]
        arr[arr < 0] = -1

        ref_path = file_paths[i]
        with rasterio.open(ref_path) as src:
            meta = src.meta.copy()

        meta.update({
            "count": 1,
            "dtype": "int16",
            "nodata": -1,
            "compress": "lzw",
        })

        out_name = (
            os.path.splitext(os.path.basename(ref_path))[0]
            + ".tif"
        )
        out_path = os.path.join(output_dir, out_name)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr, 1)

        print(f"Saved {out_path}")

"""# Convert prediction tif files to Kaggle-style run-length encoding (RLE) Submission csv"""

import numpy as np

def mask_to_rle(mask):
    """
    Convert binary mask to RLE (Kaggle format).
    Mask must be 2D numpy array with values 0 or 1.
    """
    pixels = mask.flatten(order="F")  # column-major
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

import rasterio
import pandas as pd
from pathlib import Path

tif_dir = Path("/kaggle/working/prediction")   # folder with .tif files
output_csv = "/kaggle/working/prediction/submission.csv"

rows = []

for tif_path in sorted(tif_dir.glob("*.tif")):
    with rasterio.open(tif_path) as src:
        mask = src.read(1)

    # Convert to binary
    mask = (mask > 0).astype(np.uint8)

    rle = mask_to_rle(mask)

    rows.append({
        "id": tif_path.name.replace("_image.tif", ""),
        "rle_mask": rle
    })

df = pd.DataFrame(rows)
df = df.replace("", 0).fillna(0) #replace null/ na with zero - kaggle compatible
df.to_csv(output_csv, index=False)
print(f"Saved Kaggle RLE CSV : {output_csv}")