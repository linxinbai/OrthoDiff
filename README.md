# OrthoDiff

Repository structure

```
OrthoDiff/
├─ README.md
├─ LICENSE
├─ Makefile
├─ requirements.txt
├─ config.py
├─ launch.py
├─ train.py
├─ test.py
├─ predict.py
├─ trainer.py
├─ checkpoints/
│  └─ model/
├─ data_processing/
│  ├─ augmentation.py
│  ├─ dataset.py
│  └─ preprocessor.py
├─ models/
│  ├─ basic_unet.py
│  ├─ basic_unet_denose.py
│  ├─ diffusion.py
│  ├─ losses.py
│  ├─ lightweight_blocks.py
│  ├─ nn.py
│  ├─ resample.py
│  ├─ respace.py
│  ├─ unet.py
│  ├─ fp16_util.py
│  ├─ logger.py
│  └─ __init__.py
├─ utils/
│  ├─ files_helper.py
│  ├─ log_image.py
│  ├─ lr_scheduler.py
│  ├─ metric.py
│  ├─ script_util.py
│  └─ train_util.py
├─ predictions/
└─ __pycache__/
```

OrthoDiff is a diffusion-based model for orthogonal image segmentation and denoising.

This repository includes training, testing, and prediction scripts, along with data processing utilities and model components.

## Features
- Training and evaluation pipelines (`train.py`, `test.py`, `trainer.py`)
- Prediction/inference utilities (`predict.py`)
- Data preprocessing and augmentation (`data_processing/`)
- UNet-based diffusion models (`models/`)
- Utilities for metrics, logging, and LR scheduling (`utils/`)

## Quickstart

### 1) Environment
Create a Python 3.10+ environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: For PyTorch, choose the appropriate CUDA version per your system.

### 2) Configuration
Adjust settings in `config.py` as needed. You can also use `launch.py` for orchestrated runs.

### 3) Training
Run training:

```bash
python train.py
```

Optional flags:
- dataset/path is set via `config.py` (`PATHS.data_dir`)
- device via `config.py` (`RUNTIME.device`)

### 4) Testing
Run testing/evaluation:

```bash
python test.py
```

Optional flags:
- model path via `config.py` (`PATHS.test_model_path`)
- device via `config.py` (`RUNTIME.device`)

### 5) Prediction
Run prediction:

```bash
python predict.py --input <FULL.nii[.gz] or dir> --model <checkpoint.pt> --device <cuda:0|cpu> --output <dir>
```

- `--input`: NIfTI file path or directory containing `FULL.nii[.gz]`
- `--model`: checkpoint path
- `--device`: inference device
- `--output`: output directory for NIfTI

## Configuration details (`config.py`)
`config.py` centralizes all runtime and experiment settings. Key sections:

- PATHS
  - `data_dir`: Directory containing training/validation data. Defaults to `./data`. Set this to your dataset path.
  - `logdir`: Base folder for checkpoints (default `./checkpoints/`).
  - `model_save_dir`: Derived from `logdir` (set automatically to `./checkpoints/model`).
  - `pred_output_dir`: Output folder for predictions (default `predictions`).
  - `test_model_path`: Path to a specific model checkpoint for testing/prediction. Defaults to `None`; pass via CLI or set here.

- RUNTIME
  - `env`: Runtime environment (e.g., `pytorch`).
  - `device`: CUDA device or CPU (e.g., `cuda:0` or `cpu`).
  - `num_gpus`: Number of GPUs to use.

- TRAINING
  - `max_epoch`, `batch_size`, `val_every`: Core training loop settings.
  - `dataloader_num_workers`, `pin_memory`: DataLoader performance knobs.
  - `compute_class_weights`: Whether to compute class weights at startup (can slow start-up).
  - `class_weight_samples`: Number of samples/files used when computing class weights.

- OPTIMIZER
  - `lr_init`, `lr_min`, `weight_decay`: Optimizer and LR schedule parameters.
  - `use_cosine_scheduler`: Enable cosine annealing.
  - `cosine_step_level`: Step-level (`True`) vs epoch-level (`False`) scheduling.

- SPLIT
  - `train_ratio`, `val_ratio`, `test_ratio`, `random_seed`: Dataset splits.

- PREPROCESSING
  - `target_shape`: Intended 3D volume shape.
  - `intensity_norm`: Intensity normalization toggle.
  - `clip_percentiles` / `clip_range`: Intensity clipping settings.
  - `skull_binary`: Whether to binarize skull output.
  - `allow_resize`: Permit resizing to target shape.

- AUGMENTATION_CONFIG
  - Ranges for rotation/translation/scaling, noise variance, elastic transform, and probability of applying augmentation.

- INFERENCE
  - Sliding window settings (`sw_roi_size`, `sw_batch_size`, `sw_overlap`), and `prob_threshold` for binarization.

- DIFFUSION
  - `diffusion_steps`, `sample_steps`, `beta_schedule`, `eta` (DDIM stochasticity).

- LOGGING
  - `log_file`, `log_level`, `log_format`: Logging outputs and formatting.

- DISTRIBUTED
  - `master_ip`, `master_port`: Distributed training config (single-machine by default).

- ABLATON (Ablation toggles)
  - `USE_ENCODER`, `USE_LEARNABLE_FUSION`: Enable model components.
  - `UNCERTAINTY_PREDICTIONS_PER_STEP`: Repeated predictions per DDIM step for uncertainty.
  - `USE_DIFFUSION_FUSION`, `FUSION_SIGMOID_SCALE`, `NORMALIZE_FUSION_WEIGHTS`: Fusion behavior controls.

## Repo Structure
- `models/`: UNet and diffusion components
- `data_processing/`: dataset, augmentation, and preprocessing
- `utils/`: helper functions (metrics, logging, LR schedulers, script/train utils)
- `train.py`, `test.py`, `predict.py`: entry points
- `trainer.py`: training loop utilities
- `config.py`: global configuration
- `requirements.txt`: dependencies

## License
This project is licensed under the MIT License. See `LICENSE` for details.
