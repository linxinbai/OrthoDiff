import numpy as np
import torch
import torch.nn as nn
import logging
from monai.inferers import SlidingWindowInferer
from utils.metric import dice, hausdorff_distance_95, recall, precision
from trainer import Trainer
from monai.utils import set_determinism
from utils.files_helper import save_new_model_and_delete_last
from data_processing.preprocessor import NiftiPreprocessor
from data_processing.dataset import SkullSegmentationDataset
from data_processing.augmentation import VoxelAugmentation, MedicalAugmentation
from models.basic_unet_denose import BasicUNetDe
from models.basic_unet import BasicUNetEncoder
from models.diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from models.respace import SpacedDiffusion, space_timesteps
from models.resample import UniformSampler
# Add custom combined loss and DiceLoss
from models.losses import CombinedLoss, DiceLoss as CustomDiceLoss
import torch.nn.functional as F
import os
import inspect

# Centralized configuration
import config as CFG

# Determinism for reproducibility
set_determinism(CFG.REPRODUCIBILITY['seed'])

# Logging
logging.basicConfig(
    level=CFG.LOGGING['log_level'],
    format=CFG.LOGGING['log_format'],
    handlers=[
        logging.FileHandler(CFG.LOGGING['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_data_components():
    """Create data processing components for a NIfTI-based pipeline."""
    logger.info("Initializing NIfTI data components...")

    # NIfTI preprocessor with safe kwargs filtering
    sig = inspect.signature(NiftiPreprocessor.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    cfg = {k: v for k, v in CFG.PREPROCESSING.items() if k in allowed}
    nifti_preprocessor = NiftiPreprocessor(**cfg)

    # Augmenters
    train_augmenter = VoxelAugmentation(**CFG.AUGMENTATION_CONFIG)
    medical_augmenter = MedicalAugmentation()

    return nifti_preprocessor, train_augmenter, medical_augmenter


def create_datasets(nifti_preprocessor, train_augmenter, medical_augmenter):
    """Create datasets using NIfTI files (FULL.nii[.gz] + Seg.nii[.gz])."""
    logger.info("Creating datasets (NIfTI)...")

    dataset = SkullSegmentationDataset(
        data_dir=CFG.PATHS['data_dir'],
        nifti_preprocessor=nifti_preprocessor,
        train_augmenter=train_augmenter,
        medical_augmenter=medical_augmenter,
        target_shape=CFG.DATA['target_shape'],
        cache_data=CFG.DATA['cache_data']
    )

    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(
        train_ratio=CFG.SPLIT['train_ratio'],
        val_ratio=CFG.SPLIT['val_ratio'],
        random_seed=CFG.SPLIT['random_seed']
    )

    # Disable train-time augmentations for stability
    train_dataset.apply_augmentation = False
    train_dataset.train_augmenter = None
    train_dataset.medical_augmenter = None
    logger.info("Training-time augmentations are DISABLED for stability.")

    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # ...existing code...
        self.embed_model = BasicUNetEncoder(3, CFG.DATA['num_modality'], CFG.DATA['num_classes'], [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDe(3, CFG.DATA['num_modality'] + CFG.DATA['num_classes'], CFG.DATA['num_classes'], [64, 64, 128, 256, 512, 64],
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        betas = get_named_beta_schedule(CFG.DIFFUSION['beta_schedule'], CFG.DIFFUSION['diffusion_steps'])
        # Always predict x_start (START_X)
        mean_type = ModelMeanType.START_X
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(CFG.DIFFUSION['diffusion_steps'], [CFG.DIFFUSION['diffusion_steps']]),
            betas=betas,
            model_mean_type=mean_type,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(CFG.DIFFUSION['diffusion_steps'], [CFG.DIFFUSION['sample_steps']]),
            betas=betas,
            model_mean_type=mean_type,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(CFG.DIFFUSION['diffusion_steps'])

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise
        elif pred_type == "denoise":
            embeddings = None
            try:
                if CFG.ABLATON.get('USE_ENCODER', True):
                    embeddings = self.embed_model(image)
            except Exception:
                embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)
        elif pred_type == "ddim_sample":
            embeddings = None
            try:
                if CFG.ABLATON.get('USE_ENCODER', True):
                    embeddings = self.embed_model(image)
            except Exception:
                embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, CFG.DATA['num_classes'], *CFG.DATA['target_shape']), model_kwargs={"image": image, "embeddings": embeddings}, eta=CFG.DIFFUSION.get('eta', 0.0))
            sample_out = sample_out["pred_xstart"]
            return sample_out


def compute_class_weights(dist: dict, smooth: float = 1e-6, power: float = 1.0):
    """Compute class weights from voxel counts.
    dist: {class_id(int 1..5): voxel_count}
    Returns a tensor of shape (5,) matching CFG.DATA['class_order'].
    Formula: w_i = (1 / (count_i + smooth))^power, normalized to sum=5.
    If a class has count=0, assign the maximum weight.
    """
    counts = []
    id_map = {"LR":1, "MD":2, "MX":3, "RR":4, "SK":5}
    for name in CFG.DATA['class_order']:
        cid = id_map[name]
        counts.append(float(dist.get(cid, 0)))
    counts_arr = np.array(counts, dtype=np.float64)
    inv = np.power(1.0 / (counts_arr + smooth), power)
    if np.all(counts_arr == 0):
        inv = np.ones_like(inv)
    inv_sum = inv.sum()
    weights = inv * (len(inv) / inv_sum) if inv_sum > 0 else np.ones_like(inv)
    return torch.tensor(weights, dtype=torch.float32)

class SegTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./checkpoints/",
                 master_ip='localhost', master_port=17750, training_script="train.py", class_weights: torch.Tensor = None,
                 dataloader_num_workers: int = 0, pin_memory: bool = False):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script, dataloader_num_workers=dataloader_num_workers, pin_memory=pin_memory)
        self.window_infer = SlidingWindowInferer(roi_size=list(CFG.INFERENCE['sw_roi_size']), sw_batch_size=CFG.INFERENCE['sw_batch_size'], overlap=CFG.INFERENCE['sw_overlap'])
        self.model = DiffUNet()
        self.best_mean_dice = 0.0
        self.class_weights = class_weights if class_weights is not None else torch.ones(CFG.DATA['num_classes'], dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.OPTIMIZER['lr_init'], weight_decay=CFG.OPTIMIZER['weight_decay'])
        self.scheduler = None
        # Combined loss: Dice(sigmoid, no one-hot) + BCEWithLogits with per-channel weights
        # Channel-weighted BCE wrapper
        class ChannelWeightedBCE(nn.Module):
            def __init__(self, class_weights: torch.Tensor | None = None, reduction: str = 'mean'):
                super().__init__()
                self.reduction = reduction
                # Register as buffer to move with the model
                if class_weights is not None:
                    self.register_buffer('class_weights', class_weights.float().clone().detach())
                else:
                    self.class_weights = None
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # pred/target shape: (B,C,D,H,W)
                bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
                if getattr(self, 'class_weights', None) is not None:
                    w = self.class_weights.to(pred.device).view(1, -1, 1, 1, 1)
                    bce = bce * w
                if self.reduction == 'mean':
                    return bce.mean()
                elif self.reduction == 'sum':
                    return bce.sum()
                else:
                    return bce
        # No explicit background channel; include_background=True to keep all 5 foreground channels
        dice_loss = CustomDiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, softmax=False)
        bce_loss = ChannelWeightedBCE(self.class_weights)
        self.loss_fn = CombinedLoss([dice_loss, bce_loss], weights=[0.6, 0.4], names=['dice', 'bce'])

    def _normalize_seg_shape(self, tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Ensure predictions are (B,C,D,H,W) with C=num_classes; convert common layouts."""
        t = tensor
        if t.ndim == 5 and t.shape[1] == num_classes:
            return t
        # channels-last (B,D,H,W,C)
        if t.ndim == 5 and t.shape[-1] == num_classes:
            t = t.permute(0,4,1,2,3).contiguous()
            return t
        # 4D (C,D,H,W) -> add batch
        if t.ndim == 4 and t.shape[0] == num_classes:
            t = t.unsqueeze(0)
            return t
        if t.ndim == 6 and t.shape[-1] == 1:
            t = t.squeeze(-1)
            if t.shape[1] == num_classes:
                return t
        raise ValueError(f"[SHAPE] Cannot normalize prediction tensor shape={tuple(t.shape)} expecting channels={num_classes}")

    def training_step(self, batch):
        image = batch[0]          # (B,1,D,H,W)
        label = batch[1]          # (B,5,D,H,W) one-hot foreground
        num_cls = int(self.class_weights.numel())
        # x_start is label in [-1,1]
        x_start = label * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")
        pred_xstart = self._normalize_seg_shape(pred_xstart, num_cls)
        if pred_xstart.shape != label.shape:
            raise ValueError(f"[SHAPE MISMATCH] pred {tuple(pred_xstart.shape)} vs label {tuple(label.shape)}")
        loss = self.loss_fn(pred_xstart, label)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log("train_loss", loss, step=self.global_step)
        self.log("learning_rate", current_lr, step=self.global_step)
        if hasattr(self.loss_fn, 'last_loss_dict'):
            for k, v in self.loss_fn.last_loss_dict.items():
                self.log(f"loss/{k}", v, step=self.global_step)
        return loss

    def validation_step(self, batch):
        image = batch[0]
        label = batch[1]
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > CFG.INFERENCE['prob_threshold']).float().cpu().numpy()
        target = label.cpu().numpy()
        metrics = []
        for i in range(CFG.DATA['num_classes']):
            o = output[:, i]
            t = target[:, i]
            d = dice(o, t)
            hd = hausdorff_distance_95(o, t)
            rc = recall(o, t)
            pr = precision(o, t)  # PPV
            metrics.extend([d, hd, rc, pr])
        return metrics

    def validation_end(self, mean_val_outputs):
        # mean_val_outputs: [d0, hd0, rc0, pr0, d1, hd1, rc1, pr1, ...]
        if isinstance(mean_val_outputs, (list, tuple, np.ndarray)):
            vals = list(mean_val_outputs)
            num_cls = CFG.DATA['num_classes']
            mean_dices = [vals[4 * i + 0] for i in range(num_cls)]
            mean_hd95s = [vals[4 * i + 1] for i in range(num_cls)]
            mean_recalls = [vals[4 * i + 2] for i in range(num_cls)]
            mean_precisions = [vals[4 * i + 3] for i in range(num_cls)]
            md = float(np.nanmean(mean_dices)) if len(mean_dices) else 0.0
            mhd = float(np.nanmean(mean_hd95s)) if len(mean_hd95s) else 0.0
            mrc = float(np.nanmean(mean_recalls)) if len(mean_recalls) else 0.0
            mpr = float(np.nanmean(mean_precisions)) if len(mean_precisions) else 0.0
        else:
            md = float(mean_val_outputs)
            mhd = 0.0
            mrc = 0.0
            mpr = 0.0
        self.log("val/mean_dice", md, step=self.epoch)
        self.log("val/mean_hd95", mhd, step=self.epoch)
        self.log("val/mean_recall", mrc, step=self.epoch)
        self.log("val/mean_precision", mpr, step=self.epoch)  # PPV
        # Best model selected by mean_dice
        if md > self.best_mean_dice:
            self.best_mean_dice = md
            save_new_model_and_delete_last(self.model,
                                            os.path.join(CFG.PATHS['model_save_dir'],
                                            f"best_model_{md:.4f}.pt"),
                                            delete_symbol="best_model")
        save_new_model_and_delete_last(self.model,
                                        os.path.join(CFG.PATHS['model_save_dir'],
                                        f"final_model_{md:.4f}.pt"),
                                        delete_symbol="final_model")
        print(f"val: mean_dice={md:.4f}, mean_hd95={mhd:.4f}, mean_recall={mrc:.4f}, mean_precision={mpr:.4f}")


def main():
    """Entry point for training."""
    logger.info("Starting skull segmentation training (NIfTI pipeline)...")
    try:
        # Data components
        nifti_preprocessor, train_augmenter, medical_augmenter = create_data_components()

        # Datasets
        train_dataset, val_dataset, test_dataset = create_datasets(
            nifti_preprocessor, train_augmenter, medical_augmenter
        )

        # Use 5-class distribution
        class_weights_tensor = None
        try:
            if CFG.TRAINING.get('compute_class_weights', False):
                # If enabled, fallback to uniform weights to avoid slow scans and unresolved sampling util here
                logger.info("[CLASS WEIGHTS] compute_class_weights=True but sampling util disabled; using uniform weights.")
                class_weights_tensor = torch.ones(CFG.DATA['num_classes'], dtype=torch.float32)
            else:
                logger.info("[CLASS WEIGHTS] Skipped computation (compute_class_weights=False); using uniform weights.")
                class_weights_tensor = torch.ones(CFG.DATA['num_classes'], dtype=torch.float32)
        except Exception as e:
            logger.warning(f"[CLASS WEIGHTS] failed: {e}")
            class_weights_tensor = torch.ones(CFG.DATA['num_classes'], dtype=torch.float32)

        # Number of classes
        num_classes = train_dataset.get_num_classes()
        logger.info(f"Detected {num_classes} classes")

        trainer = SegTrainer(env_type=CFG.RUNTIME['env'],
                             max_epochs=CFG.TRAINING['max_epoch'],
                             batch_size=CFG.TRAINING['batch_size'],
                             device=CFG.RUNTIME['device'],
                             logdir=CFG.PATHS['logdir'],
                             val_every=CFG.TRAINING['val_every'],
                             num_gpus=CFG.RUNTIME['num_gpus'],
                             master_port=CFG.DISTRIBUTED['master_port'],
                             training_script=__file__,
                             dataloader_num_workers=CFG.TRAINING['dataloader_num_workers'],
                             pin_memory=CFG.TRAINING['pin_memory'],
                             class_weights=class_weights_tensor)
        # Cosine annealing scheduler
        if CFG.OPTIMIZER['use_cosine_scheduler']:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            if CFG.OPTIMIZER['cosine_step_level']:
                # Estimate total steps = max_epoch * steps_per_epoch
                temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.TRAINING['batch_size'], shuffle=False)
                steps_per_epoch = len(temp_loader)
                total_steps = CFG.TRAINING['max_epoch'] * steps_per_epoch
                scheduler = CosineAnnealingLR(trainer.optimizer, T_max=total_steps, eta_min=CFG.OPTIMIZER['lr_min'])
                trainer.scheduler = scheduler  # step-level scheduler, manual step
                logger.info(f"Using STEP-level CosineAnnealingLR: initial_lr={CFG.OPTIMIZER['lr_init']}, min_lr={CFG.OPTIMIZER['lr_min']}, T_max={total_steps}")
            else:
                scheduler = CosineAnnealingLR(trainer.optimizer, T_max=CFG.TRAINING['max_epoch'], eta_min=CFG.OPTIMIZER['lr_min'])
                logger.info(f"Using EPOCH-level CosineAnnealingLR: initial_lr={CFG.OPTIMIZER['lr_init']}, min_lr={CFG.OPTIMIZER['lr_min']}, T_max={CFG.TRAINING['max_epoch']}")
        else:
            scheduler = None
        logger.info("Starting training...")
        trainer.train(train_dataset=train_dataset, val_dataset=val_dataset, scheduler=None if CFG.OPTIMIZER['cosine_step_level'] else scheduler)
        logger.info("Training completed!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
