import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from data_processing.preprocessor import NiftiPreprocessor
from models.basic_unet_denose import BasicUNetDe
from models.basic_unet import BasicUNetEncoder
from models.diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from models.respace import SpacedDiffusion, space_timesteps
from monai.inferers import SlidingWindowInferer
import nibabel as nib
try:
    from scipy.ndimage import zoom as nd_zoom
except ImportError:
    nd_zoom = None

import config as CFG

number_modality = CFG.DATA['num_modality']
number_targets = CFG.DATA['num_classes']
target_shape = CFG.DATA['target_shape']

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64],
                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        betas = get_named_beta_schedule(CFG.DIFFUSION['beta_schedule'], CFG.DIFFUSION['diffusion_steps'])
        # Always predict x_start (START_X)
        mean_type = ModelMeanType.START_X
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(CFG.DIFFUSION['diffusion_steps'], [CFG.DIFFUSION['diffusion_steps']]),
                                        betas=betas,
                                        model_mean_type=mean_type,
                                        model_var_type=ModelVarType.FIXED_LARGE,
                                        loss_type=LossType.MSE)
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(CFG.DIFFUSION['diffusion_steps'], [CFG.DIFFUSION['sample_steps']]),
                                            betas=betas,
                                            model_mean_type=mean_type,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE)

    def forward(self, image=None, pred_type="ddim_sample"):
        if pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model, (1, number_targets, *target_shape),
                model_kwargs={"image": image, "embeddings": embeddings},
                eta=CFG.DIFFUSION.get('eta', 0.0),
            )
            sample_out = sample_out["pred_xstart"]
            return sample_out
        elif pred_type == "ddim_fusion":
            # Uncertainty-weighted fusion across DDIM steps using S trajectories.
            # S = UNCERTAINTY_PREDICTIONS_PER_STEP; setting eta>0 introduces stochasticity
            S = int(CFG.ABLATON.get('UNCERTAINTY_PREDICTIONS_PER_STEP', 1))
            if S <= 0:
                S = 1
            embeddings = self.embed_model(image)
            all_step_probs = []  # list of length T; each item shape (S, C,D,H,W)
            T = CFG.DIFFUSION['sample_steps']
            for s in range(S):
                out = self.sample_diffusion.ddim_sample_loop(
                    self.model, (1, number_targets, *target_shape),
                    model_kwargs={"image": image, "embeddings": embeddings},
                    eta=CFG.DIFFUSION.get('eta', 0.0),
                )
                step_preds = out.get('all_samples', None)
                if not step_preds:
                    # fallback to last step only
                    last = out['pred_xstart']
                    step_preds = [last]
                # convert to probabilities
                step_probs = [torch.sigmoid(t.cuda() if t.is_cuda else t) for t in step_preds]
                # ensure length T by padding/truncation
                if len(step_probs) != T:
                    if len(step_probs) < T:
                        step_probs = step_probs + [step_probs[-1]] * (T - len(step_probs))
                    else:
                        step_probs = step_probs[:T]
                # stack into (T,1,C,D,H,W) then squeeze batch
                step_probs = [p.squeeze(0) for p in step_probs]
                all_step_probs.append(step_probs)
            # all_step_probs: list S with each a list length T of (C,D,H,W)
            # compute per-step mean prob and entropy uncertainty
            fused = None
            weights = []
            pbar_list = []
            for i in range(T):
                ps_i = torch.stack([all_step_probs[s][i] for s in range(S)], dim=0)  # (S,C,D,H,W)
                pbar_i = ps_i.mean(dim=0)  # (C,D,H,W)
                # entropy per voxel/channel, then reduce across channels
                eps = 1e-6
                ent = -(pbar_i * (pbar_i.clamp(min=eps, max=1-eps)).log()).sum(dim=0) / max(1, number_targets)
                # weight: exp(sigmoid(i/scale) * (1 - u_i))
                scale = float(CFG.ABLATON.get('FUSION_SIGMOID_SCALE', 10.0))
                wi = torch.exp(torch.sigmoid(torch.tensor(i / max(scale, 1e-6), device=pbar_i.device)) * (1.0 - ent))  # (D,H,W)
                weights.append(wi)
                pbar_list.append(pbar_i)
            # normalize weights across steps if enabled
            if CFG.ABLATON.get('NORMALIZE_FUSION_WEIGHTS', True):
                wsum = torch.stack(weights, dim=0).sum(dim=0).clamp(min=1e-6)
                weights = [w / wsum for w in weights]
            # fuse
            Y = 0
            for i in range(T):
                Y = Y + pbar_list[i] * weights[i]
            return Y.unsqueeze(0)  # add batch dim
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")

class SkullSegmentationPredictor:
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or CFG.RUNTIME['device']
        self.model = self._load_model()
        # Sliding-window ROI matches model input shape
        self.window_inferer = SlidingWindowInferer(roi_size=list(CFG.INFERENCE['sw_roi_size']), sw_batch_size=CFG.INFERENCE['sw_batch_size'], overlap=CFG.INFERENCE['sw_overlap'])
        # NIfTI preprocessor
        self.nifti_preprocessor = NiftiPreprocessor(target_shape=target_shape, intensity_norm=True, allow_resize=True)

    def _load_model(self):
        model = DiffUNet().to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif isinstance(state_dict, dict) and len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k[7:] if k.startswith("module.") else k
                new_state_dict[new_k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.")
        return model

    def predict(self, input_nifti_path: str):
        print(f"Processing file: {input_nifti_path}")
        # Read original image to get original dimensions and for later restoration
        orig_image = self.nifti_preprocessor.load_nifti(input_nifti_path)
        orig_shape = orig_image.shape  # (D,H,W)
        image_voxel = self.nifti_preprocessor.process_image(input_nifti_path)  # may be resized to target_shape
        if np.sum(image_voxel) == 0:
            print("Input voxel is empty.")
            empty_probs = np.zeros((number_targets, *target_shape), dtype=np.float32)
            # If original size differs, expand/resize to match
            if orig_shape != target_shape:
                empty_probs = np.zeros((number_targets, *orig_shape), dtype=np.float32)
            return empty_probs, orig_shape
        image_tensor = torch.from_numpy(image_voxel).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            if CFG.ABLATON.get('USE_DIFFUSION_FUSION', False):
                logits = self.window_inferer(image_tensor, self.model, pred_type="ddim_fusion")
            else:
                logits = self.window_inferer(image_tensor, self.model, pred_type="ddim_sample")
            # If fusion enabled, logits may already be probabilities
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy() if not CFG.ABLATON.get('USE_DIFFUSION_FUSION', False) else logits.squeeze(0).cpu().numpy()
        # Resize back to original size if needed
        if tuple(orig_shape) != target_shape:
            if nd_zoom is None:
                print("scipy not installed, cannot resample to original size, returning model size.")
            else:
                scale = tuple(float(o)/float(t) for o,t in zip(orig_shape, target_shape))
                resized = []
                for c in range(probs.shape[0]):
                    # Linear interpolation for probabilities (order=1)
                    resized.append(nd_zoom(probs[c], scale, order=1, mode='constant', cval=0.0))
                probs = np.stack(resized, axis=0)
        return probs, orig_shape

def save_segmentation_as_nifti(pred_probs: np.ndarray, input_ref_path: str, output_path: Path):
    """Save 5-channel probabilities as a single-channel label map (foreground 1..5, background 0)."""
    assert pred_probs.ndim == 4 and pred_probs.shape[0] == number_targets, "pred_probs shape must be (C, D, H, W)"
    max_prob = pred_probs.max(axis=0)
    argmax_cls = pred_probs.argmax(axis=0)  # 0..C-1 -> 1..C
    label_map = (argmax_cls + 1).astype(np.int16)
    # Background remains 0 only if a dedicated rule is applied; here we keep pure argmax without thresholding.
    affine = np.eye(4, dtype=np.float32)
    try:
        ref_img = nib.load(str(input_ref_path))
        aff = getattr(ref_img, 'affine', None)
        if aff is not None:
            affine = aff
        else:
            sform = getattr(ref_img, 'get_sform', lambda: None)()
            qform = getattr(ref_img, 'get_qform', lambda: None)()
            if sform is not None:
                affine = sform
            elif qform is not None:
                affine = qform
    except Exception:
        pass
    out_img = nib.Nifti1Image(label_map, affine)
    nib.save(out_img, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Skull Segmentation Prediction Script (NIfTI)")
    parser.add_argument("--input", type=str, required=True, help="Path to the input FULL.nii.gz (or .nii) file or its parent directory containing FULL.nii.gz.")
    parser.add_argument("--model", type=str, default=str(Path(CFG.PATHS['model_save_dir'])/ 'best_model_0.8900.pt'), help="Path to the trained model checkpoint.")
    parser.add_argument("--device", type=str, default=CFG.RUNTIME['device'], help="Device for inference (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--output", type=str, default=CFG.PATHS['pred_output_dir'], help="Output directory for prediction NIfTI.")
    args = parser.parse_args()

    input_path = Path(args.input)
    nifti_file_path = None
    if input_path.is_dir():
        candidate1 = input_path / "FULL.nii.gz"
        candidate2 = input_path / "FULL.nii"
        if candidate1.exists():
            nifti_file_path = str(candidate1)
        elif candidate2.exists():
            nifti_file_path = str(candidate2)
        else:
            print(f"Error: 'FULL.nii[.gz]' not found in directory '{input_path}'.")
            sys.exit(1)
    elif input_path.is_file() and (str(input_path).lower().endswith('.nii.gz') or str(input_path).lower().endswith('.nii')):
        nifti_file_path = str(input_path)
    else:
        print(f"Error: Input path '{args.input}' is not a valid NIfTI file or directory containing the file.")
        sys.exit(1)

    predictor = SkullSegmentationPredictor(model_path=args.model, device=args.device)
    pred_probs, orig_shape = predictor.predict(nifti_file_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "Seg_pred.nii.gz"
    save_segmentation_as_nifti(pred_probs, nifti_file_path, out_path)

    print(f"Prediction NIfTI saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()