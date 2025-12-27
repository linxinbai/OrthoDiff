import torch
import torch.nn as nn
import os
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

# 本地模块导入
from utils.metric import dice, hausdorff_distance_95, recall, precision
from trainer import Trainer
from models.diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from models.respace import SpacedDiffusion, space_timesteps
from models.basic_unet_denose import BasicUNetDe
from models.basic_unet import BasicUNetEncoder
from data_processing.dataset import SkullSegmentationDataset
from data_processing.preprocessor import NiftiPreprocessor

import config as CFG
import logging

# 设置随机种子以保证可复现性
set_determinism(CFG.REPRODUCIBILITY['seed'])

# 配置日志输出到文件 test.log 和控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 配置参数 ---
# 数据路径与模型路径
data_dir = CFG.PATHS['data_dir']
model_path = CFG.PATHS['test_model_path']

# 与 train.py 保持一致的参数
number_modality = CFG.DATA['num_modality']
number_targets = CFG.DATA['num_classes']
label_names = CFG.DATA['label_names']
# 统一到 target_shape
target_shape = CFG.DATA['target_shape']


class DiffUNet(nn.Module):
    """扩散模型定义，与 train.py 中的结构完全一致"""

    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64],
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        betas = get_named_beta_schedule(CFG.DIFFUSION['beta_schedule'], CFG.DIFFUSION['diffusion_steps'])
        # Always predict x_start (START_X)
        mean_type = ModelMeanType.START_X
        # 用于推理的采样扩散器
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(CFG.DIFFUSION['diffusion_steps'], [CFG.DIFFUSION['sample_steps']]),
                                                betas=betas,
                                                model_mean_type=mean_type,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        # 推理时支持 "ddim_sample" 与 "ddim_fusion"
        if pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model,
                (1, number_targets, *target_shape),  # 注意：尺寸应与 window_infer 匹配
                model_kwargs={"image": image, "embeddings": embeddings},
                eta=CFG.DIFFUSION.get('eta', 0.0),
            )
            return sample_out["pred_xstart"]
        elif pred_type == "ddim_fusion":
            # Uncertainty-weighted multi-step fusion using S stochastic trajectories
            S = int(CFG.ABLATON.get('UNCERTAINTY_PREDICTIONS_PER_STEP', 1))
            if S <= 0:
                S = 1
            embeddings = self.embed_model(image)
            T = CFG.DIFFUSION['sample_steps']
            all_step_probs = []  # list S, each a list length T of (C,D,H,W)
            for s in range(S):
                out = self.sample_diffusion.ddim_sample_loop(
                    self.model,
                    (1, number_targets, *target_shape),
                    model_kwargs={"image": image, "embeddings": embeddings},
                    eta=CFG.DIFFUSION.get('eta', 0.0),
                )
                step_preds = out.get('all_samples', None)
                if not step_preds:
                    step_preds = [out['pred_xstart']]
                step_probs = [torch.sigmoid(t.to(image.device)) for t in step_preds]
                if len(step_probs) != T:
                    if len(step_probs) < T:
                        step_probs = step_probs + [step_probs[-1]] * (T - len(step_probs))
                    else:
                        step_probs = step_probs[:T]
                step_probs = [p.squeeze(0) for p in step_probs]
                all_step_probs.append(step_probs)
            # per-step mean prob and entropy-based uncertainty
            weights = []
            pbar_list = []
            for i in range(T):
                ps_i = torch.stack([all_step_probs[s][i] for s in range(S)], dim=0)  # (S,C,D,H,W)
                pbar_i = ps_i.mean(dim=0)  # (C,D,H,W)
                eps = 1e-6
                ent = -(pbar_i * (pbar_i.clamp(min=eps, max=1-eps)).log()).sum(dim=0) / max(1, number_targets)
                scale = float(CFG.ABLATON.get('FUSION_SIGMOID_SCALE', 10.0))
                wi = torch.exp(torch.sigmoid(torch.tensor(i / max(scale, 1e-6), device=pbar_i.device)) * (1.0 - ent))
                weights.append(wi)
                pbar_list.append(pbar_i)
            if CFG.ABLATON.get('NORMALIZE_FUSION_WEIGHTS', True):
                wsum = torch.stack(weights, dim=0).sum(dim=0).clamp(min=1e-6)
                weights = [w / wsum for w in weights]
            Y = 0
            for i in range(T):
                Y = Y + pbar_list[i] * weights[i]
            return Y.unsqueeze(0)  # (B=1,C,D,H,W) probs
        else:
            raise NotImplementedError("Only 'ddim_sample' or 'ddim_fusion' pred_type is supported during testing.")


class SkullTrainer(Trainer):
    """用于颅骨分割测试的 Trainer"""

    def __init__(self, device="cpu", logdir="./checkpoints_test/"):
        # 调用父类构造函数，简化参数
        super().__init__(env_type="pytorch", max_epochs=1, batch_size=1, device=device, logdir=logdir)

        # 定义滑窗推理器
        self.window_infer = SlidingWindowInferer(roi_size=list(CFG.INFERENCE['sw_roi_size']),
                                                 sw_batch_size=CFG.INFERENCE['sw_batch_size'],
                                                 overlap=CFG.INFERENCE['sw_overlap'],
                                                 mode="gaussian")
        self.model = DiffUNet()

    def get_input(self, batch):
        """从数据加载器中获取输入图像和标签"""
        image, label = batch
        return image, label

    def validation_step(self, batch):
        """执行单个验证/测试步骤，返回扩展调试指标列表: [dice, hd95, recall, pred_vox, gt_vox, soft_dice] * C"""
        image, label = self.get_input(batch)
        if CFG.ABLATON.get('USE_DIFFUSION_FUSION', False):
            output = self.window_infer(image, self.model, pred_type="ddim_fusion")  # already probs
            probs = output
        else:
            output = self.window_infer(image, self.model, pred_type="ddim_sample")
            probs = torch.sigmoid(output)
        output_bin = (probs > 0.5).float().cpu().numpy()  # (B,C,D,H,W)
        target = label.cpu().numpy()
        metrics = []
        for i in range(number_targets):
            o = output_bin[:, i]
            t = target[:, i]
            d = dice(o, t)
            hd = hausdorff_distance_95(o, t)
            rc = recall(o, t)
            pr = precision(o, t)
            metrics.extend([d, hd, rc, pr])
        return metrics


if __name__ == "__main__":
    # 1. 创建数据预处理器（NIfTI）
    nifti_preprocessor = NiftiPreprocessor(target_shape=target_shape, intensity_norm=True, allow_resize=True)

    # 2. 创建测试数据集（NIfTI 扫描 FULL/Seg 文件对）
    full_dataset = SkullSegmentationDataset(
        data_dir=data_dir,
        nifti_preprocessor=nifti_preprocessor,
        apply_augmentation=False,
        cache_data=False
    )

    # 划分数据集以获取测试集部分
    _, _, test_dataset = full_dataset.split_dataset(train_ratio=CFG.SPLIT['train_ratio'], val_ratio=CFG.SPLIT['val_ratio'], random_seed=CFG.SPLIT['random_seed'])

    print(f"测试集样本数量: {len(test_dataset)}")

    # 3. 初始化 Trainer
    trainer = SkullTrainer(device=CFG.RUNTIME['device'])

    # 4. 加载预训练模型
    if os.path.exists(model_path):
        print(f"正在从 '{model_path}' 加载模型...")
        trainer.load_state_dict(model_path)
    else:
        print(f"错误: 模型文件 '{model_path}' 不存在。请检查路径。")
        exit()

    # 5. 执行评估
    print("开始在测试集上进行评估...")
    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_dataset)

    # v_mean 当前结构: [d0, hd0, rc0, pr0, d1, hd1, rc1, pr1, ...]
    # 取每类 dice/hd/recall 列表
    per_class_dice = []
    per_class_hd95 = []
    per_class_recall = []
    per_class_precision = []
    for i in range(number_targets):
        base = i * 4
        per_class_dice.append(v_mean[base + 0])
        per_class_hd95.append(v_mean[base + 1])
        per_class_recall.append(v_mean[base + 2])
        per_class_precision.append(v_mean[base + 3])
    # 逐类写入日志：Dice 和 HD95
    for i in range(number_targets):
        name = label_names[i] if i < len(label_names) else f"class_{i}"
        d = per_class_dice[i]
        hd = per_class_hd95[i]
        logger.info(f"[PER-CLASS] {name}: Dice={d:.4f}, HD95={hd:.4f}")

    # 计算总体平均（忽略 NaN）
    import math
    def _mean_valid(vals):
        valid = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        return sum(valid)/len(valid) if valid else 0.0
    mean_dice = _mean_valid(per_class_dice)
    mean_hd95 = _mean_valid(per_class_hd95)
    mean_recall = _mean_valid(per_class_recall)
    mean_precision = _mean_valid(per_class_precision)

    # 计算每类 IoU（基于二值掩码：IoU = intersection / union），以及 mean_IoU
    # 重新进行一次滑窗推理以获取二值掩码用于 IoU 计算
    per_class_iou = []
    try:
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        iou_sums = [0.0 for _ in range(number_targets)]
        iou_cnts = [0.0 for _ in range(number_targets)]
        with torch.no_grad():
            for batch in loader:
                image, label = batch
                if CFG.ABLATON.get('USE_DIFFUSION_FUSION', False):
                    output = trainer.window_infer(image.to(CFG.RUNTIME['device']), trainer.model, pred_type="ddim_fusion")
                    probs = output
                else:
                    output = trainer.window_infer(image.to(CFG.RUNTIME['device']), trainer.model, pred_type="ddim_sample")
                    probs = torch.sigmoid(output)
                output_bin = (probs > 0.5).float().cpu().numpy()
                target = label.cpu().numpy()
                # 按类计算 IoU
                for ci in range(number_targets):
                    o = output_bin[:, ci].astype(bool)
                    t = target[:, ci].astype(bool)
                    inter = (o & t).sum()
                    union = (o | t).sum()
                    if union > 0:
                        iou_sums[ci] += float(inter) / float(union)
                        iou_cnts[ci] += 1.0
        per_class_iou = [ (iou_sums[i] / iou_cnts[i]) if iou_cnts[i] > 0 else float('nan') for i in range(number_targets) ]
    except Exception as e:
        logger.warning(f"[IoU] 计算失败，错误：{e}；将回退到由 Dice 推导的 IoU 计算。")
        for d in per_class_dice:
            denom = 2.0 - d
            per_class_iou.append((d / denom) if denom != 0.0 else 0.0)

    mean_iou = _mean_valid(per_class_iou)
    # 打印逐类 IoU
    per_cls_iou_str = ', '.join([f"{(label_names[i] if i < len(label_names) else f'Class_{i}')}: {per_class_iou[i]:.4f}" for i in range(number_targets)])
    logger.info(f"Per-class IoU: {per_cls_iou_str}")

    # 打印总体结果到控制台与日志（附加 mean_IoU）
    summary = (
        f"test: mean_dice={mean_dice:.4f}, mean_hd95={mean_hd95:.4f}, "
        f"mean_recall={mean_recall:.4f}, mean_precision={mean_precision:.4f}, mean_IoU={mean_iou:.4f}"
    )
    print(summary)
    logger.info(summary)
