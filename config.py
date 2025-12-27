import os
import logging

# Reproducibility settings
REPRODUCIBILITY = {
    'seed': 122,
}

# Data/task configuration
DATA = {
    'target_shape': (96, 96, 96),
    'cache_data': False,
    'num_modality': 1,
    'num_classes': 5,
    'label_names': ["LR", "MD", "MX", "RR", "SK"],
    'class_order': ["LR", "MD", "MX", "RR", "SK"],
}

# Paths
PATHS = {
    # Use a repo-relative default instead of a personal absolute path. Users should set this as needed.
    'data_dir': '/home/wws/dataset/Volume',
    'logdir': './checkpoints/',
    'model_save_dir': None,  # set below
    'pred_output_dir': 'predictions',
    # Avoid committing local checkpoint paths; default to None so scripts require explicit model path when testing.
    'test_model_path': None,
}
PATHS['model_save_dir'] = os.path.join(PATHS['logdir'], 'model')

# Runtime
RUNTIME = {
    'env': 'pytorch',
    'device': 'cuda:0',
    'num_gpus': 1,
}

# Training/validation
TRAINING = {
    'max_epoch': 40,
    'batch_size': 2,
    'val_every': 2,
    'dataloader_num_workers': 0,
    'pin_memory': False,
    # Compute class weights on startup; set False to skip for faster loading
    'compute_class_weights': False,
    # Number of samples used if computing weights (1 = single file)
    'class_weight_samples': 1,
}

# Optimizer/LR
OPTIMIZER = {
    'lr_init': 1e-4,
    'lr_min': 1e-6,
    'weight_decay': 1e-3,
    'use_cosine_scheduler': True,
    'cosine_step_level': True,  # True: step-level; False: epoch-level
}

# Dataset split
SPLIT = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'random_seed': 42,
}

# Preprocessing (for NIfTI inputs)
PREPROCESSING = {
    'target_shape': DATA['target_shape'],
    'intensity_norm': True,
    'clip_percentiles': (1.0, 99.0),
    'clip_range': None,
    'skull_binary': False,
    'allow_resize': True,
}

# Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': (-15, 15),
    'translation_range': (-5, 5),
    'scaling_range': (0.9, 1.1),
    'noise_variance': 0.01,
    'elastic_alpha': 50.0,
    'elastic_sigma': 5.0,
    'flip_probability': 0.5,
    'augmentation_probability': 0.7,
}

# Sliding-window inference
INFERENCE = {
    'sw_roi_size': DATA['target_shape'],
    'sw_batch_size': 1,
    'sw_overlap': 0.25,
    'prob_threshold': 0.5,
}

# Diffusion settings
DIFFUSION = {
    'diffusion_steps': 1000,
    'sample_steps': 50,
    'beta_schedule': 'linear',
    # DDIM randomness; 0.0 = deterministic, >0 enables stochastic sampling
    'eta': 0.0,
}

# Logging
LOGGING = {
    'log_file': 'training.log',
    'log_level': logging.INFO,
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# Distributed config
DISTRIBUTED = {
    'master_ip': 'localhost',
    'master_port': 17751,
}

# Ablation toggles
ABLATON = {
    'USE_ENCODER': True,              # Use BasicUNetEncoder (with TinySwinBlock) to provide embeddings
    'USE_LEARNABLE_FUSION': True,     # Use LearnableFusion per scale; otherwise direct addition (with spatial align)
    # Fusion ablation: number of repeated predictions per DDIM step when estimating step-wise uncertainty.
    # Higher values stabilize uncertainty estimates but increase inference time proportionally.
    'UNCERTAINTY_PREDICTIONS_PER_STEP': 2,
    # Enable fusion based on step-wise uncertainty (False keeps "last-step only" output)
    'USE_DIFFUSION_FUSION': False,
    # Sigmoid scaling for fusion weights: w_i = exp(sigmoid(i/scale) * (1 - u_i))
    'FUSION_SIGMOID_SCALE': 10.0,
    # Normalize w_i across steps to keep probability stable
    'NORMALIZE_FUSION_WEIGHTS': True,
}
