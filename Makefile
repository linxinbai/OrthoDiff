SHELL := /bin/bash

.PHONY: lint format smoke help

help:
	@echo "Targets: lint format smoke"

lint:
	@python -m pip install --quiet flake8
	@flake8 --max-line-length=120 --ignore=E203,W503 . || true

format:
	@python -m pip install --quiet black isort
	@isort .
	@black -l 120 .

smoke:
	@python - << 'PY'
import importlib, sys
mods = [
    'config', 'train', 'test', 'predict', 'trainer',
    'data_processing.augmentation', 'data_processing.dataset', 'data_processing.preprocessor',
    'utils.files_helper', 'utils.metric', 'utils.lr_scheduler',
    'models.unet', 'models.basic_unet', 'models.diffusion', 'models.losses',
]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"FAIL: {m} -> {e}")
        sys.exit(1)
print("Smoke tests passed")
PY
