# Plant Disease Prediction

A reproducible pipeline for detecting and classifying plant leaf diseases from images using deep learning. This repository contains code, model training scripts, preprocessing, evaluation utilities, and inference helpers so you can train models on your own dataset or use pretrained checkpoints for inference.

> NOTE: This README is intentionally detailed — it explains the repository structure, how to prepare data, how to train and evaluate models, and tips for improving results. If anything below doesn't match the code in this repository, please update the README or open an issue.

Table of contents
- Project overview
- Features
- Repository structure
- Dataset and preprocessing
- Installation
- Quick start
  - Running inference on a single image
  - Training a model
- Training details and recommended hyperparameters
- Model architecture and approach
- Evaluation and metrics
- Tips to improve performance
- Experiments and reproducibility
- Contributing
- License
- Contact and acknowledgements
- Troubleshooting & FAQ

---

Project overview
----------------
Plant Disease Prediction is a deep learning project that classifies plant leaf images into healthy vs. various disease categories. The pipeline includes:
- Data loading and augmentation
- Train / validation / test splitting
- Model definitions (transfer learning + fine-tuning)
- Training loop with checkpointing and logging
- Evaluation scripts producing accuracy, precision, recall, F1, and confusion matrices
- Inference utilities to make predictions on single images or folders

This project is intended for researchers, students, and practitioners building image-based plant-disease classifiers.

Features
--------
- Trainable end-to-end pipeline using modern convnets (transfer learning)
- Configurable preprocessing and augmentations
- Checkpointing and resume training support
- Evaluation metrics and visualization (confusion matrix, class-wise metrics)
- Inference script for single-image or batch prediction
- Guidelines for dataset collection, augmentation, and deployment

Repository structure
--------------------
A typical layout for the project (adjust if files differ in your repository):

- data/
  - raw/                      # Raw dataset (not tracked)
  - processed/                # Processed images / TFRecords (optional)
- notebooks/                  # EDA and experiment notebooks
- src/                        # Source code (train, eval, predict, utils)
  - data.py                   # Dataset loading and transforms
  - model.py                  # Model definitions (ResNet/EfficientNet wrappers)
  - train.py                  # Training loop and scheduler
  - eval.py                   # Evaluation utilities and metrics
  - predict.py                # Inference script for single images / batches
  - utils.py                  # Common helper functions (logging, checkpoints)
- models/                     # Checkpoints and saved models (gitignored)
- logs/                       # Training logs (TensorBoard / CSV)
- requirements.txt            # Python dependencies
- Dockerfile (optional)       # Containerization
- README.md                   # This file

If your repository differs, adapt the instructions below to match the actual file names and argument names.

Dataset and preprocessing
-------------------------
Common public datasets for plant disease classification include:
- PlantVillage (popular benchmark)
- Your own collected dataset (images captured in-field)

Dataset format
- A recommended structure:
  data/
    train/
      class_a/
        img001.jpg
        ...
      class_b/
        img002.jpg
        ...
    val/
      class_a/
      class_b/
    test/
      class_a/
      class_b/

- Alternatively, provide CSV files listing image paths and labels.

Important preprocessing steps
- Resize images to a fixed size (224×224 or 256×256 is common for transfer learning)
- Normalize using ImageNet mean/std if using ImageNet-pretrained backbones
- Apply data augmentation during training:
  - Random cropping / scaling
  - Horizontal/vertical flips
  - Color jitter (brightness/contrast)
  - Random rotations
  - Random erasing / cutout (careful for small leaf images)
- Ensure validation/test sets are deterministic and don't use random augmentation

Installation
------------
1) Clone the repo
```bash
git clone https://github.com/Vamshikrishan/plant_disease_prediction.git
cd plant_disease_prediction
```

2) Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If there's a GPU available, make sure CUDA and the correct PyTorch/TensorFlow builds are installed. The requirements file will specify the framework (PyTorch or TensorFlow).

Optional: Docker
```bash
docker build -t plant-disease-pred .
docker run --gpus all -it --rm -v $(pwd)/data:/app/data plant-disease-pred
```

Quick start
-----------

1) Running inference on a single image

Example (predict.py):
```bash
python src/predict.py \
  --image path/to/image.jpg \
  --checkpoint models/best_checkpoint.pth \
  --arch resnet50 \
  --input-size 224
```

This should print the top prediction and class probabilities. If available, some predict scripts also save a visualization (image + predicted label + probability).

2) Training a model

Example (train.py) — adapt flags to your code:
```bash
python src/train.py \
  --data-dir data \
  --train-subdir train \
  --val-subdir val \
  --arch resnet50 \
  --pretrained \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --output-dir models \
  --num-workers 8
```

Common options to include:
- --arch: network backbone (resnet50, efficientnet_b0, mobilenet_v2, etc.)
- --pretrained: use ImageNet weights
- --scheduler: learning rate scheduler (cosine, step, reduce-on-plateau)
- --resume: path to resume checkpoint
- --amp: use mixed precision (PyTorch AMP) for faster training and lower memory

Training details and recommended hyperparameters
-----------------------------------------------
A sensible starting configuration (for transfer learning, ResNet50 backbone):
- Input size: 224
- Batch size: 16–64 (depending on GPU memory)
- Optimizer: AdamW or SGD with momentum (0.9)
- Learning rate: 1e-3 (AdamW) or 1e-2 (SGD) with warmup
- Weight decay: 1e-4
- Scheduler: CosineAnnealingLR or ReduceLROnPlateau
- Epochs: 30–100 depending on dataset size
- Early stopping: monitor validation loss / metric

If dataset is small (< 5k images), freeze the backbone initially and train the head only for a few epochs, then unfreeze and fine-tune with a lower LR.

Model architecture and approach
-------------------------------
The repository uses transfer learning with popular CNN backbones (e.g., ResNet, EfficientNet). The usual approach:
1. Replace the final classification head to match number of classes
2. Initialize backbone with ImageNet pretrained weights
3. Optionally freeze some backbone layers for initial epochs
4. Train with strong augmentations and class-balanced sampling if classes are imbalanced

If you prefer a different approach (from-scratch training, transformers, or multi-task learning), the code is modular enough to swap model definitions.

Evaluation and metrics
----------------------
Important metrics for multi-class disease classification:
- Accuracy (overall)
- Precision, Recall, F1 (per-class and macro-averaged)
- Confusion matrix (visualize to understand class confusions)
- ROC-AUC (for binary or one-vs-rest approaches)

Example evaluation command:
```bash
python src/eval.py \
  --data-dir data \
  --test-subdir test \
  --checkpoint models/best_checkpoint.pth \
  --arch resnet50 \
  --batch-size 32 \
  --output-dir logs/eval_results
```

Evaluation outputs should include:
- CSV with predictions and ground truth
- Per-class metrics JSON / CSV
- Confusion matrix plot (PNG)
- Classification report (sklearn style)

Tips to improve performance
---------------------------
- More and better labeled data: collect varied images across seasons, lighting, and plant stages.
- Stronger data augmentation (but keep domain realism)
- Use class rebalancing or weighted loss for skewed datasets
- Use image-level metadata (plant species, capture conditions) if available in a multi-input model
- Try different backbones (EfficientNet, ResNeXt) or lightweight models for deployment (MobileNet)
- Ensemble models trained with different seeds / augmentations
- Fine-tune for more epochs with lower LR and cyclic schedules
- Use test-time augmentations (TTA) at inference for improved robustness
- Analyze confusion matrix and adjust class hierarchy or merge similar classes when appropriate
- Use image segmentation to focus model on leaf region (remove background noise)

Experiments and reproducibility
-------------------------------
- Log hyperparameters, random seed, and environment (framework and CUDA/cuDNN versions).
- Save model weights, training logs, and a snapshot of the code used to produce results.
- Use deterministic settings where possible (but note that some GPU ops can still be nondeterministic).
- Save the conda/pip environment:
  ```bash
  pip freeze > requirements_frozen.txt
  ```
- Optionally use Docker to ensure identical runtime.

Contributing
------------
Contributions are welcome! Suggested ways to contribute:
- Open an issue if you find a bug or want to request a feature
- Improve documentation and README
- Add new pre-processing, augmentation techniques, or model architectures
- Add unit tests or CI workflows
- Share pre-trained model checkpoints (consider Git LFS or cloud storage)

A simple contribution workflow:
1. Fork the repo
2. Create a feature branch
3. Make changes and add tests if applicable
4. Open a pull request describing what changed and why

License
-------
This repository is distributed under the MIT License. See LICENSE file for details.

(If you prefer a different license, replace this section and update the LICENSE file accordingly.)

Contact and acknowledgements
----------------------------
Maintainer: Vamshikrishan — https://github.com/Vamshikrishan

Acknowledgements:
- Public datasets and communities that provide plant disease images (e.g., PlantVillage)
- Open-source libraries: PyTorch / TensorFlow, torchvision / timm, Albumentations, scikit-learn, matplotlib

Troubleshooting & FAQ
---------------------
Q: Training is not improving — validation loss plateaus or increases
A:
- Check for data leakage between train and validation sets.
- Reduce learning rate or use a scheduler.
- Ensure augmentations are appropriate and not excessively corrupting images.
- Try freezing backbone and training head first.

Q: Model overfits quickly
A:
- Increase augmentation strength.
- Use weight decay and dropout.
- Acquire more data or use transfer learning with frozen layers.

Q: Slow training or out-of-memory (OOM)
A:
- Reduce batch size or image size.
- Use mixed precision (AMP).
- Use gradient accumulation to emulate larger batch sizes.
- Switch to a lighter model (MobileNet, EfficientNet-lite).

Q: Inference is too slow for deployment
A:
- Convert model to ONNX / TensorRT or use TFLite for mobile.
- Prune or quantize model to reduce size and latency.
- Use a smaller backbone for edge deployment.

Final notes
-----------
This README aims to be a complete starting guide. If you want, I can:
- Generate a shorter README (one-page)
- Create example scripts (train.py, predict.py) or a notebook for demo
- Add CI configuration or a GitHub Action to run tests/formatting

Would you like me to generate any runnable example scripts (train/predict) tailored to PyTorch or TensorFlow, or draft a short contributing guide/PR template next?
