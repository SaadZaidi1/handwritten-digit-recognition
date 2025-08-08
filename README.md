# Handwritten Digit Recognition

A Python project for recognizing handwritten digits (0–9), typically using the MNIST dataset. This repository provides code and guidance to train a model, evaluate its performance, and run inference on custom images.

If the script and folder names in your repo differ from the examples below (e.g., `train.py`, `predict.py`, `models/`, `data/`), adjust the commands accordingly.

## Features

- End-to-end workflow: data loading, training, evaluation, and inference
- Works with the canonical MNIST dataset (60k train / 10k test)
- Reproducible training via seeded runs
- Supports running via Python scripts and/or Jupyter notebooks
- Clear separation of concerns for datasets, models, and utilities (if present)

## Tech Stack

- Python (100%)
- Common libraries (depending on your implementation choices), for example:
  - numpy, pandas, matplotlib
  - scikit-learn for baselines (e.g., SVM, Logistic Regression)
  - Deep learning (choose one):
    - TensorFlow/Keras, or
    - PyTorch + torchvision
  - pillow / opencv-python for image preprocessing (optional)

## Dataset

- MNIST: http://yann.lecun.com/exdb/mnist/
- Many frameworks provide built-in loaders:
  - TensorFlow: `tf.keras.datasets.mnist.load_data()`
  - PyTorch: `torchvision.datasets.MNIST(download=True, ...)`

Most training scripts will download MNIST automatically if not present.

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/SaadZaidi1/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2) Set up a Python environment

```bash
# Using venv (Python 3.9+ recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies

If your repo includes a requirements file:

```bash
pip install -U pip
pip install -r requirements.txt
```

If not, install a minimal set (adjust as needed):

```bash
# Choose one deep learning framework (optional):
# TensorFlow:
pip install numpy matplotlib scikit-learn tensorflow

# OR PyTorch (CPU example, see https://pytorch.org/get-started/locally/ for your platform):
pip install numpy matplotlib scikit-learn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Optional for image handling:
pip install pillow opencv-python
```

## Usage

Below are typical commands. Replace script and path names to match your repository.

### 1) Train

```bash
# Example (TensorFlow/Keras or a generic script):
python train.py \
  --epochs 10 \
  --batch-size 128 \
  --lr 1e-3 \
  --output-dir checkpoints
```

Common flags:
- epochs: number of training epochs
- batch-size: mini-batch size
- lr: learning rate
- output-dir: where to save models, logs

### 2) Evaluate

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt  # or .h5 depending on your framework
```

Outputs:
- Overall test accuracy
- Optional per-class metrics and confusion matrix

### 3) Inference on custom images

```bash
python predict.py \
  --image path/to/your_digit.png \
  --checkpoint checkpoints/best_model.pt
```

Tips:
- Input images should be single digits.
- Convert to grayscale and center the digit if needed.
- Resize to 28×28 for MNIST-trained models.

## Project Structure (example)

This is an example layout. Your actual structure may differ.

```
handwritten-digit-recognition/
├─ data/                 # auto-downloaded MNIST or custom data
├─ notebooks/            # Jupyter notebooks (EDA, training, experiments)
├─ models/               # model definitions / saved weights
├─ utils/                # helper scripts (metrics, transforms, viz)
├─ train.py              # training entry point
├─ evaluate.py           # evaluation script
├─ predict.py            # inference script
├─ requirements.txt      # Python dependencies
└─ README.md
```

## Baselines

- Logistic Regression / SVM (scikit-learn) on raw pixels or HOG features
- Simple CNN (2–3 conv layers + dense layers)
- Expected accuracy:
  - Classic CNN on MNIST: ~99% test accuracy
  - Simple scikit-learn baselines: ~92–98% depending on features

## Reproducibility

- Set seeds for numpy and your chosen framework.
- Fix dataloader shuffling where applicable.
- Example:

```python
import os, random, numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# TensorFlow:
# import tensorflow as tf
# tf.random.set_seed(seed)

# PyTorch:
# import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
```

## Troubleshooting

- Import errors: verify you are in the virtual environment and dependencies are installed.
- GPU vs CPU: ensure correct framework build is installed. For PyTorch, follow the official “Get Started” page. For TensorFlow, confirm GPU drivers/CUDA.
- Shape mismatches: ensure preprocessing matches model input (28×28, grayscale, normalization).

## Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Add examples, notebooks, and documentation

Please follow standard Python formatting and include concise descriptions in PRs.

## License

Add a license to clarify usage (e.g., MIT, Apache-2.0). If a `LICENSE` file exists, reference it here.

## Acknowledgments

- MNIST dataset by Yann LeCun and collaborators
- TensorFlow/Keras and PyTorch communities
- scikit-learn for classical ML baselines

## Citation

If this project is useful in your research or learning, consider citing:

```
LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278–2324.
```

---

Maintainer: @SaadZaidi1# handwritten-digit-recognition
