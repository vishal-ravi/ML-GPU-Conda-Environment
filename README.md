<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" alt="Logo" width="80" height="80">
</p>

<h1 align="center">ML-GPU Conda Environment</h1>

<p align="center">
  <b>Pre-configured, GPU-accelerated Machine Learning & Deep Learning Conda environment</b><br>
  <i>TensorFlow, PyTorch, and all essential libraries in one place</i>
  <br><br>
  <a href="#installation">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#file-structure">File Structure</a> •
  <a href="#need-help">Need Help?</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OS-Linux-green?logo=linux" />
  <img src="https://img.shields.io/badge/GPU-NVIDIA-brightgreen?logo=nvidia" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 🚀 Overview

This repository provides a **ready-to-use, GPU-accelerated Conda environment** for modern machine learning and deep learning workflows. Perfect for:

- Training deep learning models
- NLP with transformers
- Computer vision
- AutoML
- Experiment tracking with MLflow & wandb

---

## 📋 Prerequisites

- **OS:** Linux (tested on Ubuntu/Pop!_OS)
- **GPU:** NVIDIA with CUDA support (e.g., RTX A4000 or better)
- **Driver:** NVIDIA Driver ≥ 535
- **CUDA Toolkit:** 12.1 or 12.8
- **Conda:** Anaconda or Miniconda

Check your GPU:
```bash
nvidia-smi
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/vishal-ravi/ML-GPU-Conda-Environment.git
cd ML-GPU-Conda-Environment
```

### 2. Create the environment
```bash
conda env create -f ml-gpu-env.yml
```

### 3. Activate and register with Jupyter (optional)
```bash
conda activate ml-gpu
python -m ipykernel install --user --name=ml-gpu --display-name "Python (ml-gpu)"
```

---

## 🛠️ Manual Installation (Alternative)

1. **Create a new environment**
```bash
conda create -n ml-gpu python=3.10 -y
conda activate ml-gpu
```
2. **Install core frameworks**
```bash
pip install tensorflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
3. **Install essential libraries**
```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost joblib keras h5py
pip install matplotlib seaborn plotly pandas-profiling
pip install transformers datasets nltk spacy textblob sentencepiece
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
pip install opencv-python opencv-contrib-python pillow imageio scikit-image
pip install mlflow wandb optuna autogluon tqdm rich psutil accelerate
pip install openpyxl xlrd python-docx tabulate
```
4. **(Optional) Set up Jupyter kernel**
```bash
pip install jupyterlab notebook ipykernel
python -m ipykernel install --user --name=ml-gpu --display-name "Python (ml-gpu)"
```

---

## 🧪 GPU Verification

**TensorFlow:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**PyTorch:**
```python
import torch
print(torch.cuda.get_device_name(0))
```

If TensorFlow doesn’t detect GPU:
```bash
export TF_ENABLE_ONEDNN_OPTS=0
pip cache purge
pip install tensorflow --upgrade
```

---

## ✨ Features

- **Deep Learning:** TensorFlow, PyTorch, Keras
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **NLP:** Transformers, spaCy, NLTK, TextBlob
- **Computer Vision:** OpenCV, Pillow, scikit-image
- **Visualization:** Matplotlib, Seaborn, Plotly, Pandas-Profiling
- **MLOps & Utilities:** Jupyter, MLflow, Weights & Biases, Optuna, AutoGluon, tqdm, rich, psutil, accelerate

---

## 🔍 Use Cases

- Deep learning model training with GPU acceleration
- NLP using Transformers and spaCy
- Real-time computer vision projects
- AutoML and hyperparameter optimization
- Experiment tracking and collaboration with MLflow and wandb

---

## 📁 File Structure

```
ML-GPU-Conda-Environment/
├── ml-gpu-env.yml        # Pre-built environment file
└── README.md             # Full setup instructions
```

---

## 📌 Notes

- Tested on Pop!_OS and Ubuntu 22.04 LTS
- Windows users may need to modify steps or use WSL2
- GPU drivers and CUDA should be properly installed before using this setup

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙋‍♂️ Need Help?

Open an issue or connect with me on [LinkedIn](https://www.linkedin.com/in/vishal-ravi07/) for help or collaboration!


