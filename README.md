# 💻 ML-GPU Conda Environment

This repository provides a pre-configured, GPU-accelerated **Machine Learning and Deep Learning Conda environment** using **TensorFlow**, **PyTorch**, and other essential libraries.
It is designed for **Linux systems with NVIDIA GPUs** and includes everything needed for NLP, computer vision, classical ML, AutoML, and experiment tracking.

---

## 📌 Prerequisites

Before using this environment, ensure the following requirements are met:

### ✅ System Requirements:

* Operating System: Linux (tested on Ubuntu/Pop!\_OS)
* GPU: NVIDIA GPU with CUDA support (e.g., RTX A4000, RTX 30xx, etc.)
* Driver: NVIDIA Driver version ≥ 535
* CUDA: CUDA 12.1 or 12.8 (auto-handled by frameworks)
* Conda: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed

### 🔧 To check GPU:

```bash
nvidia-smi
```

---

## ⚙️ Setup Instructions

1. **Clone the repository** (or download the `.yml` file)

```bash
git clone https://github.com/your-username/ml-gpu-env.git
cd ml-gpu-env
```

2. **Create the Conda environment**

```bash
conda env create -f ml-gpu-env.yml
```

3. **Activate the environment**

```bash
conda activate ml-gpu
```

4. **(Optional) Add to Jupyter kernel**

```bash
python -m ipykernel install --user --name=ml-gpu --display-name "Python (ml-gpu)"
```

---

## 🧪 GPU Verification

### TensorFlow:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### PyTorch:

```python
import torch
print(torch.cuda.get_device_name(0))
```

If TensorFlow fails to detect GPU, try:

```bash
export TF_ENABLE_ONEDNN_OPTS=0
pip cache purge
pip install tensorflow --upgrade
```

---

## 📦 Included Libraries

### Deep Learning:

* TensorFlow (GPU-enabled)
* PyTorch, torchvision, torchaudio
* Keras

### Classical Machine Learning:

* scikit-learn, XGBoost, LightGBM, CatBoost

### Natural Language Processing:

* HuggingFace Transformers, spaCy, NLTK, TextBlob, SentencePiece

### Computer Vision:

* OpenCV, Pillow, scikit-image, imageio

### Visualization:

* Matplotlib, Seaborn, Plotly, Pandas-Profiling

### Productivity and MLOps:

* Jupyter, IPython, MLflow, Weights & Biases (wandb)
* AutoGluon, Optuna
* tqdm, rich, joblib, psutil, accelerate

---

## 🚀 Use Cases

* Train deep learning models with GPU acceleration
* Perform NLP tasks using state-of-the-art transformer models
* Execute real-time computer vision projects
* Automate model tuning and selection (AutoML)
* Track experiments and collaborate using MLflow and wandb

---

## 📁 File Structure

```
ml-gpu-env/
├── ml-gpu-env.yml        # The environment file
└── README.md             # This guide
```

---

## 📌 Notes

* This environment is built for **Linux only**. Windows/Mac users may require adjustments.
* GPU acceleration depends on proper driver installation. If `nvidia-smi` fails, install/reinstall your NVIDIA drivers.

---

## 🙋‍♂️ Need Help?

Feel free to open an issue or connect with me on [LinkedIn](https://www.linkedin.com/in/your-profile) if you face any problems or want to collaborate!

---

