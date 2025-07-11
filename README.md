# 💻 ML-GPU Conda Environment

This repository provides a **pre-configured, GPU-accelerated Machine Learning & Deep Learning Conda environment** using TensorFlow, PyTorch, and other essential libraries.

Designed for Linux systems with NVIDIA GPUs, this environment is ideal for:

* Training deep learning models
* NLP with transformers
* Computer vision
* AutoML
* Experiment tracking with MLflow & wandb

---

## 📌 Prerequisites

Before setting up, ensure the following:

### ✅ System Requirements

* OS: Linux (tested on Ubuntu/Pop!\_OS)
* NVIDIA GPU with CUDA support (e.g., RTX A4000 or better)
* NVIDIA Driver Version ≥ 535
* CUDA Toolkit: CUDA 12.1 or 12.8 (TensorFlow and PyTorch will auto-detect)
* Conda (Anaconda or Miniconda)

### 🔍 Check GPU

```bash
nvidia-smi
```

---

## 🛠️ Installation Options

### Option 1: **Recommended – Use the Conda YAML File**

1. **Clone the repository**

```bash
git clone https://github.com/your-username/ml-gpu-env.git
cd ml-gpu-env
```

2. **Create the environment from file**

```bash
conda env create -f ml-gpu-env.yml
```

3. **Activate it**

```bash
conda activate ml-gpu
```

4. *(Optional)* Register with Jupyter

```bash
python -m ipykernel install --user --name=ml-gpu --display-name "Python (ml-gpu)"
```

---

### Option 2: **Manual Installation (Step-by-Step)**

If you prefer to install manually:

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

3. **Install essential ML/DL libraries**

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost joblib keras h5py
```

4. **Install visualization tools**

```bash
pip install matplotlib seaborn plotly pandas-profiling
```

5. **Install NLP tools**

```bash
pip install transformers datasets nltk spacy textblob sentencepiece
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

6. **Install computer vision libraries**

```bash
pip install opencv-python opencv-contrib-python pillow imageio scikit-image
```

7. **Install MLOps and utility tools**

```bash
pip install mlflow wandb optuna autogluon tqdm rich psutil accelerate
pip install openpyxl xlrd python-docx tabulate
```

8. *(Optional)* Set up Jupyter kernel

```bash
pip install jupyterlab notebook ipykernel
python -m ipykernel install --user --name=ml-gpu --display-name "Python (ml-gpu)"
```

---

## 🧪 GPU Verification

### TensorFlow

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### PyTorch

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

## 📦 Included Libraries

**Deep Learning**: TensorFlow, PyTorch, Keras

**Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost

**NLP**: Transformers, spaCy, NLTK, TextBlob

**Computer Vision**: OpenCV, Pillow, scikit-image

**Visualization**: Matplotlib, Seaborn, Plotly, Pandas-Profiling

**MLOps & Utilities**: Jupyter, MLflow, Weights & Biases, Optuna, AutoGluon, tqdm, rich, psutil, accelerate

---

## 🔍 Use Cases

* Deep learning model training with GPU acceleration
* NLP using Transformers and spaCy
* Real-time computer vision projects
* AutoML and hyperparameter optimization
* Experiment tracking and collaboration with MLflow and wandb

---

## 📁 File Structure

```
ml-gpu-env/
├── ml-gpu-env.yml        # Pre-built environment file
└── README.md             # Full setup instructions
```

---

## 📌 Notes

* Tested on Pop!\_OS and Ubuntu 22.04 LTS
* Windows users may need to modify steps or use WSL2
* GPU drivers and CUDA should be properly installed before using this setup

---

## 🙋‍♂️ Need Help?

Open an issue or connect with me on [LinkedIn](https://www.linkedin.com/in/your-profile) for help or collaboration!


