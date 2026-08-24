<div align="center">

# 🌿 `TriVeda-AI`

### _Deep Vision Framework for Automated Ayurvedic Botanicals & Triphala Authentication_

[![Python](https://img.shields.io/badge/Python-3.10-10B981?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%20%2F%20Metal-F59E0B?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Custom%20CNN-EF4444?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-3B82F6?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <a href="#-abstract">Abstract</a> •
  <a href="#-botanical-classes">Botanicals</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-benchmark-matrix">Benchmarks</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-system-design">UML Design</a>
</p>

---

</div>

## 📌 Abstract

**Triphala** (_"Three Fruits"_) is a cornerstone formulation in traditional Ayurvedic pharmacology. The commercial supply chain faces serious bottlenecks: raw dried components arrive crushed and fragmented, making human visual inspection slow, subjective, and highly prone to adulteration.

**TriVeda-AI** deploys a benchmarked computer vision engine comparing an engineered **Deep Custom CNN (from scratch)** against top transfer-learning backbones (**ResNet50**, **VGG16**, **VGG19**) to achieve industrial-grade botanical authentication.

---

## 🍃 Target Botanical Species

<div align="center">

|                                      Specimen                                      | Sanskrit Name | Botanical Nomenclature |     Medicinal Part     |
| :--------------------------------------------------------------------------------: | :-----------: | :--------------------: | :--------------------: |
|  <img src="https://img.shields.io/badge/Class_01-Amla-10B981?style=flat-square"/>  |   _Amalaki_   | _Phyllanthus emblica_  | Dried Pericarp / Fruit |
| <img src="https://img.shields.io/badge/Class_02-Herda-F59E0B?style=flat-square"/>  |  _Haritaki_   |  _Terminalia chebula_  | Dried Seed / Pericarp  |
| <img src="https://img.shields.io/badge/Class_03-Bherda-3B82F6?style=flat-square"/> |  _Bibhitaki_  | _Terminalia bellirica_ |    Dried Ripe Fruit    |

</div>

---

## 📊 Dataset Specifications

The dataset comprises high-resolution images of authentic botanical specimens balanced across three primary classes.

````text
Dataset Statistics:
├── Total Sample Size: ~6,900 Images (Balanced)
├── Class 1: amla (Phyllanthus emblica)    -> ~2,300 images
├── Class 2: herda (Terminalia chebula)    -> ~2,300 images
├── Class 3: bherda (Terminalia bellirica) -> ~2,300 images
├── Resolution: 224 x 224 x 3 (RGB)
└── Partitioning: 70% Train (~4,830 imgs) | 15% Validation (~1,035 imgs) | 15% Test (~1,035 imgs)


🧠 Engineered Architecture (Model1.py)
Our custom CNN is built from the ground up for granular herbal micro-textures rather than general natural objects:

Input: (224 x 224 x 3)
  │
  ├── [Block 1] 2x Conv2D (64 filters, 3x3)   + BatchNorm + MaxPool2D + Dropout(0.25)
  ├── [Block 2] 2x Conv2D (128 filters, 3x3)  + BatchNorm + MaxPool2D + Dropout(0.25)
  ├── [Block 3] 2x Conv2D (256 filters, 3x3)  + BatchNorm + MaxPool2D + Dropout(0.30)
  ├── [Block 4] 2x Conv2D (512 filters, 3x3)  + BatchNorm + MaxPool2D + Dropout(0.30)
  │
  ├── GlobalAveragePooling2D (Eliminates spatial overfitting)
  ├── Dense (512, ReLU) + L2 Regularization (1e-4) + BatchNorm + Dropout(0.50)
  ├── Dense (256, ReLU) + L2 Regularization (1e-4) + BatchNorm + Dropout(0.50)
  └── Dense (3, Softmax Output)


  📁 Complete Project Structure

  TriVeda-AI/
├── Resized/                        # Raw Dataset (amla, bherda, herda)
│   ├── amla/                       # ~2,300 images
│   ├── bherda/                     # ~2,300 images
│   └── herda/                      # ~2,300 images
│
├── Model1.py                       # Enhanced Custom CNN Model Script
├── Resnet.py                       # ResNet50 Transfer Learning Script
├── VGG16.py                        # VGG16 Transfer Learning Script
├── VGG19 (1).py                    # VGG19 Transfer Learning Script
├── results_comparison.py           # Evaluation Suite (Generates charts & tables)
├── requirements.txt                # Python environment specifications
├── .gitignore                      # Git exclusion rules
│
├── activity.py                     # UML Activity Diagram generator
├── class.py                        # UML Class Diagram generator
├── dataflow.py                     # DFD Level-0 & Level-1 Diagram generator
├── er.py                           # Entity-Relationship (ER) Diagram generator
├── grant.py                        # Project Gantt Chart visualization
├── sequence.py                     # UML Sequence Diagram generator
├── usecase.py                      # UML Use Case Diagram generator
└── README.md                       # Comprehensive project documentation

⚡ Quickstart
1. Clone & Set Environment
git clone https://github.com/Nikky31/TriVeda-AI.git
cd TriVeda-AI

# Setup Conda Python 3.10
conda create -n triphalaenv python=3.10 -y
conda activate triphalaenv

# Install Dependencies
pip install -r requirements.txt


<div align="center">
# 🌿 `TriVeda-AI`
### *Deep Vision Framework for Automated Ayurvedic Botanicals & Triphala Authentication*
[![Python](https://img.shields.io/badge/Python-3.10-10B981?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%20%2F%20Metal-F59E0B?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Custom%20CNN-EF4444?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-3B82F6?style=for-the-badge)](https://opensource.org/licenses/MIT)
<p align="center">
  <a href="#-abstract">Abstract</a> •
  <a href="#-botanical-classes">Botanicals</a> •
  <a href="#-dataset-specifications">Dataset</a> •
  <a href="#-engineered-architecture-model1py">Architecture</a> •
  <a href="#-benchmark-matrix">Benchmarks</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-system-design">UML Design</a> •
  <a href="#-placement--defense-faq">Interview Q&A</a>
</p>
---
</div>
## 📌 Abstract
**Triphala** (*"Three Fruits"*) is a cornerstone formulation in traditional Ayurvedic pharmacology. The commercial supply chain faces serious bottlenecks: raw dried components arrive crushed and fragmented, making human visual inspection slow, subjective, and highly prone to adulteration.
**TriVeda-AI** deploys a benchmarked computer vision engine comparing an engineered **Deep Custom CNN (from scratch)** against top transfer-learning backbones (**ResNet50**, **VGG16**, **VGG19**) to achieve industrial-grade botanical authentication.


 [ Raw Dried Specimen ]
              │
              ▼
┌─────────────────────────────┐ │ Real-time Vision Transforms │ │ (Affine, Zoom, Color Shift) │ └──────────────┬──────────────┘ │ ▼ ┌─────────────────────────────┐ │ Paired Residual Conv Blocks │ ──► [ Global Average Pooling ] └──────────────┬──────────────┘ │ │ ▼ └─────────────────────► [ Multi-Class Softmax ] │ ▼ Amla │ Herda │ Bherda



---
## 🍃 Botanical Classes
<div align="center">
| Specimen | Sanskrit Name | Botanical Nomenclature | Medicinal Part |
| :---: | :---: | :---: | :---: |
| <img src="https://img.shields.io/badge/Class_01-Amla-10B981?style=flat-square"/> | *Amalaki* | *Phyllanthus emblica* | Dried Pericarp / Fruit |
| <img src="https://img.shields.io/badge/Class_02-Herda-F59E0B?style=flat-square"/> | *Haritaki* | *Terminalia chebula* | Dried Seed / Pericarp |
| <img src="https://img.shields.io/badge/Class_03-Bherda-3B82F6?style=flat-square"/> | *Bibhitaki* | *Terminalia bellirica* | Dried Ripe Fruit |
</div>
---
## 📊 Dataset Specifications
The dataset comprises high-resolution images of authentic botanical specimens balanced across three primary classes.
```text
Dataset Statistics:
├── Total Sample Size: ~6,900 Images (Balanced)
├── Class 1: amla (Phyllanthus emblica)    -> ~2,300 images
├── Class 2: herda (Terminalia chebula)    -> ~2,300 images
├── Class 3: bherda (Terminalia bellirica) -> ~2,300 images
├── Resolution: 224 x 224 x 3 (RGB)
└── Partitioning: 70% Train (~4,830 imgs) | 15% Validation (~1,035 imgs) | 15% Test (~1,035 imgs)

📥 Dataset Access Link
Download Raw Images: Kaggle / Google Drive Dataset Repository (https://drive.google.com/drive/folders/1d5nzqNDPnteR5d7ri-_pXk1HxbFygRgv?usp=sharing)


🧠 Engineered Architecture (Model1.py)
Our custom CNN is built from the ground up for granular herbal micro-textures rather than general natural objects:

Input: (224 x 224 x 3)
  │
  ├── [Block 1] 2x Conv2D (64 filters, 3x3)   + BatchNorm + MaxPool2D + Dropout(0.25)
  ├── [Block 2] 2x Conv2D (128 filters, 3x3)  + BatchNorm + MaxPool2D + Dropout(0.25)
  ├── [Block 3] 2x Conv2D (256 filters, 3x3)  + BatchNorm + MaxPool2D + Dropout(0.30)
  ├── [Block 4] 2x Conv2D (512 filters, 3x3)  + BatchNorm + MaxPool2D + Dropout(0.30)
  │
  ├── GlobalAveragePooling2D (Eliminates spatial overfitting)
  ├── Dense (512, ReLU) + L2 Regularization (1e-4) + BatchNorm + Dropout(0.50)
  ├── Dense (256, ReLU) + L2 Regularization (1e-4) + BatchNorm + Dropout(0.50)
  └── Dense (3, Softmax Output)



## 📊 Benchmark Matrix

<div align="center">

| Model Architecture | Paradigm | Epochs | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Custom Enhanced CNN** | **From Scratch** | **50** | **`~94.5%`** | **`0.95`** | **`0.94`** | **`0.94`** | ![Top](https://img.shields.io/badge/SOTA-Champion-10B981?style=flat-square) |
| 🥈 **ResNet50** | Transfer Learning | 20 | `~92.8%` | `0.93` | `0.93` | `0.93` | ![Runner](https://img.shields.io/badge/Transfer-Solid-blue?style=flat-square) |
| 🥉 **VGG19** | Transfer Learning | 20 | `~91.1%` | `0.91` | `0.91` | `0.91` | ![Good](https://img.shields.io/badge/Transfer-Good-lightgrey?style=flat-square) |
| 🏅 **VGG16** | Transfer Learning | 20 | `~90.4%` | `0.91` | `0.90` | `0.90` | ![Good](https://img.shields.io/badge/Transfer-Baseline-lightgrey?style=flat-square) |

</div>

📁 Complete Project Structure
text


TriVeda-AI/
├── Resized/                        # Raw Dataset (amla, bherda, herda)
│   ├── amla/                       # ~2,300 images
│   ├── bherda/                     # ~2,300 images
│   └── herda/                      # ~2,300 images
│
├── Model1.py                       # Enhanced Custom CNN Model Script
├── Resnet.py                       # ResNet50 Transfer Learning Script
├── VGG16.py                        # VGG16 Transfer Learning Script
├── VGG19 (1).py                    # VGG19 Transfer Learning Script
├── results_comparison.py           # Evaluation Suite (Generates charts & tables)
├── requirements.txt                # Python environment specifications
├── .gitignore                      # Git exclusion rules
│
├── activity.py                     # UML Activity Diagram generator
├── class.py                        # UML Class Diagram generator
├── dataflow.py                     # DFD Level-0 & Level-1 Diagram generator
├── er.py                           # Entity-Relationship (ER) Diagram generator
├── grant.py                        # Project Gantt Chart visualization
├── sequence.py                     # UML Sequence Diagram generator
├── usecase.py                      # UML Use Case Diagram generator
└── README.md                       # Comprehensive project documentation


⚡ Quickstart
1. Clone & Set Environment

git clone https://github.com/Nikky31/TriVeda-AI.git
cd TriVeda-AI

# Setup Conda Python 3.10
conda create -n triphalaenv python=3.10 -y
conda activate triphalaenv

# Install Dependencies
pip install -r requirements.txt

2. Dataset Setup
TriVeda-AI/
└── Resized/
    ├── amla/     # ~2,300 images
    ├── bherda/   # ~2,300 images
    └── herda/    # ~2,300 images

3. Execution Pipeline
# 🎯 Train Custom SOTA CNN
python Model1.py

# 🔬 Train Transfer Learning Models
python Resnet.py
python VGG16.py
python "VGG19 (1).py"

# 📊 Generate Automated Benchmark Analytics
python results_comparison.py


📐 System Design
Generate interactive UML and software architecture diagrams:

🔍 Expand UML Diagram Commands
# Ensure Graphviz is installed: brew install graphviz
python activity.py    # Activity Lifecycle Workflow
python class.py       # OOP Class Hierarchy
python dataflow.py    # Context & Level-1 DFD
python er.py          # Entity-Relationship Architecture
python sequence.py    # Message Passing & Sequence Flow
python usecase.py     # System Boundary & Actors
python grant.py       # Project Development Gantt Chart
````
