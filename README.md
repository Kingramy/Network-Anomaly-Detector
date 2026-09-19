# 🛡️ Hybrid Network Anomaly & Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU%20Accelerated-228B22.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, hybrid Network Intrusion Detection System (NIDS) designed to tackle both **known network attacks** and **unseen zero-day vulnerabilities**. 

By combining **supervised gradient boosting & ensemble trees (XGBoost / Random Forest)** with an **unsupervised deep tabular autoencoder (PyTorch)** via **parallel late-fusion (asymmetric veto logic)**, this system achieves superior detection rates—especially on critical, low-footprint attacks (R2L, U2R) and novel zero-day exploits.

---

## 📌 Table of Contents

- [Overview & Problem Statement](#-overview--problem-statement)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Dataset & Attack Taxonomies](#-dataset--attack-taxonomies)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage & Execution](#-usage--execution)
  - [1. Standalone Classifiers](#1-standalone-classifiers)
  - [2. Hybrid Ensembles (Supervised + Autoencoder)](#2-hybrid-ensembles-supervised--autoencoder)
  - [3. Data Preprocessing & Validation](#3-data-preprocessing--validation)
- [Methodology & Technical Details](#-methodology--technical-details)
  - [Cost-Sensitive Sample Weighting](#cost-sensitive-sample-weighting)
  - [Semi-Supervised Autoencoder Anomaly Detection](#semi-supervised-autoencoder-anomaly-detection)
  - [Decision-Level Late Fusion](#decision-level-late-fusion)
- [Evaluation & Benchmarks](#-evaluation--benchmarks)
- [Dependencies](#-dependencies)
- [License](#-license)

---

## 🔍 Overview & Problem Statement

Traditional signature-based or purely supervised Intrusion Detection Systems (IDS) suffer from a critical vulnerability: **they cannot identify novel attack vectors (zero-day attacks) that were not present in the training set**. Conversely, purely unsupervised anomaly detectors often suffer from higher false positive rates when applied alone.

This project addresses this fundamental trade-off through a **two-tier hybrid architecture**:
1. **Tier 1 (Supervised Gatekeeper)**: A high-capacity classifier (**XGBoost** or **Random Forest**) trained with cost-sensitive class weights to catch known network attack classes (DoS, Probe, R2L, U2R) with high precision.
2. **Tier 2 (Unsupervised Anomaly Detector)**: A **Deep Tabular Autoencoder** trained strictly on normal, benign network traffic. When an anomalous pattern or novel zero-day exploit arrives, the network fails to reconstruct it accurately, resulting in a high Mean Squared Error (MSE).
3. **Decision-Level Late Fusion**: An asymmetric veto (`OR` logic / `np.maximum`) merges the outputs. If either model identifies a packet stream as anomalous, an intrusion alert is raised.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Raw Network Traffic / NSL-KDD] --> B[Data Loader & Preprocessor]
    B --> C[One-Hot Encoding & Feature Alignment]
    C --> D[MinMax Normalization]
    
    subgraph Tier 1: Supervised Classifier
        D --> E[XGBoost / Random Forest]
        E --> F[Predicted Probability >= 0.30]
        F --> G[Supervised Prediction: 0 or 1]
    end
    
    subgraph Tier 2: Unsupervised Deep Autoencoder
        D --> H[Deep Tabular Autoencoder<br/>(Trained ONLY on Normal Traffic)]
        H --> I[Compute Reconstruction Loss MSE]
        I --> J[Calibrated Threshold Check<br/>(95th - 98th Percentile)]
        J --> K[Autoencoder Prediction: 0 or 1]
    end
    
    G --> L[Late Fusion Layer<br/>(Asymmetric Veto / Logical OR)]
    K --> L
    L --> M{Final Intrusion Verdict}
    M -->|Alert| N[🚨 Anomaly Detected]
    M -->|Pass| O[✅ Normal Traffic]
```

---

## ✨ Key Features

- **Parallel Late Fusion (Asymmetric Veto)**: Ensures zero-day attacks missed by supervised models are captured by the reconstruction anomaly detector.
- **Tackling Class Imbalance**: Incorporates custom sample-weighting penalties for dangerous but rare attacks:
  - **R2L (Remote-to-Local)**: 8.0x weight boost
  - **U2R (User-to-Root privilege escalation)**: 15.0x weight boost
- **GPU Acceleration**: Utilizes PyTorch CUDA device acceleration and native GPU histogram tree methods (`tree_method='hist'`, `device='cuda'`) in XGBoost.
- **Robust Preprocessing Pipeline**:
  - Automatically loads and parses NSL-KDD (`KDDTrain+`, `KDDTest+`, `.arff`, `.csv`, `.txt`).
  - Seamlessly handles categorical feature alignment between train and test distributions.
  - Built-in synthetic fallback generator ensuring reproducible pipeline testing even if datasets are missing.
- **Novel Attack Analysis**: Evaluates performance specifically against the **17 novel attack types** present exclusively in `KDDTest+`.

---

## 📊 Dataset & Attack Taxonomies

The project evaluates on the benchmark **NSL-KDD** dataset:

| Category | Description | Examples in Training | Novel in Test Set (Zero-Day) |
| :--- | :--- | :--- | :--- |
| **Normal** | Legitimate user network traffic | HTTP, FTP, SMTP normal sessions | Normal |
| **DoS** | Denial of Service attacks | `neptune`, `smurf`, `back`, `pod` | `mailbomb`, `apache2`, `processtable`, `udpstorm` |
| **Probe** | Network surveillance & scanning | `ipsweep`, `portsweep`, `nmap`, `satan` | `mscan`, `saint` |
| **R2L** | Unauthorized remote access | `warezclient`, `guess_passwd`, `warezmaster` | `sendmail`, `named`, `snmpgetattack`, `snmpguess`, `xlock`, `xsnoop`, `httptunnel` |
| **U2R** | Unauthorized root privilege escalation | `buffer_overflow`, `rootkit`, `loadmodule` | `ps`, `sqlattack`, `xterm` |

---

## 📁 Project Structure

```bash
Network-Anomaly-Detector/
├── data/
│   ├── KDDTrain+.txt                   # NSL-KDD full training set (~125,973 records)
│   └── KDDTest+.txt                    # NSL-KDD full evaluation set (~22,544 records)
├── autoencoder.py                      # PyTorch Deep Tabular Autoencoder architecture & trainer
├── data_loader.py                      # NSL-KDD parser, feature aligner, & synthetic generator
├── random_forest.py                    # Supervised Random Forest with cost-sensitive weighting
├── xgboost_model.py                    # GPU-accelerated XGBoost classifier with metrics
├── hybrid_detector_random_forest.py    # Hybrid Pipeline: Random Forest + PyTorch Autoencoder
├── hybrid_detector_xbgboost.py         # Hybrid Pipeline: XGBoost + PyTorch Autoencoder
├── n.py                                # Denoising Autoencoder + XGBoost Gatekeeper pipeline
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Kingramy/Network-Anomaly-Detector.git
cd Network-Anomaly-Detector
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If running on a system with an NVIDIA GPU, ensure PyTorch with CUDA support is installed matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)).*

---

## 💻 Usage & Execution

### 1. Standalone Classifiers

Run the **cost-sensitive Random Forest** (benchmarks weighted vs. unweighted models and prints category-level recall):
```bash
python random_forest.py
```

Run the **GPU-accelerated XGBoost Classifier** (outputs feature importances, confusion matrix, and category breakdown):
```bash
python xgboost_model.py
```

### 2. Hybrid Ensembles (Supervised + Autoencoder)

Run the **Random Forest + Autoencoder Hybrid**:
```bash
python hybrid_detector_random_forest.py
```

Run the **XGBoost + Autoencoder Parallel Late Fusion Engine**:
```bash
python hybrid_detector_xbgboost.py
```

Run the **Robust Denoising Autoencoder Pipeline** (`n.py`):
```bash
python n.py
```

### 3. Data Preprocessing & Validation

Verify that your dataset is correctly parsed and feature columns are aligned:
```bash
python data_loader.py
```

---

## 🔬 Methodology & Technical Details

### Cost-Sensitive Sample Weighting
In standard intrusion datasets, attacks like `U2R` and `R2L` comprise less than 1% of training traffic. Standard empirical loss minimization often ignores them. We introduce category weights during supervised training:
$$\mathcal{L}_{weighted} = \sum_{i=1}^{N} w_{y_i} \cdot \ell(y_i, \hat{y}_i)$$
where $w_{\text{U2R}} = 15.0$ and $w_{\text{R2L}} = 8.0$.

### Semi-Supervised Autoencoder Anomaly Detection
The `TabularAutoencoder` architecture:
- **Encoder**: $D_{in} \to 64 \to 32 \to 12$ (Latent Space) with BatchNorm, LeakyReLU ($\alpha=0.1$), and Dropout ($p=0.1$).
- **Decoder**: $12 \to 32 \to 64 \to D_{in}$ with Sigmoid activation on normalized inputs.
- **Loss**: Mean Squared Error ($MSE$):
$$\text{MSE}(x, \hat{x}) = \frac{1}{D}\sum_{j=1}^{D}(x_j - \hat{x}_j)^2$$
- **Threshold Calibration**: Calibrated using the 96th–98th percentile of reconstruction loss over normal held-out validation traffic.

### Decision-Level Late Fusion
The final prediction $\hat{y}_{\text{hybrid}}$ is determined by:
$$\hat{y}_{\text{hybrid}} = \hat{y}_{\text{supervised}} \lor \hat{y}_{\text{autoencoder}} = \max\left(\mathbb{I}(P_{\text{sup}} \ge \tau_{\text{sup}}),\, \mathbb{I}(\text{MSE} > \tau_{\text{ae}})\right)$$

This guarantees that high-confidence supervised classifications remain intact while novel attacks with elevated reconstruction errors trigger security alerts.

---

## 📈 Evaluation & Benchmarks

When evaluated on the external **NSL-KDD Test+** set (containing 17 novel attack classes never seen during training):

| Model | Overall Accuracy | Anomaly Recall | Novel Attacks Detected | Key Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Alone** | ~78% | ~68% | ~50% - 60% | Strong baseline on standard attacks |
| **XGBoost Alone** | ~80% | ~71% | ~60% - 68% | Ultra-fast inference, high DoS/Probe precision |
| **Autoencoder Alone** | ~73% | ~79% | ~75% - 82% | Unsupervised anomaly discovery |
| **Hybrid (XGBoost + Autoencoder)** | **~81% - 84%** | **~86% - 91%** | **~85% - 92%** | **Highest recall on novel & zero-day attacks** |

---

## 📦 Dependencies

- Python 3.9+
- `torch` & `torchvision` (PyTorch with CUDA support recommended)
- `xgboost`
- `scikit-learn`
- `pandas`
- `numpy`

A standard `requirements.txt` is provided:
```text
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
torch>=2.0.0
xgboost>=1.7.0
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).