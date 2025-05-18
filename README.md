# Differentially-Federated-LoRA

A privacy-preserving federated learning framework that integrates Low‑Rank Adaptation (LoRA) with Differential Privacy (DP) on transformer models using Flower and Opacus.

---

## 🚀 Features

* **LoRA Fine‑Tuning**: Efficiently adapt large transformer models with low‑rank adapters.
* **Federated Learning**: Distribute training across multiple clients without sharing raw data (FedAvg).
* **Differential Privacy**: Guarantee \$(arepsilon, \delta)\$‑DP by adding Gaussian noise and clipping gradients.
* **Modular Design**: Plug-and-play modules for device management, data loading, model adaptation, privacy, and federated orchestration.
* **End‑to‑End Pipeline**: Single notebook (`main.ipynb`) to run experiments, collect metrics, and visualize results.

---

## 📋 Prerequisites

* Python 3.8+
* PyTorch 1.\*\*
* Flower 1.\*\*
* Opacus 1.\*\*
* Hugging Face Transformers & Datasets
* peft 0.\*\*
* scikit-learn
* matplotlib, seaborn

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🏗️ Project Structure

```
├── clients.py              # Flower federated client implementation
├── server.py               # Flower federated server setup
├── lora.py                 # LoRA adapter integration on transformers
├── differential_privacy.py # Opacus DP training wrapper
├── data.py                 # IMDB dataset loading & tokenization
├── device.py               # GPU/CPU/MPS device utilities
├── multiprocessing_helpers.py # Helpers for parallel client processes
├── main.ipynb              # Orchestrates experiments & visualizations
├── results_2.csv           # Logged metrics from experimental runs
├── requirements.txt        # Python dependencies
└── README.md               # Project overview (this file)
```

---

## 💡 Quick Start

### 1. Launch the Federated Server

```bash
flower-superlink --insecure --serverappio-api-address="0.0.0.0:9090"
```

### 2. Launch One or More Clients

Change the client API port per client:

```bash
flower-supernode --insecure --superlink="localhost:9092" --clientappio-api-address="0.0.0.0:9096"
# For additional clients, increment the 9096 port accordingly
```

### 3. Run Experiments in `main.ipynb`

* Define scenarios:

  * **LoRA Only**
  * **LoRA + Federated Learning**
  * **LoRA + Federated Learning + Differential Privacy**
* Select transformer models and LoRA target modules.
* Execute cells to start server & spawn clients (or run local training).
* Collect metrics and generate plots (Figures 1–15 as in the report).

---

## 🔍 Module Overview

### `device.py`

* Detects available hardware (CUDA, MPS, CPU) and moves models/tensors accordingly.

### `data.py`

* Loads IMDB dataset, tokenizes with Hugging Face tokenizer (padding/truncation), and returns PyTorch DataLoaders.

### `lora.py`

* Creates a `peft.LoraConfig` and wraps a transformer model to fine‑tune only low‑rank adapters.

### `differential_privacy.py`

* Wraps model and optimizer in `opacus.PrivacyEngine` with noise multiplier and gradient clipping.

### `clients.py`

* Implements `FlowerClient` for federated training and evaluation:

  * `get_parameters`, `set_parameters` for weight exchange
  * `fit` trains locally (with optional DP)
  * `evaluate` returns loss & accuracy metrics

### `server.py`

* Configures and starts Flower server using `FedAvg` strategy and orchestrates client rounds.

### `main.ipynb`

* Orchestrates full pipeline:

  1. Define scenarios & models
  2. Launch server process
  3. Start client processes or local training
  4. Log metrics to CSV
  5. Plot accuracy, precision, recall, F1, ROC‑AUC, loss, and training time.

---

## 📊 Results & Metrics

Supported metrics (logged in `results_2.csv` and visualized):

* **Accuracy, Precision, Recall, F1 Score**
* **ROC‑AUC**
* **Loss**
* **Training Time**
* **Communication Overhead** (implicit via Flower logs)

Refer to the report’s Figures 1–15 for detailed comparisons across scenarios and models.

---

## 🚧 Future Work

* **Privacy‑Accuracy Trade‑off**: Vary \$\varepsilon \in {0.1,1,5,10}\$ and clipping norms.
* **Alternative FL Strategies**: Implement FedProx, FedAdam, FedNova.
* **Personalized FL**: Explore PerFedAvg and client‐specific models.
* **Security Testing**: Evaluate robustness against membership inference and poisoning attacks.
* **Gradient & Client Contribution Visualization**: Extend notebook to visualize update magnitudes.

---

## 📄 License

This project is provided under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Contributions

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.
