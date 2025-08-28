# QuantGPT

[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace-blue.svg)](https://huggingface.co/AryanNsc/QuantGPT)

QuantGPT is a smaller GPT-2 (124M) variant with optimized training scripts.  
It comes with training code and Colab-friendly settings for fast prototyping on limited resources.

---

## 🚀 Base Model
We provide a compact GPT-2-style language model (~124M parameters) optimized for speed and efficiency.

- **Framework**: PyTorch  
- **Dataset**: [FineWeb-Mini](https://huggingface.co/datasets/AryanNsc/FineWeb-Mini)  
- **Training Tokens**: ~100M  
- **Precision**: Uses `fp16` for Colab compatibility. (`bf16` requires NVIDIA Ampere GPUs and above)  

---

## 📂 Dataset
For training, we use the **FineWeb-MINI** dataset.  

```python
from datasets import load_dataset

dataset = load_dataset("AryanNsc/FineWeb-Mini")
```

This dataset is already tokenized and sharded for optimized training.

---

## 💻 Training on Colab
1. First, prepare data shards:

```bash
python data_prep.py
```

2. Then, start training:

```bash
python train/gpt-2-arch.py
```

Colab uses a single T4 GPU; `fp16` precision is enabled by default.

---

## 🧩 New Models Training
In addition to QuantGPT, we are also training **QuantMobile-17M** —  
a small, highly efficient, and deployable transformer-based language model.

Key highlights of **MobileLLM**:
- Uses **RMSNorm** and **Rotary Position Embeddings (RoPE)** for better efficiency.
- Implements **multi-query attention** for faster inference.
- Integrated **safetensors** for optimized checkpointing.
- Uses **PyArrow Parquet** data loaders for high-throughput token streaming.
- Fully supports **DDP (Distributed Data Parallel)** training out of the box.
- Integrated with **Weights & Biases** (`wandb`) for live monitoring.

### Training Example
```bash
python train/mobile_llm-arch.py
```

---

## 📦 Model Checkpoints
We save model checkpoints using `safetensors` format for fast loading and secure storage.

- Final trained model → `mobile_llm_final.safetensors`
- Intermediate checkpoints → `checkpoints/mobile_llm_step_<step>.safetensors`

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---


## 🌐 Links
- **Model**: [QuantGPT on HuggingFace](https://huggingface.co/AryanNsc/QuantGPT)
- **QuantMobile**: [QuantMobile on HuggingFace](https://huggingface.co/AryanNsc/QuantMobile-17M)
- **Dataset**: [FineWeb-MINI](https://huggingface.co/datasets/AryanNsc/FineWeb-Mini)
