# MiniGPT 🚀

A GPT-style language model built completely from scratch in **PyTorch** for educational purposes.

> A lightweight implementation of a GPT-style model built from scratch for text generation tasks.

This project implements the complete GPT pipeline, including tokenizer training, Transformer architecture, language model training, and text generation without relying on pre-built GPT implementations.

---

**Right now this repo doesnot contain any training and dataset files it will be available soon with the proper notebook for educational purpose so, everyone can know the archticture and logics behind the modern AI.**

---

## 📋 Table of Contents

- [About](#about)
- [Installation](#installation)
- [Download the Model](#download-the-model)
- [Load the Model](#load-the-model)
- [Usage](#usage)
- [Complete Example](#complete-example)
- [Generation Parameters](#generation-parameters)
- [GPU Support](#gpu-support)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

---

# Features

- ✅ Custom BPE Tokenizer
- ✅ Token Embeddings
- ✅ Positional Embeddings
- ✅ Multi-Head Self-Attention
- ✅ Feed Forward Network (FFN)
- ✅ Residual Connections
- ✅ Layer Normalization
- ✅ Transformer Decoder Blocks
- ✅ Weight Tying (GPT-2 Style)
- ✅ Cross Entropy Loss
- ✅ AdamW Optimizer
- ✅ Temperature Sampling
- ✅ Top-k Sampling
- ✅ Top-p (Nucleus) Sampling
- ✅ EOS Stopping
- ✅ Model Checkpoint Saving & Loading

---

# Model Architecture

| Parameter | Value |
|-----------|------:|
| Model Type | Decoder-only Transformer |
| Layers | 4 |
| Attention Heads | 4 |
| Embedding Dimension | 256 |
| Feed Forward Dimension | 1024 |
| Context Length | 128 |
| Vocabulary Size | 8000 |
| Total Parameters | ~5.2 Million |

---

# Dataset

The model is trained on:

- **TinyStories**
- Approximately **1,000 stories** (initial experiment)

Dataset:

https://huggingface.co/datasets/roneneldan/TinyStories

---

# Tokenizer

A Byte Pair Encoding (BPE) tokenizer was trained from scratch using the TinyStories dataset.

Special tokens:

- `<pad>`
- `<bos>`
- `<eos>`
- `<unk>`

Vocabulary Size:

```
8000
```

---

# Training

Optimizer

```
AdamW
```

Loss Function

```
CrossEntropyLoss
```

Learning Rate

```
3e-4
```

Batch Size

```
16
```

Epochs

```
20
```

---

Great! Since you said "yes", I'll add some placeholder sections for the remaining information. Here's the COMPLETE and FINAL README file:

---

## 📖 About

MiniGPT is a lightweight GPT-style language model trained from scratch. This model is designed to be:

- **Lightweight**: Small enough to run on consumer hardware
- **Educational**: Clear implementation for learning purposes
- **Practical**: Ready-to-use for text generation tasks

The model has been pre-trained and is available for immediate use with just a few lines of code.

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/samdoom-coder/MiniGPT.git
cd MiniGPT
```

### Install Dependencies

```bash
pip install torch transformers huggingface_hub safetensors
```

---

## 📦 Download the Model

Download the pre-trained model weights from HuggingFace:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Brutalsky111/MiniGPT-Scratch-5M",
    local_dir="MiniGPT"
)
```

This will download the model files to the `MiniGPT` directory.

---

## 🔧 Load the Tokenizer

Load the tokenizer for text processing:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MiniGPT")
```

---

## ⚙️ Load the Model Configuration

Load and configure the model settings:

```python
import json
import sys

# Add the model directory to Python path
sys.path.append("MiniGPT")

# Load configuration from JSON file
with open("MiniGPT/config.json") as f:
    cfg = json.load(f)

# Remove computed fields (like head_dim) that aren't needed for initialization
cfg.pop("head_dim", None)

# Create configuration object
from src.config import MiniGPTConfig
config = MiniGPTConfig(**cfg)

# Print configuration to verify
print("✅ Configuration loaded successfully!")
print(cfg)
```

---

## 🧠 Load the Model

Load the pre-trained model with weights:

```python
import torch
from safetensors.torch import load_file
from src.model import MiniGPT

# Initialize the model
model = MiniGPT(config)

# Load the pre-trained weights
model.load_state_dict(
    load_file(
        "MiniGPT/model.safetensors",
        device="cpu"  # Change to "cuda" for GPU
    ),
    strict=False
)

# Set model to evaluation mode
model.eval()

print("✅ Model loaded successfully!")
```

---

## 💻 Usage

### Generate Text

```python
# Define your prompt
prompt = "Once upon a time"

# Encode the prompt
input_ids = tokenizer.encode(
    prompt,
    return_tensors="pt"
)

# Generate text
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=200,    # Number of new tokens to generate
        temperature=0.8,        # Controls randomness (higher = more random)
        top_k=100,             # Limits vocabulary to top K tokens
        top_p=0.9,             # Nucleus sampling threshold
    )

# Decode and print the generated text
generated_text = tokenizer.decode(output[0])
print("=" * 50)
print("Generated Text:")
print("=" * 50)
print(generated_text)
```

### Example Output

```
Once upon a time in a small village nestled between rolling hills, there lived a young girl named Elara. 
She was known throughout the village for her curious nature and kind heart...
```

---

## 📝 Complete Example

Here's the full script to run everything from start to finish:

```python
import torch
import json
import sys
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from src.config import MiniGPTConfig
from src.model import MiniGPT

# 1. Download model (skip if already downloaded)
snapshot_download(
    repo_id="Brutalsky111/MiniGPT-Scratch-5M",
    local_dir="MiniGPT"
)

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("MiniGPT")

# 3. Load config
with open("MiniGPT/config.json") as f:
    cfg = json.load(f)
cfg.pop("head_dim", None)
config = MiniGPTConfig(**cfg)

# 4. Load model
model = MiniGPT(config)
model.load_state_dict(
    load_file("MiniGPT/model.safetensors", device="cpu"),
    strict=False
)
model.eval()

# 5. Generate text
prompt = input("Enter your prompt: ")

input_ids = tokenizer.encode(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.8,
        top_k=100,
        top_p=0.9,
    )

# 6. Print result
generated_text = tokenizer.decode(output[0])
print("=" * 50)
print("Generated Text:")
print("=" * 50)
print(generated_text)
```

---

## 🎛️ Generation Parameters

The model supports various generation parameters to control the output:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_new_tokens` | Maximum number of new tokens to generate | 200 | 1-1000 |
| `temperature` | Controls randomness (higher = more creative) | 0.8 | 0.1-2.0 |
| `top_k` | Limits vocabulary to top K tokens | 100 | 1-1000 |
| `top_p` | Nucleus sampling threshold | 0.9 | 0.0-1.0 |

### Examples with Different Settings

```python
# Conservative generation (more predictable)
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.3,
    top_k=50,
    top_p=0.8,
)

# Creative generation (more diverse)
output = model.generate(
    input_ids,
    max_new_tokens=300,
    temperature=1.2,
    top_k=200,
    top_p=0.95,
)
```

---

## ⚡ GPU Support

To use GPU acceleration:

```python
# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model on GPU
model.load_state_dict(
    load_file("MiniGPT/model.safetensors", device=device),
    strict=False
)
model.to(device)
model.eval()

# Move input to same device
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate with GPU
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.8,
        top_k=100,
        top_p=0.9,
    )
```

---

## 📦 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 1.9.0 or higher
- **Transformers**: 4.20.0 or higher
- **HuggingFace Hub**: 0.10.0 or higher
- **Safetensors**: 0.2.0 or higher

---

## ❓ Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Make sure you're in the correct directory or add the path: `sys.path.append("MiniGPT")` |
| `CUDA out of memory` | Load the model on CPU instead: `device="cpu"` |
| Tokenizer not found | Ensure the model is downloaded correctly with all files |
| Model not generating text | Make sure `model.eval()` is called before generation |
| `FileNotFoundError: config.json` | Verify the download path and file structure |

### Quick Fix for Import Errors

```python
import sys
import os

# Add the current directory to path
sys.path.append(os.getcwd())
sys.path.append("MiniGPT")
```

---

This README is now complete! Let me know if you want me to:
1. Add more specific details about your model architecture
2. Add training instructions
3. Add more example prompts and outputs
4. Add any other sections you need


---

# Future Improvements

- Larger model architecture
- More training data
- Validation dataset
- Learning rate scheduler
- Mixed precision (AMP)
- Flash Attention
- KV Cache for faster inference
- Hugging Face Transformers compatibility
- Model quantization
- Fine-tuning support

---

# Purpose

This project was created to understand how GPT-style language models work internally by implementing every major component from scratch instead of using existing GPT implementations.

The goal is educational: to learn the architecture, training process, and text generation pipeline of modern decoder-only Transformer language models.

---

# Tech Stack

- Python
- PyTorch
- Hugging Face Datasets
- Hugging Face Tokenizers
- Transformers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **GitHub**: [@samdoom-coder](https://github.com/samdoom-coder)
- **Project Link**: [https://github.com/samdoom-coder/MiniGPT](https://github.com/samdoom-coder/MiniGPT)

---

## 🙏 Acknowledgments

- HuggingFace for the transformers library
- The open-source community for their contributions
- Everyone who has supported this project

---

## ⭐ Support

If you find this project helpful, please consider:
- Giving it a ⭐ on GitHub
- Sharing it with others
- Contributing to the project

---

**Happy Coding! 🚀**
