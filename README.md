# HollowIA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)](#project-status)

> **HollowIA** is an experimental autoregressive language model built from scratch in Python and PyTorch using the Transformer architecture.

The project explores how a **compact Transformer** learns statistical language patterns directly from raw data and how lexical retrieval techniques (such as **BM25**) can complement generative models during retrieval-assisted inference.

---

## 📌 Overview

HollowIA is designed as a practical, hands-on exploration of the core stages involved in building language models (LLMs):
- **Data preparation and encoding**
- **Transformer architecture and training**
- **Autoregressive text generation**
- **Information retrieval integration (Lexical RAG)**

---

## ✨ Key Features

* **Transformer Architecture**: Neural sequence modeling tailored for autoregressive text generation.
* **Byte-Level Representation**: Text is encoded directly as UTF-8 bytes, bypassing conventional subword tokenizers.
* **BM25 Lexical Retrieval**: An experimental relevance ranking system to search and retrieve pertinent passages from a corpus.
* **Custom Training Pipeline**: Preprocessing and training workflows adapted for custom datasets.
* **Interactive CLI Inference**: A command-line interface for real-time testing and prompt evaluation.
* **Checkpointing System**: Save and reload model parameters to resume training or run inference seamlessly.

---

## 🏗 Pipeline Architecture

```text
  [ Training Corpus ]
           │
           ▼
  [ Text Preprocessing ]
           │
           ▼
  [ Byte-Level Encoding (UTF-8) ]
           │
           ▼
  [ Transformer Training ]
           │
           ▼
  [ Model Checkpoint (.pt) ]
           │
           ▼
  [ Autoregressive Generation ]

Retrieval & Generation Workflow
Retrieval and generation act as complementary components:
 * BM25: Identifies and ranks relevant corpus passages based on exact keyword matches.
 * Transformer: Generates continuous sequence outputs based on learned neural representations.
 * Retrieval Integration: Extracted context from BM25 can be prepended to the prompt to inform model inference.

## 🛠 Technology Stack
| Technology | Role in Project |
|---|---|
| Python | Primary development language |
| PyTorch | Neural network modeling, autograd, and training routines |
| Transformer | Autoregressive language modeling |
| UTF-8 Byte Encoding | Tokenization-free byte-level text representation |
| BM25 | Algorithmic lexical retrieval |

## 🚀 Getting Started
Prerequisites
 * Python 3.8+
 * PyTorch (Refer to the official PyTorch Installation Guide to select the correct build for your hardware)
 * Git

> 💡 Note for Mobile/Termux Environments: Installing PyTorch on Android or Termux may require additional setup or custom dependency binaries.
> 
Installation
 * Clone the repository:
   git clone [https://github.com/YOUR_USERNAME/HollowIA.git](https://github.com/YOUR_USERNAME/HollowIA.git)
cd HollowIA

 * Install the required dependencies:
   pip install -r requirements.txt

## 💻 Usage
1. Training
To train a new model instance using default parameters:
python hollowia.py --treinar --novo --passos 1000

2. Interactive Inference
To interact with the model using a saved checkpoint:
python conversa.py

## ⚠️ Project Status & Limitations
HollowIA is strictly an experimental research project under active development.
 * Generation Quality: Output text may sometimes be incoherent, repetitive, or factually inaccurate (hallucinations).
 * Byte-Level Sequence Length: While language-agnostic, byte-level modeling increases the sequence length needed to represent text compared to subword tokenizers (e.g., BPE).
 * Retrieval Independence: BM25 ranking does not guarantee that the Transformer will perfectly synthesize or cite retrieved context.
 * Intended Use: Built for educational, experimentation, and research purposes—not designed for production environments.

## 🗺️ Roadmap
 * [ ] Improve generation coherence and output consistency.
 * [ ] Implement formal evaluation metrics (e.g., Perplexity, validation loss tracking).
 * [ ] Refine corpus preprocessing and data cleaning pipelines.
 * [ ] Optimize the integration between BM25 retrieval contexts and the attention window.
 * [ ] Explore advanced positional encoding and inference optimization techniques.
 * [ ] Add automated testing and experiment reproducibility scripts.

## 🤝 Contributing
Contributions, feedback, and reproducible experiment reports are welcome!
If you wish to contribute:
 * Open an Issue describing the bug or feature request.
 * Submit a Pull Request explaining the changes made and tests conducted.

## 📄 License
No license is currently specified. Proprietary rights apply until a formal LICENSE file is committed to the repository.
<p align="center">
<i>HollowIA — Independent experimentation in neural language modeling.</i>
</p>

