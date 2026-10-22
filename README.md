HollowIA

An experimental Transformer-based language model implemented in Python and PyTorch.

HollowIA is an independent machine learning project focused on exploring neural language modeling, autoregressive text generation, and information retrieval.

The project investigates how a compact Transformer can learn statistical patterns from text and how retrieval techniques, such as BM25, can complement generative models.

Overview

HollowIA is designed as a practical exploration of the core components of language model development, including data preparation, model training, text generation, and retrieval-assisted inference.

The project emphasizes experimentation, implementation, and understanding the underlying mechanisms of neural language models.

Key Components

- Transformer architecture: Neural sequence modeling for autoregressive text generation.
- Byte-level representation: Text is represented as UTF-8 bytes, avoiding the need for a conventional subword tokenizer.
- Custom training corpus: Model training using a project-specific text dataset.
- BM25 retrieval: Experimental lexical retrieval for identifying relevant passages in a corpus.
- Interactive inference: Command-line interaction for evaluating generated text.
- Checkpointing: Saving model parameters for subsequent inference and training experiments.

Architecture

The project explores the following processing pipeline:

Training Corpus
      |
      v
Text Preprocessing
      |
      v
Byte-Level Encoding
      |
      v
Transformer Training
      |
      v
Model Checkpoint
      |
      v
Autoregressive Generation

When retrieval is enabled, BM25 can identify relevant passages from the available corpus. These passages may then be incorporated into the model's input, depending on the implementation.

Retrieval and generation are distinct components: BM25 ranks documents using lexical relevance, while the Transformer generates sequences based on learned statistical patterns.

Technology Stack

Technology| Purpose
Python| Core implementation
PyTorch| Neural network construction and training
Transformer| Autoregressive language modeling
UTF-8 byte encoding| Text representation
BM25| Experimental information retrieval

Requirements

Requirements depend on the execution environment and the specific project version.

- Python
- PyTorch, with a compatible installation for the target platform
- Project dependencies specified in "requirements.txt", if provided
- A training corpus in the format expected by the training scripts

Mobile environments, including Android and Termux, may require additional setup or platform-specific dependency installation.

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/HollowIA.git
cd HollowIA

Install the dependencies listed by the project:

pip install -r requirements.txt

Replace "YOUR_USERNAME" with the repository owner's GitHub username.

Note: PyTorch availability varies by operating system, Python version, and hardware configuration. Consult the official "PyTorch installation guide" (https://pytorch.org/get-started/locally/) when necessary.

Usage

The commands below are examples and must match the arguments and filenames implemented in the checked-out version.

Training

python hollowia.py --treinar --novo --passos 1000

This example assumes the training script supports the specified arguments.

Interactive Inference

python conversa.py

This example assumes that "conversa.py" is included in the repository and that a compatible trained checkpoint is available.

Current Status

HollowIA is an experimental project under active development.

Current work focuses on evaluating training behavior, improving text generation, and investigating the integration of information retrieval with neural language modeling.

Generated text may be incoherent, repetitive, or factually inaccurate. Retrieval results should be evaluated independently from the quality of the generated responses.

The project is intended for learning, experimentation, and research into language model implementation rather than as a production-ready conversational AI system.

Limitations

- Model performance depends on dataset quality, model capacity, and training configuration.
- Byte-level modeling can require more sequence positions to represent ordinary text than subword tokenization.
- BM25 retrieval does not guarantee that retrieved information will be interpreted or reproduced correctly.
- Generated responses may contain errors or unsupported statements.
- Compatibility depends on the model checkpoint, architecture configuration, dependencies, and execution environment.

Roadmap

Potential areas for future development include:

- [ ] Improve generation coherence and consistency.
- [ ] Evaluate training and validation performance.
- [ ] Refine corpus preprocessing and dataset quality.
- [ ] Improve retrieval integration and relevance evaluation.
- [ ] Investigate positional encoding and inference optimizations.
- [ ] Expand automated evaluation and reproducibility tools.

Roadmap items represent development goals and are not necessarily implemented.

Contributing

Contributions, technical feedback, and reproducible experiments are welcome.

When proposing changes, include a clear description of the modification, its motivation, and any relevant test results.

License

A license has not yet been specified. Until one is added to the repository, permission to redistribute or modify the project should not be assumed.

---

HollowIA — Independent experimentation in neural language modeling.