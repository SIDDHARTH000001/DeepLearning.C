# Deep Learning in C

This repository contains my experiments and implementations of deep learning concepts using the C programming language. So far, I have focused on building foundational components for Recurrent Neural Networks (RNNs) and Self-Attention mechanisms from scratch.

## Overview
This project is an attempt to delve into the core principles of deep learning by implementing neural network components without relying on high-level libraries or frameworks. It emphasizes understanding the underlying math and mechanics of deep learning.

### Key Features
- **Recurrent Neural Networks (RNNs):**
  - Built from scratch using matrix operations.
  - Implemented forward and backward passes for sequential data processing.
  - Gradient descent for training.

- **Self-Attention Mechanism:**
  - Implementation of scaled dot-product attention.
  - Key, Query, and Value matrix operations.
  - Foundation for attention-based models like Transformers.

### Why C?
C provides direct control over memory and performance, making it an excellent choice for exploring the efficiency of neural network computations at a low level. This project aims to demonstrate that even advanced deep learning concepts can be implemented in C.

## Directory Structure
```
├── src
│   ├── rnn.c                # Implementation of Recurrent Neural Networks
│   ├── self_attention.c     # Self-Attention mechanism code
│   └── utils.c              # Utility functions for matrix operations
├── include
│   ├── rnn.h                # Header file for RNNs
│   ├── self_attention.h     # Header file for Self-Attention
│   └── utils.h              # Header file for utilities
├── examples
│   ├── rnn_example.c        # Example usage of RNN implementation
│   └── self_attention_example.c # Example usage of Self-Attention
├── Makefile                 # Build system
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites
- GCC or any other C compiler.
- Make (optional, for building with the provided Makefile).

### Building the Project
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Build the project using the Makefile:
   ```bash
   make
   ```

### Running Examples
Run the example programs to test the implementations:

- RNN Example:
  ```bash
  ./bin/rnn_example
  ```

- Self-Attention Example:
  ```bash
  ./bin/self_attention_example
  ```

## Future Work
- Implementation of Transformers.
- Optimization of matrix operations for larger-scale models.
- Support for GPU acceleration using CUDA.

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests for improvements or new features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Inspiration from foundational deep learning research papers.
- Online resources for matrix operations and C programming practices.

# Execute LLM.C >  .\a.exe train/test my_trained_model.bin "<input>" #char
