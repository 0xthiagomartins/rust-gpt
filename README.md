# RustGPT

**A decoder-only Transformer (GPT-style) implemented from scratch in pure Rust.**

RustGPT is an educational and engineering-focused project that builds the core of a language model without ML frameworks.  
The objective is to implement the full stack ourselves: tokenization, tensor operations, attention, training loop, and text generation.

## Why This Project

Most developers can call an LLM API.  
Far fewer can build the engine behind one.

RustGPT demonstrates:

- Deep understanding of Transformer internals
- Systems-level thinking (performance, memory, safety)
- End-to-end ML infrastructure design in a low-level language
- Ownership of both model behavior and runtime constraints

## Product Scope

### Primary goals

- Build a functional GPT-style model from scratch
- Implement forward + backward pass manually
- Enable local training and text generation through a Rust CLI
- Keep full control over memory layout and performance decisions

### Non-goals

- Competing with production-scale LLMs
- Supporting billion-parameter models
- Using PyTorch, TensorFlow, ONNX, or similar frameworks

## Current Status

This repository is currently in **bootstrap stage**.

- [x] Rust project initialized
- [ ] Tensor core (`Tensor`, shape ops, matmul, softmax, layer norm)
- [ ] Character tokenizer
- [ ] Embedding + positional embedding
- [ ] Multi-head self-attention
- [ ] Feed-forward network (MLP + GELU)
- [ ] Transformer block stack
- [ ] GPT forward pass
- [ ] Cross-entropy loss + backprop
- [ ] Adam optimizer
- [ ] Greedy text generation

## High-Level Architecture

```text
Raw Text
  -> Tokenizer
  -> Token Embedding + Positional Embedding
  -> N x Transformer Decoder Blocks
       - Multi-Head Self-Attention
       - Feed-Forward Network
       - Residual Connections + LayerNorm
  -> Final LayerNorm
  -> Vocab Projection (Logits)
  -> Sampling
  -> Generated Text
```

## Planned Component Design

```rust
struct Config {
    vocab_size: usize,
    block_size: usize,
    n_layers: usize,
    n_heads: usize,
    embed_dim: usize,
}
```

```rust
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

```rust
struct GPT {
    token_embedding: Embedding,
    position_embedding: PositionalEmbedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    head: Tensor,
}
```

## Training Pipeline (Target Design)

1. Load raw text dataset
2. Tokenize into integer ids
3. Build mini-batches (`x`, `y` shifted targets)
4. Forward pass to obtain logits
5. Compute cross-entropy loss
6. Backpropagate gradients through all layers
7. Update parameters with Adam
8. Periodically evaluate and sample text

## Proposed Repository Layout

```text
rust-gpt/
├── src/
│   ├── main.rs
│   ├── model/
│   │   ├── gpt.rs
│   │   ├── transformer.rs
│   │   ├── attention.rs
│   │   ├── feedforward.rs
│   │   └── norm.rs
│   ├── tensor/
│   │   ├── tensor.rs
│   │   └── ops.rs
│   ├── tokenizer/
│   │   └── tokenizer.rs
│   ├── training/
│   │   ├── trainer.rs
│   │   └── optimizer.rs
│   └── utils/
├── data/
├── models/
└── Cargo.toml
```

## Roadmap

### Phase 1 (MVP)

- Tensor primitives
- Character-level tokenizer
- Forward-only GPT pass
- Minimal text generation

### Phase 2

- Manual backpropagation
- Stable training loop
- Loss monitoring and gradient debugging tools

### Phase 3

- Model checkpoint save/load
- Better batching and memory efficiency
- Benchmark comparisons against Python reference

### Phase 4 (Advanced)

- SIMD and multithreading
- Optional GPU backends (`wgpu` or CUDA bindings)
- Quantization experiments

## Engineering Risks and Mitigations

- Numerical instability: compare intermediate outputs against PyTorch references
- Slow kernels: optimize memory access and reduce allocations
- Exploding gradients: apply clipping and monitor norms
- Debug complexity: expose layer-wise logs and validation hooks

## Local Development

### Prerequisites

- Rust stable toolchain (`rustup`)
- Cargo

### Run

```bash
cargo run
```

### Build

```bash
cargo build --release
```

## Vision

RustGPT is not a wrapper around an AI service.  
It is a foundational model engine built at the systems level.

This project is intentionally difficult, and that is the point.
It is designed to be strong evidence of hands-on capability in:

- Rust systems engineering
- Deep learning fundamentals
- Performance-aware software architecture

## License

To be defined.
