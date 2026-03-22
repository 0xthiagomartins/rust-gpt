# Building GPT from Scratch in Pure Rust (Step-by-Step)

## Introduction

In this guide, we will build a GPT-style decoder-only Transformer in pure Rust.
No PyTorch, no TensorFlow, no hidden ML framework. Only Rust, math, and careful engineering.

The goal is not to beat production LLMs. The goal is to understand every moving part and own the full stack:

- tokenization
- tensor operations
- attention
- forward and backward pass
- optimizer
- training loop
- autoregressive text generation

If you can build this, you are not just using AI, you are building AI infrastructure.

---

## Part 1: Project Setup

Start with a clean Rust binary project:

```bash
cargo new rust-gpt --bin
cd rust-gpt
```

Suggested layout:

```text
src/
  main.rs
  model/
    gpt.rs
    transformer.rs
    attention.rs
    feedforward.rs
    norm.rs
  tensor/
    tensor.rs
    ops.rs
  tokenizer/
    tokenizer.rs
  training/
    trainer.rs
    optimizer.rs
  utils/
    rng.rs
```

Update `src/main.rs` to declare modules early:

```rust
mod model;
mod tensor;
mod tokenizer;
mod training;
mod utils;

fn main() {
    println!("RustGPT bootstrap ready");
}
```

Why this matters: a clear module structure prevents the project from collapsing into one giant file.

---

## Part 2: Random Number Generation

We need randomness for:

- parameter initialization
- sampling during inference
- data shuffling

Create `src/utils/rng.rs`:

```rust
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn uniform(&mut self) -> f32 {
        const DEN: f64 = (1u64 << 53) as f64;
        let v = (self.next_u64() >> 11) as f64 / DEN;
        v as f32
    }

    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        let mut u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        if u1 <= 0.0 {
            u1 = 1e-7;
        }
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + std * z0
    }
}

pub fn shuffle<T>(items: &mut [T], rng: &mut XorShift64) {
    for i in (1..items.len()).rev() {
        let j = (rng.uniform() * (i as f32 + 1.0)).floor() as usize;
        items.swap(i, j);
    }
}
```

This gives us deterministic, reproducible runs from a seed.

---

## Part 3: Data Loading and Char Tokenizer

For MVP, a char-level tokenizer is enough and keeps the pipeline simple.

Create `src/tokenizer/tokenizer.rs`:

```rust
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone)]
pub struct Tokenizer {
    stoi: HashMap<char, usize>,
    itos: Vec<char>,
    bos_id: usize,
}

impl Tokenizer {
    pub fn build_from_text(text: &str) -> Self {
        let mut set = BTreeSet::new();
        for ch in text.chars() {
            set.insert(ch);
        }

        let mut itos: Vec<char> = set.into_iter().collect();
        let bos = '^';
        if !itos.contains(&bos) {
            itos.push(bos);
        }

        let mut stoi = HashMap::with_capacity(itos.len());
        for (i, ch) in itos.iter().enumerate() {
            stoi.insert(*ch, i);
        }

        let bos_id = *stoi.get(&bos).unwrap();
        Self { stoi, itos, bos_id }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.stoi.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.itos.get(id).copied())
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub fn bos_id(&self) -> usize {
        self.bos_id
    }
}
```

Load data from a plain text file (one corpus for now):

```rust
use std::fs;

pub fn load_text(path: &str) -> std::io::Result<String> {
    fs::read_to_string(path)
}
```

---

## Part 4: Tensor Core

Now we build the minimal tensor abstraction.

Create `src/tensor/tensor.rs`:

```rust
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "data length must match shape product");
        Self { data, shape: shape.to_vec() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn rows_cols(&self) -> (usize, usize) {
        assert_eq!(self.shape.len(), 2, "expected 2D tensor");
        (self.shape[0], self.shape[1])
    }

    pub fn get2d(&self, r: usize, c: usize) -> f32 {
        let (_, cols) = self.rows_cols();
        self.data[r * cols + c]
    }

    pub fn set2d(&mut self, r: usize, c: usize, v: f32) {
        let (_, cols) = self.rows_cols();
        self.data[r * cols + c] = v;
    }
}
```

Create `src/tensor/ops.rs`:

```rust
use super::tensor::Tensor;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (ar, ac) = a.rows_cols();
    let (br, bc) = b.rows_cols();
    assert_eq!(ac, br, "matmul shape mismatch");

    let mut out = Tensor::zeros(&[ar, bc]);
    for i in 0..ar {
        for k in 0..ac {
            let aik = a.get2d(i, k);
            for j in 0..bc {
                let idx = i * bc + j;
                out.data[idx] += aik * b.get2d(k, j);
            }
        }
    }
    out
}

pub fn add_inplace(a: &mut Tensor, b: &Tensor) {
    assert_eq!(a.shape, b.shape);
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x += *y;
    }
}

pub fn transpose2d(x: &Tensor) -> Tensor {
    let (r, c) = x.rows_cols();
    let mut out = Tensor::zeros(&[c, r]);
    for i in 0..r {
        for j in 0..c {
            out.set2d(j, i, x.get2d(i, j));
        }
    }
    out
}

pub fn softmax_last_dim(x: &Tensor) -> Tensor {
    let (rows, cols) = x.rows_cols();
    let mut out = Tensor::zeros(&[rows, cols]);

    for r in 0..rows {
        let row = &x.data[r * cols..(r + 1) * cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;

        for c in 0..cols {
            let e = (row[c] - m).exp();
            out.data[r * cols + c] = e;
            sum += e;
        }

        let inv = 1.0 / sum.max(1e-9);
        for c in 0..cols {
            out.data[r * cols + c] *= inv;
        }
    }

    out
}

pub fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x * x * x)).tanh())
}
```

At this point, you have the core math primitives needed for forward pass.

---

## Part 5: Model Configuration and Parameters

Create a shared config:

```rust
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub embed_dim: usize,
}
```

Basic components:

```rust
use crate::tensor::tensor::Tensor;

pub struct Embedding {
    pub weight: Tensor, // [vocab_size, embed_dim]
}

pub struct LayerNorm {
    pub gamma: Tensor, // [embed_dim]
    pub beta: Tensor,  // [embed_dim]
}

pub struct Linear {
    pub w: Tensor, // [out, in]
    pub b: Tensor, // [out]
}
```

Then assemble Transformer pieces:

```rust
pub struct MultiHeadAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
}

pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub ff: FeedForward,
}

pub struct GPT {
    pub token_embedding: Embedding,
    pub position_embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    pub head: Linear,
}
```

Initialize parameters with small Gaussian noise (`std ~ 0.02`) and zeros for bias.

---

## Part 6: Forward Building Blocks

### 6.1 LayerNorm (forward)

```rust
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| {
        let d = *v - mean;
        d * d
    }).sum::<f32>() / n as f32;

    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        let norm = (x[i] - mean) * inv;
        out[i] = norm * gamma[i] + beta[i];
    }
}
```

### 6.2 Linear projection

```rust
pub fn linear(x: &[f32], w: &Tensor, b: &[f32], out: &mut [f32]) {
    let (rows, cols) = w.rows_cols();
    assert_eq!(cols, x.len());
    assert_eq!(rows, out.len());

    for r in 0..rows {
        let mut acc = b[r];
        for c in 0..cols {
            acc += w.get2d(r, c) * x[c];
        }
        out[r] = acc;
    }
}
```

### 6.3 Causal self-attention (single head concept)

For position `t`, attend only to `0..=t`.

```text
scores = (Q K^T) / sqrt(d_head)
scores = mask_future(scores)
probs  = softmax(scores)
out    = probs V
```

For multi-head, split embedding into `n_heads`, run this per head, then concatenate.

---

## Part 7: Transformer Block Forward

A standard pre-norm block:

```text
x = x + Attention(LN(x))
x = x + MLP(LN(x))
```

Pseudo-code:

```rust
fn block_forward(x: &mut [f32], block: &TransformerBlock, ctx: &AttentionCache) {
    let x_res = x.to_vec();
    let x_ln = ln(x, &block.ln1);
    let attn = mha_forward(&x_ln, &block.attn, ctx);
    for i in 0..x.len() {
        x[i] = x_res[i] + attn[i];
    }

    let x_res2 = x.to_vec();
    let x_ln2 = ln(x, &block.ln2);
    let ff = ff_forward(&x_ln2, &block.ff);
    for i in 0..x.len() {
        x[i] = x_res2[i] + ff[i];
    }
}
```

---

## Part 8: Full GPT Forward Pass

For each token position:

1. Get token embedding
2. Add positional embedding
3. Pass through all transformer blocks
4. Apply final layer norm
5. Project to vocabulary logits

Shape intuition:

- input ids: `[T]`
- hidden states: `[T, C]`
- logits: `[T, V]`

Where:

- `T = block_size`
- `C = embed_dim`
- `V = vocab_size`

---

## Part 9: Loss Function (Cross-Entropy)

For target token `y` and logits `z`:

```text
p = softmax(z)
loss = -log(p[y])
```

Stable implementation tip: subtract max logit before exponentiation.

Minimal function:

```rust
pub fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for &z in logits {
        sum += (z - m).exp();
    }
    let log_prob = logits[target] - m - sum.ln();
    -log_prob
}
```

---

## Part 10: Backward Pass (Manual Gradients)

This is the hardest part.

Recommended order:

1. Implement backward for `linear`
2. Implement backward for `layer_norm`
3. Implement backward for `softmax + cross_entropy`
4. Implement backward for attention (Q, K, V, projection)
5. Chain all gradients through each block in reverse order

Core linear derivatives:

```text
y = W x + b

dL/dx = W^T dL/dy
dL/dW = dL/dy * x^T
dL/db = dL/dy
```

For training, store forward activations in a cache struct per layer and position.

```rust
pub struct TrainCache {
    pub x_embed: Vec<Vec<f32>>,
    pub x_ln1: Vec<Vec<f32>>,
    pub attn_out: Vec<Vec<f32>>,
    pub x_ln2: Vec<Vec<f32>>,
    pub ff_hidden: Vec<Vec<f32>>,
}
```

Without this cache, manual backprop becomes extremely painful.

---

## Part 11: Adam Optimizer

Adam update rule:

```text
m_t = beta1 * m_{t-1} + (1-beta1) * g_t
v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
m_hat = m_t / (1-beta1^t)
v_hat = v_t / (1-beta2^t)
param -= lr * m_hat / (sqrt(v_hat)+eps)
```

Rust struct:

```rust
pub struct Adam {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}
```

Use one optimizer state vector per parameter tensor.

---

## Part 12: Training Loop

A simple training loop looks like this:

```rust
for step in 0..max_steps {
    let (x_batch, y_batch) = sample_batch(&tokens, batch_size, block_size, &mut rng);

    let (logits, cache) = model.forward_train(&x_batch);
    let loss = compute_batch_loss(&logits, &y_batch);

    let grads = model.backward(&cache, &y_batch);
    optimizer.step(model.parameters_mut(), grads);

    if step % 100 == 0 {
        println!("step {:6} | loss {:.4}", step, loss);
    }
}
```

Training stability tips:

- start with tiny model (`embed_dim=64`, `n_layers=2`, `n_heads=4`)
- use gradient clipping (`max_norm=1.0`)
- keep learning rate conservative (`1e-3` to `3e-4`)
- verify small pieces against known outputs before full training

---

## Part 13: Inference and Text Generation

Generation is autoregressive:

1. encode prompt
2. run forward
3. take last-position logits
4. sample next token
5. append token and repeat

Greedy version (MVP):

```rust
pub fn greedy_next_token(logits: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}
```

Then optionally add temperature and top-k.

---

## Part 14: Running the Project

During early development:

```bash
cargo check
cargo run
cargo test
```

For performance tests:

```bash
cargo run --release
```

Profiling suggestions:

- `cargo flamegraph` for hotspots
- benchmark matrix multiply and attention loops separately
- reduce allocations in inner loops

---

## Part 15: Validation Checklist

Before calling your implementation “working”, validate these items:

- tokenizer encode/decode round-trip works
- matmul tests pass on small known matrices
- softmax sums to ~1.0 for every row
- forward pass produces finite logits (no NaN/Inf)
- loss decreases on a small toy dataset
- generation outputs coherent local patterns

If loss explodes:

- lower learning rate
- clip gradients
- inspect layer norm and softmax numerical stability

---

## Part 16: Next Upgrades

After MVP is stable, prioritize:

- BPE tokenizer
- model checkpoint save/load
- mixed precision experimentation
- multi-threaded kernels
- optional GPU backend (`wgpu`)

---

## Final Notes

This project is intentionally hard.

That is exactly why it is a high-signal portfolio piece.
You are demonstrating that you can reason across:

- math
- systems programming
- memory/performance constraints
- software architecture

From here, the best move is to implement each part in separate PR-sized chunks and keep tests close to every math primitive.
