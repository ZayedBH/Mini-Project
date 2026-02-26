# 🐉 Mini Project: Efficient Code Debugging & Simplification

### Using BDH (Baby Dragon Hatchling) Architecture

------------------------------------------------------------------------

# 📌 Project Overview

This project develops a task-specific, lightweight language model for
code debugging and simplification using a BDH (Baby Dragon Hatchling)
architecture.

Instead of relying on large, dense transformer models, we design a
150--180M parameter decoder-only BDH model optimized for:

-   Structured code transformations\
-   Localized bug fixing\
-   Deterministic refactoring\
-   High-efficiency inference

The objective is to demonstrate that post-transformer architectures can
outperform traditional transformers in efficiency, cost, and latency for
narrow code-centric tasks.

------------------------------------------------------------------------

## 🧩 Problem Statement

Modern transformer models are inefficient for narrowly scoped code tasks
due to:

-   Quadratic attention complexity\
-   Dense always-on computation\
-   High inference latency\
-   Expensive training and deployment

However, debugging and simplification tasks are:

-   Local\
-   Structured\
-   Deterministic

Using full dense transformers for such tasks is computationally
excessive.

------------------------------------------------------------------------

## 💡 Proposed Solution

Build a BDH-inspired small language model trained via task-specific
distillation to:

-   Detect syntax & logical bugs\
-   Generate corrected Python code\
-   Simplify verbose implementations\
-   Optionally explain corrections

The model is distilled from larger open-source teacher models to
drastically reduce compute cost.

------------------------------------------------------------------------

## 🏗 Technical Specifications

  Component        Specification
  ---------------- --------------------------------------------------
  Architecture     Decoder-only BDH-inspired model
  Parameters       150--180M
  Context Window   2048 tokens
  Precision        BF16 training
  Inference        8-bit quantization
  Tokenizer        SentencePiece (DeepSeek-Coder tokenizer, frozen)
  Framework        PyTorch / JAX
  Hardware         Single A100 80GB GPU

------------------------------------------------------------------------

## 📂 Dataset Strategy

Target: 15K--20K samples

Distribution:

-   Pure Python: 65%\
-   NumPy: 9%\
-   Pandas: 9%\
-   Torch: 7%\
-   Scikit-learn: 5%\
-   Matplotlib: 3%\
-   Datasets: 2%

------------------------------------------------------------------------

## 📦 Unified Output Format

    ### INSTRUCTION
    <user instruction>

    ### INPUT CODE
    <buggy / complex code or <EMPTY>>

    ### OUTPUT CODE
    <corrected / generated Python code>
    ### END OUTPUT

    ### EXPLANATION
    <optional, ≤ 100 tokens>

------------------------------------------------------------------------

## 💰 Budget

Estimated total: ₹5,000 -- ₹9,000\
Within ₹10,000 constraint.

------------------------------------------------------------------------

## 🎯 Expected Outcome

-   Lightweight debugging model\
-   Reduced inference cost\
-   Clear efficiency vs accuracy tradeoff\
-   Proof-of-concept for BDH viability

------------------------------------------------------------------------

## 🧠 Conclusion

Architecture choice matters.\
Domain-specific efficiency can outperform brute-force scaling.
