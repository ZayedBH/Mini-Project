# Mini Project: Efficient Code Debugging & Simplification

## Using BDH (Baby Dragon Hatchling) Architecture

------------------------------------------------------------------------

# 1. Project Overview

This project develops a task-specific, lightweight language model for
**code debugging and simplification**, built using a **BDH (Baby Dragon
Hatchling) architecture**.

Rather than relying on large, dense transformer models, we design a
**180M parameter decoder-only BDH model** optimized specifically
for:

-   Structured code transformations
-   Localized bug fixing
-   Deterministic simplification
-   High-efficiency inference

The objective is to demonstrate that **post-transformer architectures**
can outperform transformers in **efficiency, cost, and latency** for
narrow code-centric domains.

------------------------------------------------------------------------

# 2. Problem Statement

Modern transformer models are inefficient for narrow debugging tasks
because of:

-   Quadratic attention complexity
-   Dense always-on computation
-   High inference latency
-   Expensive training and deployment

However, code debugging and simplification tasks are:

-   Local (small region edits)
-   Structured (syntax-constrained)
-   Deterministic (predictable transformations)

Using full-scale transformers for such workloads is computationally
excessive.

------------------------------------------------------------------------

# 3. Proposed Solution

Build a **BDH-inspired decoder-only small language model** trained via
**task-specific distillation** to:

-   Detect syntax bugs
-   Detect logical bugs
-   Generate corrected Python code
-   Simplify complex implementations
-   Optionally explain fixes (≤100 tokens)

Instead of pretraining from scratch, the model learns from larger
teacher LLMs using distillation to significantly reduce compute cost.

------------------------------------------------------------------------

# 4. Architecture Specification (Frozen)

These parameters are fixed prior to training.

  -----------------------------------------------------------------------
  Component                        Specification
  -------------------------------- --------------------------------------
  Architecture                     Decoder-only, BDH-inspired

  Model Size                       180M parameters

  Context Window                   2048 tokens

  Precision                        BF16 training

  Inference                        8-bit (or lower) quantization

  Tokenizer                        SentencePiece BPE (reused from
                                   DeepSeek-Coder-6.7B, frozen)

  Framework                        PyTorch / JAX

  Hardware                         Single A100 80GB GPU
  -----------------------------------------------------------------------

Tokenizer reuse ensures:

-   Code-specific subword preservation
-   Alignment between teacher and student during distillation

------------------------------------------------------------------------

# 5. Dataset Design

## 5.1 Target Dataset Size

~86,000 samples

Each sample contains:

-   Instruction
-   Input code (buggy or complex)
-   Output code (fixed or generated)
-   Optional explanation (≤100 tokens)

------------------------------------------------------------------------

## 5.2 Dataset Ratios (86K Total)

  Category       Samples      Percentage
  -------------- ------------ ------------
  Pure Python    13,000       65%
  NumPy          1,800        9%
  Pandas         1,800        9%
  Torch          1,400        7%
  Scikit-learn   1,000        5%
  Matplotlib     600          3%
  Datasets       400          2%
  **Total**      **~86,000**   **100%**

------------------------------------------------------------------------

# 6. Atomic Intent List (\~52 Intents)

## A. Core Python (12)

### Code Generation (5)

1.  Generate Python function
2.  Generate Python script
3.  Convert pseudocode to Python
4.  Generate loop-based solution
5.  Generate recursive solution

### Bug Fixing (5)

6.  Fix syntax error
7.  Fix undefined variable
8.  Fix incorrect return statement
9.  Fix wrong conditional expression
10. Fix off-by-one loop error

### Simplification (2)

11. Simplify redundant logic
12. Convert verbose loops to Pythonic constructs

------------------------------------------------------------------------

## B. PyTorch (8)

Scope: tensors, basic models, training skeletons
Excluded: distributed training, CUDA internals, advanced APIs

1.  Generate tensor operations
2.  Fix tensor shape mismatch
3.  Fix incorrect dtype
4.  Fix missing `.backward()`
5.  Fix missing optimizer step
6.  Correct loss computation
7.  Fix forward pass bug
8.  Generate simple `nn.Module`

------------------------------------------------------------------------

## C. HuggingFace Datasets (4)

Scope: loading, mapping, filtering

1.  Load dataset
2.  Fix split usage
3.  Apply map correctly
4.  Fix column access bug

------------------------------------------------------------------------

## D. NumPy (9)

1.  Generate NumPy operations
2.  Fix array indexing
3.  Fix shape mismatch
4.  Correct broadcasting error
5.  Fix aggregation misuse
6.  Simplify expressions
7.  Convert loop logic to NumPy
8.  Fix `axis` misuse
9.  Generate vectorized solution

------------------------------------------------------------------------

## E. Pandas (9)

1.  Load DataFrame
2.  Fix column selection
3.  Fix `groupby` misuse
4.  Fix chained indexing
5.  Handle missing values
6.  Fix filtering condition
7.  Simplify transformation pipeline
8.  Generate aggregation code
9.  Fix datatype conversion

------------------------------------------------------------------------

## F. Matplotlib (4)

1.  Generate basic plot
2.  Fix dimension mismatch
3.  Fix missing labels/titles
4.  Fix subplot misuse

------------------------------------------------------------------------

## G. Scikit-learn (6)

1.  Generate training pipeline
2.  Fix train-test split misuse
3.  Fix missing `.fit()`
4.  Fix incorrect `.predict()`
5.  Fix X/y shape mismatch
6.  Generate evaluation metrics

------------------------------------------------------------------------

# 7. Teacher Model

  -----------------------------------------------------------------------
  Role                Model                     Parameters
  ------------------- ------------------------- -------------------------
  Primary Teacher     Qwen/Qwen2.5-Coder-32B    ~32B parameters
  -----------------------------------------------------------------------

This single teacher model is used to generate synthetic training data across all atomic intents through task-specific distillation.

------------------------------------------------------------------------

# 8. Training Configuration

### Distillation Parameters

-   **Temperature**: 0.15
-   **Top-p**: 0.95
-   **Max Output Tokens**: 1024
-   **Learning Rate**: 3e-4
-   **Batch Size**: 16
-   **Epochs**: 3

### Data Generation

-   **Prompt Variations**: 4 per intent
-   **Total Samples Target**: ~86K
-   **Explanation Length**: ≤100 tokens (when included)

------------------------------------------------------------------------

# 9. Data Filtering Rules

Reject any sample if:

-   Output code is malformed or incomplete
-   `ast.parse()` fails
-   Non-whitelisted libraries appear
-   Output identical or trivial variation of input
-   Multiple atomic intents mixed in one sample
-   Code exceeds length limits (token-based)
-   Torch usage outside allowed scope
-   Semantic errors detected in output
-   Duplicate or near-duplicate found

------------------------------------------------------------------------

# 10. Evaluation Metrics

-   Exact bug-fix accuracy
-   AST match rate
-   Compilation / execution success
-   Inference latency
-   FLOPs and parameter efficiency
-   Comparison against small transformer baseline

------------------------------------------------------------------------

# 11. Conclusion

This project demonstrates:

-   Larger models are not universally optimal
-   Architecture choice significantly impacts efficiency
-   Domain-specific optimization outperforms brute-force scaling
-   High-impact ML systems can be built under tight compute budgets
