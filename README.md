# Symbolic Regression

This repository contains a symbolic regression project that explores two distinct methods for discovering mathematical expressions from data: an **EQL-Based** approach and a **Seq2Seq-Based** approach using Transformers. The goal is to provide a comprehensive comparison of traditional and modern AI-driven techniques for learning symbolic expressions.

---

## 1. EQL-Based Symbolic Regression

This approach is based on the **Equation Learner (EQL)**, a neural network designed for interpretable symbolic regression. Instead of using standard activation functions like ReLU, the network incorporates mathematical operations such as `sin(x)`, `x^2`, and `x*y` directly into its layers. This architecture encourages the discovery of simple, human-readable expressions.

**Key Features:**
- **Interpretable Models:** The network weights directly correspond to coefficients in a mathematical expression.
- **Regularization:** The implementation includes an L1 regularization term to penalize complexity, leading to more concise expressions.
- **Trial-Based Training:** The model is trained over multiple trials, and the best-performing model (with the lowest test loss) is selected.

---

## 2. Seq2Seq-Based Symbolic Regression with Transformers

This method frames symbolic regression as a **sequence-to-sequence translation problem**. A Transformer model is trained to translate a sequence of numerical data points into a corresponding mathematical expression represented as a sequence of tokens. This approach leverages the powerful self-attention mechanism to identify complex patterns and dependencies in the data.

**Key Features:**
- **Transformer Architecture:** The model consists of an encoder that processes the numerical data and a decoder that generates the symbolic expression.
- **Beam Search Decoding:** When generating an expression, the model uses beam search to explore multiple potential sequences and select the most likely one, improving the quality of the final result.
- **Parameter Optimization:** A constant (`C`) in the predicted expression is optimized using `scipy.optimize` to achieve the best fit for the given dataset.

---

## Project Structure

The code is contained within a single Python notebook. It is structured into several sections:

- **Introduction:** A conceptual overview of the two symbolic regression methods.
- **Datset Generation:** Functions to generate synthetic datasets for evaluation, including one with a known formula and another with a hidden formula.
- **EQL Section:**
    - **Base Functions:** Classes for mathematical operations used as activation functions.
    - **Model Architecture:** The `SymbolicLayer` and `SymbolicNet` classes define the EQL network.
    - **Training and Testing:** Functions to train the EQL model, evaluate its performance, and extract the final expression.
    - **L1 Regularization:** An optional bonus section that demonstrates how L1 regularization can be used to simplify expressions.
- **Seq2Seq Section:**
    - **Data Generation:** Code to generate a large number of expressions and corresponding datasets for training the Transformer model.
    - **Transformer Architecture:** The `TransformerModel` and related classes (`TokenEmbeddings`, `MultiHeadAttention`, etc.) that define the Seq2Seq model.
    - **Training and Evaluation:** Functions to train the Transformer model and evaluate it on new datasets using beam search.
    - **Parameter Optimization:** Code to fit the predicted expression to the data by optimizing a constant term.

---

## Getting Started

### Prerequisites

- Python 3.x
- `numpy`
- `sympy`
- `torch`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `scipy`

You can install these dependencies using pip:
```
pip install numpy sympy torch pandas matplotlib scikit-learn tqdm scipy
```
### Running the Notebook

To run the project, simply execute the `symbolic_regression.ipynb` notebook in a Jupyter-compatible environment like JupyterLab or Google Colab. The notebook is self-contained and will walk you through the entire process, from data generation to model training and evaluation for both methods.

