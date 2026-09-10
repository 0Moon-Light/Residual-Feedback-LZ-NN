# Residual-Feedback LZ+NN 

[![Author: 0Moon-Light](https://img.shields.io/badge/Author-0Moon--Light-purple.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Topic: Neural Compression](https://img.shields.io/badge/Domain-Neural%20Compression-green.svg)]()

> **Residual Modeling of Neural Probabilities Conditioned on LZ Machine States & KL-Budget Expert Arbitration.**

---

## 📌 Concept & Intuition

Currently, deterministic LZ-style algorithms (like LZ77/LZSS) are used in neural data compression as passive context providers. However, LZ divides data streams into two fundamental operational regimes:
1. **Match (Exact pattern copying)**
2. **Literal (Novel byte emission)**

Attempting to model both regimes using a single monolithic neural network or context mixer is suboptimal. 

**0Moon-Light C2** offloads the deterministic copying mechanism from the neural model to LZ. Instead of estimating absolute probabilities $P(x_t \mid \text{context})$, the network models **residuals conditional on the LZ state machine**.

---

## 🧮 Formal Mathematical Architecture

### 1. LZ State Space ($S_t$)
At byte index $t$, the LZ machine state is defined as:
$$S_t = (\text{is\_match}, \text{match\_len}, \text{match\_dist}, \text{literal\_run})$$

### 2. Dual-Head Conditional Prediction
Instead of learning a single monolithic distribution, **0Moon-Light C2** evaluates conditional probability heads:

* **Probability of Match Regime:**
  $$P(\text{is\_match}_t \mid S_{t-1}, \text{context})$$

* **Literal Head (Conditioned on Unmatched Suffix):**
  $$P(x_t \mid S_{t-1}, \text{unmatched\_suffix}, \text{context}) \quad \text{if } \text{is\_match}_t = 0$$

* **Match Head (Conditioned on Deterministic Copy History):**
  $$P(\text{next\_byte\_after\_match} \mid \text{history}[\text{pos} + \text{len}]) \quad \text{if } \text{is\_match}_t = 1$$

---

## ⚖️ Integration with KL-Budget Expert Arbitrator

To blend the outputs of the **Match Head**, **Literal Head**, and **Gated Context Head**, we deploy the **KL-Budget Expert Arbitrator**:

$$W_i(c) = \frac{\exp\left(-\alpha L_i(c) - \beta V_i(c) + \gamma A_i(c)\right)}{\sum_j \exp\left(-\alpha L_j(c) - \beta V_j(c) + \gamma A_j(c)\right)}$$

Where:
* $L_i(c)$: Expected Cross-Entropy Loss of Head $i$
* $V_i(c)$: Loss Variance (Penalizes unstable heads)
* $A_i(c) = -D_{\text{KL}}(p_i \parallel p_{\text{consensus}})$: Consensus agreement penalizing overconfident outliers

---

## 📊 Benchmark & Performance Visualization

Here is the empirical benchmark visualization generated directly by [`c2_residual_lz_nn.py`](file:///home/moonlight/Pulpit/PHANTOMAI/c2_residual_lz_nn.py):

![0Moon-Light C2 Benchmark](./0moon_light_c2_benchmark.png)

### Why 0Moon-Light C2 Outperforms Standard Architectures:
1. **Entropy Reduction**: Offloading long-range copying (e.g. 500-byte matches) to LZ drastically reduces the target entropy for the neural network.
2. **Specialized Literal Learning**: The neural network focuses purely on predicting novel, non-repeating bytes (`unmatched_suffix`).

---

## 🚀 Quickstart

### Project Code Files
- Main Simulation & Plotting Script: [`c2_residual_lz_nn.py`](file:///home/moonlight/Pulpit/PHANTOMAI/c2_residual_lz_nn.py)
- Math & Arbitrator Verification Script: [`kl_budget_arbitrator.py`](file:///home/moonlight/Pulpit/PHANTOMAI/kl_budget_arbitrator.py)

### Running the Benchmark
```bash
python c2_residual_lz_nn.py
```

---

## 📜 License

Distributed under the **MIT License**. Author: **0Moon-Light**.
