# Run 1: NCN Architecture (18M Parameters)

**Status:** Complete  
**Date:** Dec 5, 2025 - Dec 7, 2025  
**Device:** Single Nvidia GPU (CUDA)  
**Total Duration:** ~58 Hours

## Scientific Summary
This run serves as the alpha validation of the **Neuromodulatory Control Network (NCN)** architecture on the TinyStories binary dataset. 

Unlike standard Transformer training, this experiment tests the hypothesis that a parallel hypernetwork can **implicitly learn an optimal processing strategy** (Section 2.1 of the paper) by modulating the main network's gain, precision, and gating dynamics. The goal was to observe if the NCN could stabilize without "Entropy Shock" and achieve competitive perplexity through dynamic resource allocation rather than static weight optimization.

## Theoretical Hypotheses Tested
This run specifically targets three biological mechanisms proposed in the NCN paper:
1.  **Thermodynamic Regulation (Exploration vs. Exploitation):** Can the `precision` signal ($\beta$) dynamically regulate the entropy of the attention mechanism, mimicking the signal-to-noise ratio modulation of Norepinephrine?
2.  **Gradient Shielding:** Does the multiplicative `gain` ($g$) allow the model to selectively down-regulate layers during specific contexts, theoretically shielding specialized weights from catastrophic interference (Plasticity-Stability Dilemma)?
3.  **Metabolic Efficiency:** Verifying if **Homeostatic Regularization** ($\mathcal{L}_{reg}$) prevents the control manifold from collapsing into a rigid state or exploding.

## Final Metrics
| Metric | Value |
| :--- | :--- |
| **Final Training Loss** | 1.6243 |
| **Validation Loss** | 1.5082 |
| **Validation PPL** | **4.5184** |
| **Total Steps** | 15,860 |
| **Avg Throughput** | ~2,490 tok/s |

## Transformer Backbone Hyperparameters
*   **Context Window:** 512
*   **Embedding Dim:** 256
*   **Layers:** 6
*   **Heads:** 8
*   **Feedforward Dim:** 1024
*   **Dropout:** 0.1
*   **Act Function:** GELU
*   **Total Params:** 18.01M

## NCN (Neuromodulatory) Hyperparameters
*   **Role:** Meta-controller / Hypernetwork
*   **Input Dimension:** 256 (Tied to d_model)
*   **Hidden Dimension:** 64
*   **NCN Heads:** 2
*   **Activation Function:** Tanh
*   **Modulation Signals:**
    1.  `gain` ($g$): Signal-to-noise / Layer integration rate.
    2.  `precision` ($\beta$): Inverse temperature / Attention entropy control.
    3.  `ffn_gate` ($\gamma$): Metabolic gating of FFN blocks.
*   **Parameter Overhead:** 281.30K (1.56% of total)

## Training Configuration
*   **Dataset:** TinyStories (Binary)
*   **Learning Rate:** 6e-4 (Linear Decay with 100 warmup steps)
*   **Batch Size:** 64 (16 per device * 4 gradient accumulation steps)
*   **Optimizer:** AdamW (Weight Decay 0.1)
*   **Initialization:** Bias Initialization Strategy (Section 4.1.4 of paper) applied to prevent "Metabolic Throttling."

## Training Dynamics & Observations
The log confirms the efficacy of the **Bias Initialization Strategy** described in Section 4.1.4. The model avoided the "Entropy Shock" typical of hypernetworks; the loss curve shows immediate, stable descent from step 0. 

The validation perplexity of **4.51** on a small-scale model (18M) suggests that the NCN is successfully compressing the loss manifold by dynamically altering the effective depth and sharpness of the network per token, rather than treating all tokens with uniform computational intensity.

**Log file:** `training.log` (Attached in this directory)
