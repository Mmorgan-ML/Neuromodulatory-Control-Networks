# Run 1: NCN Architecture (18M Parameters)

**Status:** Complete  
**Date:** Dec 5, 2025 - Dec 7, 2025  
**Device:** Single Nvidia GPU (CUDA)  
**Total Duration:** ~58 Hours

## Scientific Summary
This run represents the alpha test of the Neuromodulatory Control Network (NCN) architecture. The model was trained for **1 epoch** on the **TinyStories** dataset (mapped to binary format) to verify the hypothesis that neuromodulatory gating can improve sample efficiency on narrative data.

The model achieved a final validation perplexity of **4.5184**. This is a significant result for an 18M parameter model trained for only one pass over the data, suggesting the NCN hypernetwork effectively regulates the plasticity of the main transformer backbone.

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
*   **Act Function:** GELU (Standard)
*   **Total Params:** 18.01M

## NCN (Neuromodulatory) Hyperparameters
*   **Role:** Hypernetwork / Meta-controller
*   **Input Dimension:** 256 (Tied to d_model)
*   **Hidden Dimension:** 64
*   **NCN Heads:** 2
*   **Activation Function:** Tanh
*   **Modulation Signals:**
    1.  `gain` (Layer scaling)
    2.  `precision` (Attention sharpening)
    3.  `ffn_gate` (Feedforward throttling)
*   **Parameter Overhead:** 281.30K (1.56% of total)

## Training Configuration
*   **Dataset:** TinyStories (Binary)
*   **Learning Rate:** 6e-4 (Linear Decay with 100 warmup steps)
*   **Batch Size:** 64 (16 per device * 4 gradient accumulation steps)
*   **Optimizer:** AdamW (Weight Decay 0.1)
*   **Precision:** Mixed Precision (AMP) Enabled

## Training Dynamics
The training was stable with no loss spikes. The `grad_clip` of 1.0 was rarely triggered after the warmup phase. The NCN parameters introduced a computational overhead of roughly <2% compared to a vanilla forward pass.

**Log file:** `training.log` (Attached in this directory)
