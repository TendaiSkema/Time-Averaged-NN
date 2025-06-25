# README

## Dynamic Recurrent Neural Network with Temporal Averaging

### Overview

This project implements a simple yet experimental neural network that simulates **recurrent dynamics at the layer level**. Unlike classical feedforward networks, this network processes each input through **multiple iterations**, where each layer is influenced not only by the current input, but also by its own previous activation state and, optionally, by the next layer.

---

### Core Ideas

* **Layer-level Recurrence:**
  Each layer receives as input not only the raw data, but also its own last activation (from the previous iteration). This enables a **memory effect** and allows the network to "settle" or stabilize over time.

* **Post-Layer Feedback:**
  Optionally, each layer can also take as input the state of the next (higher) layer. This allows feedback and coordination mechanisms between layers.

* **Iterative Dynamics:**
  For each input, the network performs several forward passes ("time steps"), with each layer's activation in each step depending on its own previous state and the state of neighboring layers. The final output is computed as the mean over all time steps.

* **Stochastic Input:**
  The input layer works in a binary and stochastic fashion, allowing the network to handle "fuzzy" or noisy data.

* **Binary Activations:**
  Neurons work with thresholds and fire either 1 or 0, mimicking hard threshold logic.

---

### What is the Goal?

This network is designed to explore how

* **recurrent dynamics** (feedback, memory)
* and **temporal averaging**
  affect learning behavior and pattern recognition.

Questions to investigate:

* Does the network find more stable or robust solutions than pure feedforward models?
* Can it generalize with few examples or on noisy input?
* How does feedback between layers affect learning and stability?

---

### Possible Applications and Experiments

* **Criticality & Homeostasis:**
  Investigate how neural layers regulate their activity and maintain "critical" activity levels.

* **Pattern Completion:**
  Can the network reconstruct the correct pattern from incomplete or noisy images?

* **Reservoir Computing/Echo-State Dynamics:**
  Simulate a reservoir where recurrent dynamics allow for new forms of classification, time-series analysis, or memory.

* **Biologically Inspired Modeling:**
  Test-bed for neuroscience ideas around memory, stability, and emergent states in neural systems.

---

### Usage

1. **Training:**

   * Select a (small) dataset, e.g., MNIST, as training examples.
   * One-hot encode your target labels.
   * Train the network with the provided training loop. The training process is standard, but internally the network computes iteratively over several time steps.
2. **Testing:**

   * Test how robust the network is to new or noisy input.
   * Analyze how layer activations evolve over time.

---

### Example: Training Start

```python
train_model(model, train_samples, train_labels, epochs=100, batch_size=training_size)
```

---

### Note

This network is **not a standard ML approach** but rather an experimental, research-oriented framework to study dynamics and self-regulation in neural systems.

**Strengths:** Robustness, memory, self-organization
**Weaknesses:** Possibly slower learning and less classical performance on large datasets.

---

**Have fun experimenting with recurrent layer dynamics!**

---

Let me know if you want sections for installation, code structure, or usage tips!
