# Dynamic Recurrent Spiking Neural Network with Temporal Averaging

## Overview

This project explores an experimental neural network model combining **stochastic spiking neurons**, **temporal averaging**, and **layer-level recurrence**. Unlike traditional feedforward models, this architecture is designed to simulate biologically inspired dynamics while remaining compatible with gradient-based learning techniques such as backpropagation.

We show that, under repeated stochastic input, the time-averaged output of a spiking neuron converges to a **ReLU-like function**, making it differentiable in expectation. This not only enables standard training techniques without surrogate gradients but also suggests that classical dense ReLU networks are effectively **time-averaged approximations of spiking networks**.

---

## Core Concepts

### 1. Spiking Neurons with Temporal Averaging

Each neuron fires stochastically with a probability equal to its input $x \in [0, 1]$. Its output is binary (0 or 1), determined by a thresholded weighted input. When evaluated over multiple iterations, the average firing rate converges to a **clipped linear function**:

$\bar{y}(x) = \begin{cases} x & \text{if } w \geq \theta \\ 0 & \text{otherwise} \end{cases}$

This is equivalent to a shifted and scaled ReLU. Hence, the network is **backpropagation-compatible without surrogate gradient tricks**.

### 2. Layer-Level Recurrence

Each layer receives input not just from the previous layer, but also from its own previous activation state (recurrent dynamics) and optionally the next layer (feedback). This allows the network to develop temporal memory and stabilize patterns over multiple iterations.

### 3. No Looping Problem

Because the network operates over **discrete time steps** and aggregates output only after a fixed number of iterations, self-connections and feedback within a layer or across layers do **not create cyclic computation issues**. This temporal unfolding resolves the loop problem inherently.

### 4. Dense Networks as Averaged SNNs

By reversing the usual approximation logic, we propose:

> A ReLU neuron in a standard dense network can be understood as the **expected output of a spiking neuron** under repeated stochastic input.

This provides a fresh theoretical link between deep learning and spike-based computation.

---

## Mathematical Justification

Let $x \in [0, 1]$ be a real-valued input interpreted as spike probability.
At each iteration $t$, the input neuron fires:
$s_t \sim \text{Bernoulli}(x)$

A post-synaptic neuron computes:
$a_t = w \cdot s_t \quad \text{and} \quad y_t = \begin{cases} 1 & \text{if } a_t \geq \theta \\ 0 & \text{otherwise} \end{cases}$

Averaging over $T$ steps:
$\bar{y} = \frac{1}{T} \sum_{t=1}^T y_t \to \mathbb{E}[y_t]$

Expected output becomes:
$\mathbb{E}[y_t] = \begin{cases} x & \text{if } w \geq \theta \\ 0 & \text{otherwise} \end{cases}$

This is equivalent to:
$\mathbb{E}[\bar{y}(x)] = \text{ReLU}(w \cdot x - \theta) \quad \text{(piecewise linear)}$

This justifies why time-averaged stochastic spiking networks are compatible with backpropagation.

---

## Goals of This Project

* Explore **recurrent dynamics** and **temporal memory** in deep learning models.
* Validate that **spiking behavior over time approximates ReLU**, supporting gradient-based training.
* Investigate how **feedback between layers** and **self-connections** contribute to stability and computation.

---

## Experiments and Applications

* **Criticality & Homeostasis:** How does the network self-regulate over time?
* **Pattern Completion:** Can it recover missing or noisy inputs?
* **Temporal Echo-State Dynamics:** Use internal recurrence to encode memory.
* **Neuroscience Modeling:** Test biologically inspired connectivity and firing behaviors.

---

## Training and Usage

1. **Train:**

   * Use a dataset like MNIST
   * One-hot encode targets
   * Apply standard training, with internal spiking dynamics computed over multiple time steps

```python
train_model(model, train_samples, train_labels, epochs=100, batch_size=training_size)
```

2. **Test:**

   * Evaluate robustness to noisy or incomplete inputs
   * Analyze how activation patterns evolve over time

---

## Why This Matters

* **Bridges biological realism and ML practicality**
* **Eliminates the need for surrogate gradients**
* **Provides deeper insight into the origin of common activations like ReLU**

---
