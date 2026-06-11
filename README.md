# How AI Learns — Linear Regression from Scratch

A minimal, educational Python tutorial that shows **how machine learning models train** without using any AI libraries. You will see exactly how weights (`w`, `b`) are updated step by step using **gradient descent**.

## What you will learn


| Concept              | What it means                                                  |
| -------------------- | -------------------------------------------------------------- |
| **Model**            | A function that maps inputs to outputs: `y_pred = w * x + b`   |
| **Weights**          | Learnable numbers (`w`, `b`) the model adjusts during training |
| **Forward pass**     | Compute a prediction from the current weights                  |
| **Loss**             | A number that measures how wrong the predictions are (MSE)     |
| **Gradients**        | How much each weight contributes to the loss                   |
| **Gradient descent** | Update weights to make the loss smaller                        |


This is the same core loop used inside neural networks — just with the simplest possible model.

## Requirements

- Python 3.10 or newer (uses built-in `random` only — no pip install needed)

## Quick start

```bash
python main.py
```

## Example output

```
============================================================
  How AI Learns: Training a Linear Model from Scratch
============================================================

[1] Initial random weights
    w = 0.7175
    b = 0.4891
    loss = 15.8729

[2] Training for 100 steps (learning_rate=0.03)
  step   0 | w=1.2341  b=0.8123  loss=8.4521
  ...
  step  99 | w=2.0092  b=0.9729  loss=0.000123

[3] Updated weights
    w = 2.0092  (target ~ 2.0)
    b = 0.9729  (target ~ 1.0)
    loss = 0.000123

[4] Predictions vs. ground truth
    x=1  y_true=3  y_pred=2.9821
    x=2  y_true=5  y_pred=4.9913
    ...
```

The dataset follows `y = 2*x + 1`, so the trained weights should converge near `w ~ 2` and `b ~ 1`.

## How training works

### 1. Forward pass

For each input `x`, the model predicts:

```
y_pred = w * x + b
```

### 2. Loss (Mean Squared Error)

```
error  = y_pred - y_true
loss   = (1/n) * Σ error²
```

### 3. Gradients (manual derivatives)

For each data point:

```
dL/dw = 2 * error * x
dL/db = 2 * error
```

Gradients are averaged over the full dataset each step.

### 4. Weight update (gradient descent)

```
w = w - learning_rate * dL/dw
b = b - learning_rate * dL/db
```

Repeat for many steps until the loss is small.

## Training flow

```
┌─────────────┐
│  Dataset    │  x, y_true
└──────┬──────┘
       ▼
┌─────────────┐
│ Forward     │  y_pred = w*x + b
└──────┬──────┘
       ▼
┌─────────────┐
│ Loss (MSE)  │  how wrong are we?
└──────┬──────┘
       ▼
┌─────────────┐
│ Gradients   │  dL/dw, dL/db
└──────┬──────┘
       ▼
┌─────────────┐
│ Update w,b  │  move weights to reduce loss
└──────┬──────┘
       │
       └──► repeat for N steps
```

## Experiments to try

1. **Change `LEARNING_RATE`** — try `0.001` (slow) vs `0.1` (fast, may overshoot)
2. **Change `STEPS`** — more steps usually means lower loss
3. **Edit the dataset** — change `X` and `Y_TRUE` to your own values
4. **Fix initial weights** — set `w = 0` and `b = 0` instead of random to see slower learning

## Project structure

```
.
├── main.py      # Tutorial script with the full training loop
└── README.md    # This file
```

## License

Free to use for learning and teaching.