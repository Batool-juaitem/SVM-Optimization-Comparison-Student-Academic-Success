# SVM Optimization Comparison: Student Academic Success Prediction

A multi-class linear **Support Vector Machine implemented from scratch in NumPy**, trained with five different optimization algorithms and compared on the same data, the same split and the same objective function.

**Task:** predict whether a student will **Dropout**, stay **Enrolled**, or **Graduate**.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Model](#model)
3. [Methodology](#methodology)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Results](#results)
6. [Discussion](#discussion)
7. [Repository Structure](#repository-structure)
8. [How to Run](#how-to-run)
9. [Limitations](#limitations)

---

## Dataset

**Predict Students' Dropout and Academic Success**, UCI Machine Learning Repository (id 697), license CC BY 4.0.
Link: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
Please cite the dataset as recommended on the UCI page.

| Property | Value |
|---|---|
| Samples | 4,424 |
| Original features | 36 |
| Classes | 3 (Dropout, Enrolled, Graduate) |
| Missing values | none |
| Class balance | imbalanced (see the EDA section of each notebook) |

After preprocessing the design matrix has **237 features** (including the bias column).

| Split | Samples |
|---|---|
| Training (80%) | 3,539 |
| Test (20%) | 885 |

The split is **stratified** with `random_state = 42` and is identical for all five optimizers.

---

## Model

A linear multi-class SVM with the **Crammer-Singer multiclass hinge loss** and L2 regularization. The loss, the subgradient and all optimizer updates are implemented manually with NumPy (no scikit-learn SVM class is used for training).

For a weight matrix `W` with one column `w_k` per class:

```
J(W) = (1/n) * sum_i max(0, 1 + max_{j != y_i} w_j.x_i - w_{y_i}.x_i)  +  (lambda / 2) * ||W||^2
```

* A sample is penalised when the score of the correct class does not beat the score of the **most violating wrong class** by a margin of 1.
* The bias is a constant feature and is **not** regularised.
* Prediction: `argmax_k  w_k . x`.
* Each notebook verifies the implementation before training: comparison with a brute-force loop, the `W = 0` sanity check (loss must equal 1), and a numerical gradient check.

---

## Methodology

| Step | Choice |
|---|---|
| Nominal categorical features (9) | One-Hot Encoding |
| Numerical features | `StandardScaler` |
| Binary flags | kept as 0/1 |
| Data leakage | preprocessing is fitted on the **training data only** |
| Split | stratified 80/20, `random_state = 42` |
| Objective | same Crammer-Singer loss for all optimizers |
| Learning rate | selected per optimizer from a logarithmic grid, as the value giving the **lowest final training loss** (the test set is never used for tuning) |
| Evaluation | accuracy, macro precision / recall / F1, confusion matrix, classification report, final training loss |

Macro averaging is used because the classes are imbalanced.

### Shared experimental setup

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Epochs | 100 |
| Regularization (lambda) | 0.001 |
| Momentum (GD, SGD, Mini-Batch, RMSProp) | 0.9 |
| SGD batch size | 1 |
| Mini-Batch / RMSProp / Adam batch size | 64 |
| RMSProp rho | 0.9 |
| Adam beta1 / beta2 | 0.9 / 0.999 |

---

## Optimization Algorithms

| # | Optimizer | Momentum |
|---|---|---|
| 1 | Gradient Descent (full batch) | external, 0.9 |
| 2 | Stochastic Gradient Descent | external, 0.9 |
| 3 | Mini-Batch Gradient Descent | external, 0.9 |
| 4 | RMSProp | external, 0.9 |
| 5 | Adam | **built-in** (first-moment estimate, beta1); no external momentum is added |

---

## Results

Test-set performance (885 samples). Precision, Recall and F1 are **macro-averaged**. Best value per column in bold.

| Optimizer | Learning Rate | Accuracy | Precision | Recall | F1-Score | Final Training Loss |
|---|---|---|---|---|---|---|
| GD + Momentum | 0.3 | 0.7559 | 0.6930 | 0.6465 | 0.6482 | 0.4546 |
| SGD + Momentum | 0.0001 | 0.7582 | 0.6938 | 0.6537 | 0.6606 | 0.4597 |
| Mini-Batch GD + Momentum | 0.01 | **0.7695** | **0.7109** | **0.6659** | **0.6724** | 0.4584 |
| RMSProp + Momentum | 0.0001 | 0.7537 | 0.6873 | 0.6455 | 0.6480 | 0.4565 |
| Adam | 0.001 | 0.7571 | 0.6936 | 0.6486 | 0.6512 | **0.4524** |

Additional analyses (training-loss curves, convergence speed, loss against cumulative update time, per-class metrics and confusion matrices for all five optimizers) are in `06-comparison.ipynb`.

---

## Discussion

* **All five optimizers reach very similar results.** They minimise the same convex objective, so similar final loss and accuracy are expected: the final training losses lie within about 0.007 of each other (0.4524 to 0.4597) and the accuracies within about 1.6 percentage points (0.7537 to 0.7695), which corresponds to roughly 14 of the 885 test samples.
* **Mini-Batch GD with momentum scored highest** on accuracy, macro precision, macro recall and macro F1 in this experiment.
* **Adam reached the lowest final training loss**, but this did not translate into the best test performance. A lower training loss does not automatically mean better generalisation.
* Because the differences are small and come from **one split and one random seed**, they should not be read as a statistically established ranking of the optimizers.

---

## Repository Structure

```
.
├── 01-gd.ipynb                # Gradient Descent + Momentum
├── 02-sgd.ipynb               # Stochastic Gradient Descent + Momentum
├── 03-mini-batch-gd.ipynb     # Mini-Batch Gradient Descent + Momentum
├── 04-rmsprop.ipynb           # RMSProp + Momentum
├── 05-adam.ipynb              # Adam
├── 06-comparison.ipynb        # Comparison of the five optimizers
├── data.csv                   # UCI dataset (CC BY 4.0)
└── README.md
```

Each optimizer notebook follows the same structure: introduction, imports, dataset loading, EDA, preprocessing, train/test split, SVM implementation, optimizer implementation, training, loss curve, evaluation, confusion matrix, classification report, final results. Each one saves its results to `results/<name>.json`, which `06-comparison.ipynb` reads (it also checks that all runs used the same split, seed and epochs).

---

## How to Run

**Kaggle**

1. Import a notebook (*File > Import notebook*).
2. Provide the data: either turn **Internet on** in the notebook settings, or attach a dataset that contains `data.csv` with *Add Input*.
3. Run All. Repeat for the five optimizer notebooks, then run `06-comparison.ipynb` with the five `results/*.json` files available.

**Local**

```bash
pip install numpy pandas scikit-learn matplotlib ucimlrepo jupyterlab
jupyter lab
```

Run `01` to `05`, then `06`. Without internet access, keep `data.csv` next to the notebooks.

---

## Limitations

* One stratified split and one random seed: no confidence intervals. Repeating the experiment with several seeds or cross-validation would be needed to compare optimizers statistically.
* Only the learning rate is tuned; momentum, batch size, lambda and the number of epochs are fixed in advance and are the same for all optimizers.
* The objective is not re-weighted for class imbalance, so the minority class is the most likely to be under-predicted.
* The model is linear; kernel SVMs or other model families could perform differently.
