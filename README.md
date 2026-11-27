# SVM Optimization Comparison – Student Academic Success Prediction

## Project Overview

This project implements and compares five optimization algorithms for a Support Vector Machine (SVM) classifier using the **Predict Students' Dropout and Academic Success** dataset.

The main goal is to evaluate how different optimization algorithms with momentum affect the model's classification performance, training loss, and convergence behavior.

The project compares:

1. Gradient Descent (GD) with Momentum
2. Stochastic Gradient Descent (SGD) with Momentum
3. Mini-Batch Gradient Descent with Momentum
4. RMSProp with Momentum
5. Adam Optimizer

## Dataset

**Dataset Name:** Predict Students' Dropout and Academic Success

**Source:** UCI Machine Learning Repository

**Dataset Link:** [UCI Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

The dataset contains student academic and demographic information. The target variable has three classes:

- Dropout
- Enrolled
- Graduate

### Dataset Information

| Description | Value |
|---|---:|
| Training samples | 3,539 |
| Testing samples | 885 |
| Number of features | 237 |
| Number of classes | 3 |
| Random seed | 42 |

## Data Preprocessing

The dataset was prepared before training the SVM models.

The preprocessing workflow includes:

- Preparing the input features and target variable.
- Encoding categorical features.
- Scaling numerical features.
- Splitting the data into training and testing sets.
- Applying consistent preprocessing and data splitting across all five optimizers.

## Optimization Algorithms

The project implements five optimization algorithms to train and compare SVM models:

1. **Gradient Descent (GD) with Momentum**
2. **Stochastic Gradient Descent (SGD) with Momentum**
3. **Mini-Batch Gradient Descent with Momentum**
4. **RMSProp with Momentum**
5. **Adam Optimizer**

## Experimental Setup

All five optimizers were evaluated under consistent experimental conditions.

| Parameter | Value |
|---|---:|
| Random Seed | 42 |
| Number of Epochs | 100 |
| Momentum | 0.9 |
| Regularization Parameter (λ) | 0.001 |
| SGD Batch Size | 1 |
| Mini-Batch Size | 64 |
| RMSProp Batch Size | 64 |
| Adam Batch Size | 64 |

The learning rate was selected separately for each optimizer.

## Results and Performance Comparison

The following table presents the test-set performance of the five optimizers.

| Optimizer | Learning Rate | Accuracy | Precision | Recall | F1-Score | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| GD + Momentum | 0.3 | 0.7559 | 0.6930 | 0.6465 | 0.6482 | 0.4546 |
| SGD + Momentum | 0.0001 | 0.7582 | 0.6938 | 0.6537 | 0.6606 | 0.4597 |
| Mini-Batch GD + Momentum | 0.01 | **0.7695** | **0.7109** | **0.6659** | **0.6724** | 0.4584 |
| RMSProp + Momentum | 0.0001 | 0.7537 | 0.6873 | 0.6455 | 0.6480 | 0.4565 |
| Adam | 0.001 | 0.7571 | 0.6936 | 0.6486 | 0.6512 | **0.4524** |

*Precision, Recall, and F1-Score are macro-averaged metrics.*

## Evaluation and Visualization

The project includes several evaluation and visualization techniques:

- Accuracy, Precision, Recall, and F1-Score comparison.
- Training loss comparison across epochs.
- Convergence analysis.
- Training loss versus cumulative update time.
- Per-class Precision, Recall, and F1-Score analysis.
- Confusion matrices for all five optimizers.

These visualizations help evaluate not only the final performance but also how each optimizer learns and converges during training.

## Conclusion

Based on the experimental results, **Mini-Batch Gradient Descent with Momentum achieved the best overall classification performance** among the five optimizers.

It achieved the highest Accuracy (0.7695), Precision (0.7109), Recall (0.6659), and Macro F1-Score (0.6724).

Adam achieved the lowest final training loss (0.4524), but this did not translate into the highest test-set classification performance.

Overall, the results demonstrate that the choice of optimization algorithm and learning rate can influence SVM classification performance and convergence behavior.

## Project Structure

The project is organized into separate notebooks for the five optimization algorithms and a notebook for comparing their results.

- `gd.ipynb`
- `sgd.ipynb`
- `mini-batch-gd.ipynb`
- `rmsprop.ipynb`
- `adam.ipynb`
- `comparison.ipynb`

## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Kaggle Notebooks
