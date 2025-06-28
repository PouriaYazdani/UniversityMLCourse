# Machine Learning Course &nbsp;–&nbsp;  Assignments

*A curated collection of hands-on notebooks (except the final project) from the university **machine_learning** course I intended in university.  
Theory-only (“paper”) assignments are also stored in the repo, but this README focuses on the four practical assigments each including several labs.*

---

## 📑 Table of Contents

1. [Assignment&nbsp;1 – K-NN, Decision Trees & Random Forests](#assignment-1--k-nn-decision-trees--random-forests)  
2. [Assignment&nbsp;2 – Linear & Logistic Regression](#assignment-2--linear--logistic-regression)  
3. [Assignment&nbsp;3 – PyTorch Basics, MLP from Scratch & MNIST Classification](#assignment-3--pytorch-basics-mlp-from-scratch--mnist-classification)  
4. [Assignment&nbsp;4 – Custom CNN with Residual Block & MobileNet architecture](#assignment-4--custom-cnn-with-residual-block--mobilenet-architecture)  

---

## Assignment 1 – K-NN, Decision Trees & Random Forests

| Folder | Notebook(s) & report | Topic |
|--------|----------------------|-------|
| `1/` | `investigation.ipynb`, `split_normalize_undersample.ipynb`, `split_undersample_normalize.ipynb`, `decision_tree_random_forest.ipynb`, `ML_practical_Q2 report.pdf` | Distance-based and tree-based classifiers |

### What this assignment covers
* **Data prep & imbalance handling**  
  * One-hot encoding, z-score scaling, and several random undersampling ratios to counter a ≈ 5 % positive class.
* **K-Nearest Neighbors**  
  * Experimented with multiple *k* values and distance metrics; grid-searched undersampling + hyper-parameters in tandem.
* **Decision Trees & Random Forests**  
  * Built depth-controlled trees as interpretable baselines.  
  * Extended to an ensemble to reduce variance and compare against single-tree performance.

Main challenges addressed: severe class imbalance, heterogeneous feature types, and selecting hyper-parameters that balance recall for the minority class while controlling model complexity.

## Assignment 2 – Linear & Logistic Regression

| Folder | Notebook(s) | Topic |
|--------|-------------|-------|
| `2/` | `linear_regression.ipynb`, `logistic_regression.ipynb` | Regression & binary-classification fundamentals |

### What this assignment covers
* **Linear regression**  
  * Built univariate and multivariate models from scratch and with scikit-learn.  
  * Compared closed-form solution vs. gradient descent; experimented with learning-rate schedules and feature scaling.  
  * Tackled multicollinearity by introducing \(L_2\) regularisation and principal-component preprocessing.

* **Logistic regression**  
  * Implemented binary classifier for a moderately imbalanced dataset.  
  * Explored polynomial feature expansion to capture non-linear decision boundaries.  
  * Applied \(L_1\) and \(L_2\) penalties to control overfitting and analysed decision boundary shifts.

Main challenges addressed: handling correlated predictors, choosing appropriate regularisation strength, and stabilising gradient descent across differently scaled features.

## Assignment 3 – PyTorch Basics, MLP from Scratch & MNIST Classification

| Folder | Notebook(s) | Topic |
|--------|-------------|-------|
| `3/` | `pytorch_basic.py`,`Pytorch_Basics.ipynb`, `MLP_from_scratch.ipynb`, `3-MNIST_Classification.ipynb` | Deep-learning foundations |

### What this assignment covers
* **PyTorch fundamentals**  
  * Hands-on with tensors, autograd, and GPU transfers; verified gradient calculations with a custom numeric checker.
* **MLP built from scratch**  
  * Implemented forward and back-prop manually—layers, activations, weight initialisation, mini-batch SGD, plus optional dropout.  
  * Benchmarked against PyTorch’s `nn.Module` to confirm parity.
* **MNIST digit classification**  
  * Trained a small convolution-free network on 28×28 images; experimented with depth, hidden units, and regularisation.  
  * Applied early stopping and learning-rate scheduling to stabilise training.

Main challenges addressed: writing back-prop without high-level helpers, managing GPU/CPU tensor placement, and preventing overfitting on a small image dataset.

## Assignment 4 – Custom CNN with Residual Block & MobileNet architucture

| Folder | Notebook(s) | Topic |
|--------|-------------|-------|
| `4/` | `cnn_residual_example.ipynb`, `mobile_net.ipynb` | Modern convolutional architectures |

### What this assignment covers
* **From-scratch CNN with residual block**  
  * Built a small image classifier that inserts a custom skip-connection module between two conv layers—mirroring the ResNet idea without pre-built helpers.  
  * Verified gradient flow improvements by tracking training curves against a plain CNN.
* **MobileNet vs. standard CNN**
  * Implemented a lightweight custom **MobileNet-style CNN** from scratch — no use of pretrained models or torchvision helpers.
  * The architecture was constructed using repeated **depthwise separable convolution blocks**, each consisting of:
    * A **depthwise convolution** (groups = in_channels)
    * Followed by a **pointwise convolution** (1x1 kernel)
    * Each followed by **BatchNorm2d** and **ReLU activation**
  * The custom standard CNN used a traditional block structure:
    * `Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d`
    * With increasing channel sizes and a final linear classifier.
  * Compared both models on the same dataset using:
    * **Training/validation accuracy curves**
    * **Model parameter counts**
    * **Inference time** measured via forward-pass timing
    * **Checkpoint sizes** to assess memory/storage efficiency
  * Key implementation features:
    * Manual creation of MobileNet blocks with custom class definitions.
    * Clear modular breakdown of the architecture.
    * Use of **BatchNorm2d** in both CNN variants to ensure stable gradient flow.

Main challenges addressed: implementing residual logic manually, manual implementation of depthwise separable conv blocks, understanding channel grouping mechanics, training lightweight CNNs from scratch, and conducting fair architectural comparisons on accuracy, speed, and size.
