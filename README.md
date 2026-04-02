# Obesity Risk Classification: Linear vs Neural Network Models

## 📋 Assignment Overview

This assignment explores the implementation and comparison of linear and non-linear models for a real-world multi-class classification problem using the **Obesity Risk Dataset** from the UCI Machine Learning Repository.

### Models Implemented (From Scratch)
- **Softmax Regression** (Baseline Linear Model)
- **Multi-Layer Perceptron (MLP)** (Non-linear Neural Network)

## 🎯 Objectives

- Implement classification algorithms using only NumPy (no high-level ML libraries)
- Understand gradient descent and backpropagation fundamentals
- Compare linear vs non-linear model performance
- Perform hyperparameter tuning for optimal results

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | UCI Machine Learning Repository |
| **Samples** | 2,111 |
| **Features** | 16 (26 after encoding) |
| **Target Classes** | 7 obesity levels |
| **Problem Type** | Multi-class Classification |

### Target Classes
- Insufficient Weight
- Normal Weight
- Overweight Level I & II
- Obesity Type I, II, III

## 🔧 Implementation Details

### Softmax Regression
- Softmax activation for multi-class probabilities
- Categorical cross-entropy loss
- Gradient descent optimization
- Parameter initialization (He initialization)

### Multi-Layer Perceptron (MLP)
- **Architecture**: Input → 64 → 32 → 16 → Output (7 classes)
- **Activation**: ReLU (hidden layers), Softmax (output)
- **Loss**: Categorical cross-entropy
- **Optimization**: Backpropagation with gradient descent

## 📈 Results

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Softmax Regression (Baseline) | 76.36% | 75.33% | 0.31s |
| Softmax Regression (Best) | 77.30% | 76.43% | 0.47s |
| MLP (Baseline) | 89.60% | 89.54% | 1.76s |
| **MLP (Best)** | **92.67%** | **92.67%** | **3.58s** |

### Performance Improvement
- **MLP vs Softmax**: ~**16-17%** improvement in both accuracy and F1-score
- **Softmax tuning**: ~**1%** improvement
- **MLP tuning**: Minimal improvement (baseline was near-optimal)

## 🧪 Hyperparameter Tuning

### Softmax Regression
| Learning Rate | Iterations | Accuracy | F1-Score |
|---------------|------------|----------|----------|
| 0.001 | 1000 | 23.40% | 20.65% |
| 0.01 | 1000 | 63.12% | 59.53% |
| 0.1 | 1500 | **77.07%** | **76.14%** |

### MLP Architectures Tested
- Small: [64, 32]
- Medium: [128, 64]
- Deep: [64, 32, 16] ⭐ (Best)
- Single Hidden: [128]

## 🔬 Key Findings

1. **Non-linear models significantly outperform linear models** for complex pattern recognition
2. **Hyperparameter tuning impact** varies by model complexity
3. **Computational cost** increases with model depth (MLP ~7x slower to train)
4. **Deep architecture** [64, 32, 16] achieved best performance

## 🚀 How to Run

1. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Ensure `ObesityDataset.csv` is in the working directory

3. Run the Jupyter notebook sequentially

## 📁 Repository Structure

```
├── ObesityDataset.csv          # Dataset file
├── assignment1.ipynb           # Main notebook with all implementations
├── assignment1.html            # HTML File with output
└── README.md                   # This file
```

## 🛠️ Technologies Used

- **Python 3.11+**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **Scikit-learn** - Data preprocessing & metrics

## 📚 References

- [UCI Obesity Risk Dataset](https://archive.ics.uci.edu/ml/datasets/Estimation+of+Obesity+Levels+Based+on+Eating+Habits+and+Physical+Condition)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*

## 👨‍🎓 Author

**Name:** Gagan Rajput  
**Course:** M.Tech AI & ML, BITS Pilani

---

*Note: All models were implemented from scratch without using high-level ML libraries (TensorFlow, PyTorch, scikit-learn's classifiers).*
