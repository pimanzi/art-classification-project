# 🎨 Art Movement Classification Project

## Problem Statement
This project tackles the challenge of classifying artworks into 13 different art movements using image data. While classical machine learning methods like Random Forest provide a starting point, they often fall short on complex image data. Transfer learning with deep neural networks like ResNet50 offers a more powerful alternative, and this project compares their performances while exploring the impact of hyperparameter tuning.

## Project Motif
The core aim is to understand how different modeling approaches and their hyperparameters affect classification accuracy, precision, recall, and F1 score in artwork categorization. We want to see how fine-tuning ResNet50’s parameters or using classical methods with dimensionality reduction (PCA) impacts results.

---

## Dataset
We used the **WikiArt dataset**, which contains a rich collection of paintings labeled by their respective art movements. This dataset spans 13 categories including Primitivism, Romanticism, Baroque, Renaissance, and more, providing a diverse basis for training and evaluation.

---

## Model Performance Comparison

### ResNet50 Neural Network Results

| Training Instance   | Optimizer | Regularizer | Epochs | Early Stopping | Layers                  | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|--------------------|-----------|-------------|--------|----------------|-------------------------|---------------|----------|----------|--------|-----------|
| Baseline (No tuning) | Adam      | None        | 15     | No             | 177 (frozen) + 2        | Default       | 0.4790   | 0.4752   | 0.4790 | 0.4818    |
| Model 2 (Tuned)      | Adam      | L2          | 15     | Yes            | 177 (frozen) + 2        | 0.0001        | 0.4441   | 0.4361   | 0.4441 | 0.4404    |
| Model 3 (Tuned)      | RMSprop   | L1          | 15     | Yes            | 177 (frozen) + 2        | 0.0001        | 0.4646   | 0.4554   | 0.4646 | 0.4612    |

---

### Random Forest with PCA Results

**Best Hyperparameters:**

```python
{
    'clf__max_depth': 20,
    'clf__max_features': 'sqrt',
    'clf__min_samples_split': 5,
    'clf__n_estimators': 137,
    'pca__n_components': 100
}
```
| Model Type    | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------- | --------- | ------ | -------- |
| Random Forest | 0.2380   | 0.2405    | 0.2380 | 0.2315   |


## Insights and Discussion

- **Random Forest with PCA** provided a baseline but struggled to capture the complexity in artwork images. The hyperparameters helped manage model size and prevent overfitting, but precision, recall, and F1 remained low due to limited feature extraction capability.

- The **ResNet50 baseline model** excelled without tuning, thanks to pretrained layers that efficiently extract image features. This led to balanced accuracy, precision, recall, and F1 scores around 47-48%, confirming transfer learning’s strength.

- Adding **L2 regularization and early stopping** (Model 2) helped prevent overfitting but slightly lowered overall performance metrics. The model became more conservative, which stabilized training but reduced precision and recall somewhat.

- Switching to the **RMSprop optimizer with L1 regularization** (Model 3) encouraged sparsity and feature focus. This tuning improved precision and recall compared to Model 2 but still didn’t beat the baseline, showing optimizer and regularization choices impact the balance of false positives and negatives.

Overall, hyperparameter tuning influenced the trade-offs between accuracy, precision, recall, and F1 score — highlighting the importance of regularization and optimizer choice for reliable generalization in image classification.

---

## 📺 Demo Video

You can watch a detailed walkthrough of this project here:  
[Insert your YouTube video link here]

---

## Future Improvements

- Unfreeze and fine-tune deeper layers of ResNet50 to better adapt to artwork-specific features.  
- Use more sophisticated data augmentation techniques to enrich training diversity.  
- Explore advanced architectures such as EfficientNet or Vision Transformers.  
- Implement learning rate schedules and train longer for improved convergence.  
- Combine deep learning and classical models in ensemble methods for stronger predictions.  
- Incorporate explainability tools to understand how models classify art movements.

