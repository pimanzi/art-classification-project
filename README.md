# 🎨 Art Movement Classification Project

## Problem Statement

Classifying artworks based on their styles often requires the expertise of trained art historians or curators, making the process time-consuming and inaccessible to the general public. This poses a challenge for websites, museums, galleries, and educational platforms that need efficient tools to tag, organize, and present artworks. There is a need for an automated system that can accurately classify artworks by style, making it easier for non-experts to explore and understand art. Such a system can enhance digital catalogs, improve visitor experience in museums, and support online art platforms in presenting culturally relevant content.

## Project Motif

The core aim is to understand how different modeling approaches and their hyperparameters affect classification accuracy, precision, recall, and F1 score in artwork categorization. We want to see how fine-tuning ResNet50’s parameters or using classical methods with dimensionality reduction (PCA) impacts results.

---

## Dataset

We used the **WikiArt dataset**, which contains a rich collection of paintings labeled by their respective art movements. This dataset spans 13 categories including Primitivism, Romanticism, Baroque, Renaissance, and more, providing a diverse basis for training and evaluation.
Link of dataset : [WikiArt - Art Movements/Styles Dataset on Kaggle](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles)

---

## Model Performance Comparison

### Efficient Net B0 Neural Network Results

### 📊 Model Comparison Table

| Training Instance | Optimizer | Regularizer | Epochs | Early Stopping | Total Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
| ----------------- | --------- | ----------- | ------ | -------------- | ------------ | ------------- | -------- | -------- | ------ | --------- |
| `cnn_notuning`    | Adam      | None        | 15     | No             | 309          | default       | 0.6092   | 0.6108   | 0.6092 | 0.6220    |
| `model_2`         | Adam      | L2          | 15     | Yes            | 309          | 0.0005        | 0.5918   | 0.5821   | 0.5918 | 0.5901    |
| `model_3`         | RMSprop   | L1          | 15     | Yes            | 309          | 0.0001        | 0.6379   | 0.6322   | 0.6379 | 0.6394    |
| `model_adagrad`   | Adagrad   | L1 + L2     | 12     | Yes            | 309          | 0.01          | 0.5826   | 0.5752   | 0.5826 | 0.5765    |

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

1. Random Forest with PCA provided a baseline but struggled to capture the complexity in artwork images. The hyperparameters helped manage model size and prevent overfitting, but precision, recall, and F1 remained low due to limited feature extraction capability.

2. The baseline model without any hyperparameter tuning achieved a solid accuracy of 60.9%, showing that EfficientNetB0 alone has strong performance, though it showed signs of overfitting after epoch 10.

3. Model 2, using Adam optimizer with L2 regularization and a lower learning rate, showed more stable training but slightly reduced accuracy at 59.2%, likely due to early stopping halting training before full convergence to avoid overfitting.

4. Model 3 delivered the best results with 63.8% accuracy, where RMSprop and L1 regularization helped the model steadily improve without significant overfitting.

5. Model 4, trained with Adagrad optimizer and combined L1_L2 regularization, showed slower convergence and early plateau at 58.3% accuracy, with its aggressive learning rate decay limiting further improvements and making the model more prone to underfitting.

## Summary

Random Forest with PCA provided a basic baseline but struggled with the complexity of art images. The untuned EfficientNetB0 model achieved solid accuracy (60.9%) but showed overfitting after epoch 10. Model 2’s use of Adam optimizer with L2 regularization and early stopping improved training stability but slightly lowered accuracy to 59.2%. **_Model 3_**, with RMSprop and L1 regularization, was the best performer at 63.8% accuracy, balancing steady learning and minimal overfitting. Model 4’s Adagrad optimizer and combined L1_L2 regularization led to slower convergence and underfitting, reaching 58.3% accuracy.

### Why Model 3 is the best:

1. It achieved the highest accuracy, showing better learning and generalization on the validation set.
2. Its use of RMSprop with L1 regularization helped reduce overfitting while allowing steady improvements during training.

### Overall summary:

The experiments show that careful selection of optimizer and regularization is crucial. While baseline models perform reasonably, tuning these hyperparameters significantly impacts model stability and accuracy, with Model 3 demonstrating the best balance for classifying complex art styles effectively.

## 📺 Demo Video

You can watch a detailed walkthrough of this project here:

---
