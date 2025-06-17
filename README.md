# Offensive Language Detection in Malayalam

This project aims to automatically identify offensive language in Malayalam text, leveraging the power of modern NLP and machine learning techniques. The focus is on detecting not only offensive content but also distinguishing it from non-offensive text and non-Malayalam inputs.

---

## 🔍 Project Overview

Detecting offensive language is crucial for maintaining healthy online communities. This system uses **DistilBERT multilingual embeddings** to represent text and trains an **XGBoost classifier** to categorize Malayalam sentences into three classes:

- **Offensive**
- **Not_offensive**
- **Not Malayalam**

The model is trained and validated on the **DravidianCodeMix @ FIRE 2025 dataset**, tailored for Malayalam offensive language detection.

---

## 🚀 Key Features

- **State-of-the-art embeddings:** Utilizes the lightweight yet powerful DistilBERT model for multilingual text encoding.
- **Robust classifier:** XGBoost with balanced sample weights to handle class imbalance effectively.
- **Multi-class classification:** Effectively distinguishes offensive language, clean Malayalam text, and non-Malayalam content.
- **Baseline comparisons:** Also experimented with KNN and SVM classifiers for benchmarking.
- **Efficient inference:** Embeddings can be computed batch-wise, making it scalable for large datasets.

---

## 🛠️ Getting Started

### Prerequisites

Make sure to install the following Python libraries:


pip install transformers xgboost scikit-learn torch matplotlib tqdm

sample_text = "ഇത് വളരെ മോശമായ ഒരു അഭിപ്രായമാണ്"
prediction = predict(sample_text)
print(f"Prediction: {prediction}")  # Output: Offensive / Not_offensive / Not Malayalam

📈 Results & Evaluation
The project compares multiple classifiers including XGBoost, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).

Each model was trained on DistilBERT embeddings extracted from the Malayalam text dataset.

We evaluated these models using accuracy scores and detailed classification reports.

Confusion matrices were plotted to visualize and analyze the performance of each classifier.

Based on these evaluations, XGBoost was selected as the best performing model due to its superior accuracy and balanced handling of all classes.

All training, evaluation, and visualization code, including model comparisons, is available in the notebook file index.ipynb.

Thank you for checking out this project!
Feel free to reach out if you want to collaborate or have questions.
