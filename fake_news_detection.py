"""
Fake News Detection using Machine Learning Algorithms
------------------------------------------------------
Course: Machine Learning Mini-Project
Authors:
  - Hema Varshini (25RBAIM0040)
  - Madhu Sri (25RBAIM0057)
  - Divya Madeti (25RBAIM0058)
  - Ch. J V V Sandeep (25RBAIM0061)

Department: CSE (AI & ML), II Year, I Semester
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# ---------------------------------------------------------
# 1. PREPROCESSING FUNCTION
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans raw news text:
    - Lowercasing
    - Removing URLs and web addresses
    - Removing non-alphabetic characters and digits
    - Normalizing whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # Remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)           # Keep alphabetic letters only
    text = re.sub(r"\s+", " ", text).strip()         # Remove extra whitespace
    return text


# ---------------------------------------------------------
# 2. PLOTTING FUNCTIONS
# ---------------------------------------------------------
def save_class_distribution(df, output_path="outputs/fig1_class_distribution.png"):
    """Generates Figure 1: Class distribution of the dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = df["label"].value_counts().sort_index()
    labels = ["Real (0)", "Fake (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    
    plt.figure(figsize=(7, 5), dpi=300)
    bars = plt.bar(labels, values, color=["#1f77b4", "#b83227"], width=0.6)
    plt.title("Class distribution of the dataset", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Number of articles", fontsize=12)
    plt.ylim(0, max(values) + 60)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h}", ha="center", va="bottom", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved: {output_path}")


def save_model_comparison(results_df, output_path="outputs/fig3_model_comparison.png"):
    """Generates Figure 3: Performance comparison of the ML models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = results_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    rects1 = ax.bar(x - 1.5*width, results_df["Accuracy"] / 100, width, label="Accuracy", color="#1c315e")
    rects2 = ax.bar(x - 0.5*width, results_df["Precision"] / 100, width, label="Precision", color="#227c9d")
    rects3 = ax.bar(x + 0.5*width, results_df["Recall"] / 100, width, label="Recall", color="#8ecae6")
    rects4 = ax.bar(x + 1.5*width, results_df["F1-score"] / 100, width, label="F1-score", color="#c13b2a")
    
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
    ax.set_title("Performance comparison of the ML models", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=0)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved: {output_path}")


def save_confusion_matrix(cm, output_path="outputs/fig4_confusion_matrix.png"):
    """Generates Figure 4: Confusion matrix of Logistic Regression."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5.5), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix - Logistic Regression", fontsize=13, fontweight="bold", pad=12)
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Real", "Fake"], fontsize=11)
    plt.yticks(tick_marks, ["Real", "Fake"], fontsize=11)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=16)
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved: {output_path}")


def save_roc_curves(roc_data, output_path="outputs/fig5_roc_curves.png"):
    """Generates Figure 5: ROC curves of the models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    
    colors = {
        "Logistic Regression": "#1c315e",
        "Multinomial Naive Bayes": "#1f77b4",
        "Linear SVM": "#c13b2a",
        "Passive Aggressive": "#2ca02c",
        "Random Forest": "#9467bd"
    }
    
    for name, data in roc_data.items():
        fpr, tpr, roc_auc = data
        plt.plot(fpr, tpr, lw=2, color=colors.get(name, "gray"), label=f"{name} (AUC={roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], color="black", lw=1.2, linestyle="--")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False positive rate", fontsize=11)
    plt.ylabel("True positive rate", fontsize=11)
    plt.title("ROC curves", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved: {output_path}")


def save_top_features(vec, model, output_path="outputs/fig6_top_features.png", top_n=10):
    """Generates Figure 6: Top features of Logistic Regression model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_names = np.array(vec.get_feature_names_out())
    coef = model.coef_[0]
    
    # Top fake features (most positive coefficients)
    top_fake_idx = np.argsort(coef)[-top_n:]
    # Top real features (most negative coefficients)
    top_real_idx = np.argsort(coef)[:top_n]
    
    indices = np.hstack([top_real_idx, top_fake_idx])
    features = feature_names[indices]
    coefficients = coef[indices]
    
    colors = ["#1f77b4" if c < 0 else "#c13b2a" for c in coefficients]
    
    plt.figure(figsize=(9, 6), dpi=300)
    y_pos = np.arange(len(features))
    plt.barh(y_pos, coefficients, color=colors, align="center")
    plt.yticks(y_pos, features, fontsize=10)
    plt.xlabel("Coefficient  (\u2190 real | fake \u2192)", fontsize=11)
    plt.title("Top words/bigrams driving the Logistic Regression decision", fontsize=13, fontweight="bold", pad=12)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved: {output_path}")


# ---------------------------------------------------------
# 3. MAIN TRAINING & EVALUATION PIPELINE
# ---------------------------------------------------------
def main():
    print("=" * 70)
    print("  FAKE NEWS DETECTION USING MACHINE LEARNING ALGORITHMS  ")
    print("  B.Tech CSE (AI & ML) - Mini Project Pipeline           ")
    print("=" * 70)

    # 1. Locate dataset
    dataset_path = "dataset/fake_news_sample.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "fake_news_sample.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path} or fake_news_sample.csv")

    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path).dropna().drop_duplicates()
    total_articles = len(df)
    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    print(f"[*] Total Articles Loaded: {total_articles}")
    print(f"[*] Class Distribution: Real (0) = {real_count}, Fake (1) = {fake_count}")

    # Generate Figure 1
    save_class_distribution(df)

    # 2. Text Pre-processing
    print("[*] Pre-processing news text (cleaning, URL removal, lowercasing)...")
    df["clean"] = df["text"].apply(clean_text)

    # 3. Train-Test Split (80% Train, 20% Test, Stratified)
    print("[*] Splitting dataset into 80% train and 20% test (stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"],
        df["label"],
        test_size=0.20,
        stratify=df["label"],
        random_state=42
    )
    print(f"[*] Training samples: {len(X_train)} ({len(X_train)/total_articles*100:.1f}%)")
    print(f"[*] Testing samples:  {len(X_test)} ({len(X_test)/total_articles*100:.1f}%)")

    # 4. Feature Extraction using TF-IDF (Unigrams & Bigrams)
    print("[*] Extracting TF-IDF features (unigrams + bigrams, sublinear TF)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"[*] Extracted TF-IDF Features (Vocabulary size): {X_train_vec.shape[1]}")

    # 5. Define ML Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(C=1.0, random_state=42),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    # 6. Train and Evaluate Each Model
    print("\n" + "=" * 70)
    print("  MODEL TRAINING & EVALUATION RESULTS ON TEST SET (240 Articles) ")
    print("=" * 70)

    results = []
    roc_data = {}
    best_model_name = "Logistic Regression"
    trained_models = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train_vec, y_train)
        trained_models[name] = model

        # Predict on test set
        y_pred = model.predict(X_test_vec)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        })

        # Calculate ROC and AUC
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test_vec)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_vec)
        else:
            scores = y_pred

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # 7. Generate Figures 3, 4, 5, 6
    save_model_comparison(results_df)

    lr_model = trained_models["Logistic Regression"]
    lr_pred = lr_model.predict(X_test_vec)
    cm = confusion_matrix(y_test, lr_pred)
    print(f"\n[*] Logistic Regression Confusion Matrix (Test Set):")
    print(f"    True Real (TN): {cm[0, 0]} | False Fake (FP): {cm[0, 1]}")
    print(f"    False Real (FN): {cm[1, 0]}  | True Fake (TP):  {cm[1, 1]}")
    save_confusion_matrix(cm)

    save_roc_curves(roc_data)
    save_top_features(vectorizer, lr_model)

    # 8. Save Serialized Models
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_model.pkl"
    vec_path = "models/tfidf_vectorizer.pkl"
    joblib.dump(lr_model, best_model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"\n[+] Serialized best model saved to: {best_model_path}")
    print(f"[+] Serialized TF-IDF vectorizer saved to: {vec_path}")

    # 9. Verify Sample Test Predictions (Table 8)
    print("\n" + "=" * 70)
    print("  VERIFYING SAMPLE PREDICTIONS (FROM REPORT TABLE 8)  ")
    print("=" * 70)
    sample_tests = [
        "BREAKING: The finance minister is secretly hiding the truth about your bank savings. Share this before it gets deleted!",
        "The health ministry published results of a clinical study, according to an official statement."
    ]
    for text in sample_tests:
        cleaned = clean_text(text)
        vec_feat = vectorizer.transform([cleaned])
        pred = lr_model.predict(vec_feat)[0]
        prob = lr_model.predict_proba(vec_feat)[0]
        pred_label = "FAKE" if pred == 1 else "REAL"
        confidence = prob[pred] * 100
        print(f"\n[Text]: {text}")
        print(f"[Prediction]: -> {pred_label} (Confidence: {confidence:.2f}%)")

    print("\n[+] Training & evaluation pipeline complete!")
    print("[*] Launch interactive Web UI with: streamlit run app.py\n")


# ---------------------------------------------------------
# 4. PREDICTION INFERENCE UTILITY
# ---------------------------------------------------------
def predict_news(text: str, model_path="models/best_model.pkl", vec_path="models/tfidf_vectorizer.pkl"):
    """Inference helper for custom user texts."""
    model = joblib.load(model_path)
    vec = joblib.load(vec_path)
    cleaned = clean_text(text)
    features = vec.transform([cleaned])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    return {
        "label": "FAKE" if pred == 1 else "REAL",
        "confidence": prob[pred],
        "prob_real": prob[0],
        "prob_fake": prob[1]
    }


if __name__ == "__main__":
    main()

