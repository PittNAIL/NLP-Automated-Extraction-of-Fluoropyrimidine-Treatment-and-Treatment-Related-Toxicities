# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


TAGS = [
    'handfootpreventative',
    'cartoxarrhythmia',
    'cartoxheartfailure',
    'cartoxvalvularcomplications',
    'drugsofinterest'
]

def train_model(tag, model_type):
    print(f"\nTraining {model_type.upper()} model for tag: {tag}")

    base_path = f"D:\\Github\\MedSDoH\\data\\train_cape\\split_datasets\\{tag}"
    train_path = os.path.join(base_path, 'train.csv')
    val_path = os.path.join(base_path, 'val.csv')
    model_dir = f"{model_type}_models"
    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    X_train, y_train = train_df['sentence'].values, train_df['target'].values
    X_val, y_val = val_df['sentence'].values, val_df['target'].values

    if model_type == 'rf':
        vectorizer = CountVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'lr':
        clf = LogisticRegression(
            max_iter=3000,
            class_weight='balanced',
            random_state=42
        )
    elif model_type == 'svm': 
        clf = SVC(
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        )

    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_val_vec)
    report = classification_report(y_val, y_pred, digits=4)
    cm = confusion_matrix(y_val, y_pred)

    report_path = os.path.join(model_dir, f"{tag}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report - {tag} ({model_type.upper()})\n\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    print("Classification Report:")
    print(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {tag} ({model_type.upper()})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{tag}_confusion_matrix.png"))
    plt.close()

    model_file = os.path.join(model_dir, f"{tag}_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump({'model': clf, 'vectorizer': vectorizer}, f)

    print(f"Model saved to: {model_file}")

def main():
    for tag in TAGS:
        train_model(tag, model_type='svm')
        train_model(tag, model_type='rf')
        train_model(tag, model_type='lr')

if __name__ == "__main__":
    main()
