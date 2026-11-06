import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.title("📱 Mobile Price Classification Explorer")

# Map models to saved joblib file names
files = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm"
}

choice = st.selectbox("Choose a model", list(files.keys()))
up = st.file_uploader("Upload test CSV (with 'price_range' column)", type="csv")

if up:
    df = pd.read_csv(up)
    if "price_range" not in df.columns:
        st.error("The uploaded CSV must contain the 'price_range' target column.")
    else:
        X, y = df.drop(columns=["price_range"]), df["price_range"]

        # Load pre-trained scaler and selected model
        scaler = joblib.load("model/scaler.joblib")
        model = joblib.load(f"model/{files[choice]}.joblib")

        # Scale features and run predictions
        Xs = scaler.transform(X)
        pred = model.predict(Xs)
        proba = model.predict_proba(Xs)

        # Metrics calculations adjusted for multi-class classification
        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, proba, multi_class="ovr")
        mcc = matthews_corrcoef(y, pred)
        prec = precision_score(y, pred, average="weighted")
        rec = recall_score(y, pred, average="weighted")
        f1 = f1_score(y, pred, average="weighted")

        # Display key metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("AUC (OVR)", f"{auc:.3f}")
        c3.metric("MCC", f"{mcc:.3f}")

        st.write(f"**Weighted Precision:** {prec:.3f} | **Weighted Recall:** {rec:.3f} | **Weighted F1 Score:** {f1:.3f}")

        # Plot Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
        ax.set_xlabel("Predicted Price Range")
        ax.set_ylabel("True Price Range")
        st.pyplot(fig)

        # Display Classification Report
        st.subheader("Detailed Classification Report")
        st.text(classification_report(y, pred))