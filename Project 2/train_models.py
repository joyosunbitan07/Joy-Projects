import os
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def main():
    # =========================================================
    # PRODUCTION-READY MALWARE DETECTION TRAINING PIPELINE
    # =========================================================
    print("Loading clean processed datasets...")
    # Loading flat datasets from the current folder
    train_df = pd.read_csv("./train_flat.csv")
    val_df = pd.read_csv("./val_flat.csv")
    test_df = pd.read_csv("./test_flat.csv")

    # Extracting features and targets
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].values.ravel()

    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"].values.ravel()

    print(f"Training Shape: {X_train.shape} | Validation Shape: {X_val.shape}")

    # =========================================================
    # MODEL FIT (Using LightGBM)
    # =========================================================
    print("\n[!] Training LightGBM Classifier...")
    # 'is_unbalance=True' internally handles the 90/10 skew during training
    final_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        is_unbalance=True,
        random_state=42,
        n_jobs=1,
        deterministic=True, #ensures exact same weights
    )
    final_model.fit(X_train, y_train)

    # =========================================================
    # THRESHOLD TUNING FOR OPERATIONAL REQUIREMENT (90/90)
    # =========================================================
    print("\nEvaluating model probabilities...")
    # to confirm the threshold ill look at validation probabilities
    y_val_prob = final_model.predict_proba(X_val)[:, 1]
    custom_threshold = 0.38

    #now ill run evaluation on the test set
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].values.ravel()

    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob > custom_threshold).astype(int)
    print(f"Applied Decision Threshold: {custom_threshold}")
    print("\n=== FINAL REPRODUCIBLE VALIDATION REPORT ===")

    print(
        classification_report(
            y_test, y_test_pred, target_names=["Benign", "Malware"], digits=4
        )
    )

    # =========================================================
    # ASSET STORAGE & EXPORT
    # =========================================================
    print("\nArchiving model and generating report visuals...")
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(
        final_model, max_num_features=15, importance_type="split", ax=plt.gca()
    )
    plt.tight_layout()

    # Saving assets flat into the project2 directory
    plt.savefig("./feature_importance.png")

    with open("./malware_lgb_model.pkl", "wb") as file:
        pickle.dump(final_model, file)

    print("!!! MODEL PIPELINE EXECUTION COMPLETE AND SAVED !!!")


if __name__ == "__main__":
    main()