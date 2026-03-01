import numpy as np
import joblib

from src.config import MODELS_DIR, FIGURES_DIR, SEED
from src.data_loader import load_dataset
from src.preprocessing import prepare_train_test
from src.train import train_lgbm
from src.evaluate import (
    predict_proba,
    save_roc_train_test,
    save_pr_train_test,
    save_confusion_matrices,
)
from src.interpretability import shap_summary

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df_prepared, features, pack = prepare_train_test(df)
    X_train, y_train, X_test, y_test, *_ = pack

    model, best_params, best_cv_auc = train_lgbm(X_train, y_train)

    p_train = predict_proba(model, X_train)
    p_test = predict_proba(model, X_test)
    
    auc_tr, auc_te = save_roc_train_test(y_train, p_train, y_test, p_test, filename="roc_train_test.png")
    ap_tr, ap_te, baseline = save_pr_train_test(y_train, p_train, y_test, p_test, filename="pr_train_test.png")

    thresholds = (0.5, 0.685, 0.715)
    cm_summaries = save_confusion_matrices(y_test.values, p_test, thresholds=thresholds, filename="confusion_matrices.png")
   
    rng = np.random.default_rng(SEED)
    sample_n = min(15000, len(X_train))
    sample_idx = rng.choice(len(X_train), size=sample_n, replace=False)
    X_sample = X_train.iloc[sample_idx].copy()
    shap_summary(model, X_sample, filename="shap_summary.png")
    
    joblib.dump(
        {"model": model, "features": features, "best_params": best_params},
        MODELS_DIR / "lgbm_injury_risk.joblib",
    )
    
    print(f"Best CV ROC-AUC: {best_cv_auc:.4f}")
    print("Best params:", best_params)
    print(f"Train ROC-AUC: {auc_tr:.4f} | Test ROC-AUC: {auc_te:.4f}")
    print(f"Train AP:      {ap_tr:.4f} | Test AP:      {ap_te:.4f} | Baseline: {baseline:.4f}")
    print("Confusion matrix summaries (test):")
    for s in cm_summaries:
        print(s)

    print("Saved:")
    print("- outputs/models/lgbm_injury_risk.joblib")
    print("- outputs/figures/roc_train_test.png")
    print("- outputs/figures/pr_train_test.png")
    print("- outputs/figures/confusion_matrices.png")
    print("- outputs/figures/shap_summary.png")


if __name__ == "__main__":
    main()