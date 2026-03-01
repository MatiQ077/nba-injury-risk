import shap
import matplotlib.pyplot as plt
from .config import FIGURES_DIR

def shap_summary(model, X_sample, filename="shap_summary.png"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()