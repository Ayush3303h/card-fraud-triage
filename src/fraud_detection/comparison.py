import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def format_comparison_table(gmm_results, ae_results):
    """
    Format a side-by-side comparative table between GMM (Phase 1) and Autoencoder (Phase 2).
    Designed to present findings clearly for project reviews, papers, and vSphere presentations.
    """
    gmm_test = gmm_results.get("test", {})
    ae_test = ae_results.get("test", {})

    gmm_val = gmm_results.get("validation", {})
    ae_val = ae_results.get("validation", {})

    gmm_fpr = {r["target_fpr"]: r for r in gmm_results.get("fpr_analysis", [])}
    ae_fpr = {r["target_fpr"]: r for r in ae_results.get("fpr_analysis", [])}

    gmm_rev = {r["review_capacity"]: r for r in gmm_results.get("review_capacity", [])}
    ae_rev = {r["review_capacity"]: r for r in ae_results.get("review_capacity", [])}

    sep = "=" * 76
    sub = "-" * 76

    lines = []
    lines.append(sep)
    lines.append("     PHASE 1 (GMM) vs. PHASE 2 (AUTOENCODER) - COMPARATIVE BENCHMARK")
    lines.append(sep)

    lines.append(" 1. OVERALL TEST PERFORMANCE (Fixed F1 Threshold)")
    lines.append(f"    {'Metric':<30} | {'Phase 1 (GMM K=38)':<18} | {'Phase 2 (Autoencoder)':<20}")
    lines.append("    " + "-" * 72)
    lines.append(f"    {'Average Precision (PR-AUC)':<30} | {gmm_test.get('average_precision', 0):>18.4f} | {ae_test.get('average_precision', 0):>20.4f}")
    lines.append(f"    {'F1-Score':<30} | {gmm_test.get('f1', 0):>18.4f} | {ae_test.get('f1', 0):>20.4f}")
    lines.append(f"    {'Precision':<30} | {gmm_test.get('precision', 0)*100:>17.2f}% | {ae_test.get('precision', 0)*100:>19.2f}%")
    lines.append(f"    {'Recall (Fraud Caught)':<30} | {gmm_test.get('recall', 0)*100:>17.2f}% | {ae_test.get('recall', 0)*100:>19.2f}%")
    lines.append(f"    {'False Positive Rate (FPR)':<30} | {gmm_test.get('false_positive_rate', 0)*100:>17.4f}% | {ae_test.get('false_positive_rate', 0)*100:>19.4f}%")
    lines.append(f"    {'Frauds Detected / Total':<30} | {gmm_test.get('true_positives', 0):>15} / 52 | {ae_test.get('true_positives', 0):>17} / 52")
    lines.append(f"    {'False Alarms (out of 42,507)':<30} | {gmm_test.get('false_positives', 0):>18} | {ae_test.get('false_positives', 0):>20}")

    lines.append(f"\n{sub}")
    lines.append(" 2. RECALL AT CONTROLLED FALSE POSITIVE BUDGETS")
    lines.append(f"    {'Target FPR Budget':<20} | {'GMM Recall (Caught)':<24} | {'AE Recall (Caught)':<24}")
    lines.append("    " + "-" * 72)
    for target in [0.01, 0.005, 0.001]:
        g = gmm_fpr.get(target, {})
        a = ae_fpr.get(target, {})
        g_str = f"{g.get('fraud_recall', 0)*100:.1f}% ({g.get('fraud_detected', 0)}/52, FP:{g.get('false_alarms', 0)})"
        a_str = f"{a.get('fraud_recall', 0)*100:.1f}% ({a.get('fraud_detected', 0)}/52, FP:{a.get('false_alarms', 0)})"
        lines.append(f"    FPR <= {target*100:>4.2f}%         | {g_str:<24} | {a_str:<24}")

    lines.append(f"\n{sub}")
    lines.append(" 3. OPERATIONAL TRIAGE: PRECISION AT FIXED REVIEW CAPACITY")
    lines.append(f"    {'Review Capacity':<20} | {'GMM Frauds Found':<24} | {'AE Frauds Found':<24}")
    lines.append("    " + "-" * 72)
    for cap in [50, 100, 200]:
        g = gmm_rev.get(cap, {})
        a = ae_rev.get(cap, {})
        g_str = f"{g.get('fraud_found', 0)}/52 ({g.get('precision', 0)*100:.1f}% prec)"
        a_str = f"{a.get('fraud_found', 0)}/52 ({a.get('precision', 0)*100:.1f}% prec)"
        lines.append(f"    Top {cap:<16} | {g_str:<24} | {a_str:<24}")

    lines.append(sep)
    return "\n".join(lines)


def plot_comparative_pr_curves(
    y_test,
    gmm_scores,
    ae_scores,
    save_path="artifacts/comparative_pr_curve.png",
):
    """Plot overlayed Precision-Recall curves comparing GMM and Autoencoder."""
    precision_gmm, recall_gmm, _ = precision_recall_curve(y_test, gmm_scores)
    ap_gmm = average_precision_score(y_test, gmm_scores)

    precision_ae, recall_ae, _ = precision_recall_curve(y_test, ae_scores)
    ap_ae = average_precision_score(y_test, ae_scores)

    baseline = np.mean(y_test)

    plt.figure(figsize=(9, 6), dpi=150)
    plt.plot(
        recall_gmm,
        precision_gmm,
        color="#2b5c8f",
        lw=2.2,
        label=f"Phase 1: GMM (AP = {ap_gmm:.4f})",
    )
    plt.plot(
        recall_ae,
        precision_ae,
        color="#d95f02",
        lw=2.2,
        linestyle="-",
        label=f"Phase 2: Autoencoder (AP = {ap_ae:.4f})",
    )
    plt.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        lw=1.2,
        label=f"No-Skill Prevalence ({baseline*100:.3f}%)",
    )

    plt.title("Precision-Recall Benchmark: GMM vs. Autoencoder", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recall (Fraction of Frauds Caught)", fontsize=11)
    plt.ylabel("Precision (Fraction of True Frauds in Alarms)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
