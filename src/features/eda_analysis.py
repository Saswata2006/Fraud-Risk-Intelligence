"""Exploratory Data Analysis and Visualization Engine for Fraud Detection.

Provides statistical profiling, correlation analysis, amount tiering,
and publication-quality chart generation using Matplotlib and Plotly.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute fraud vs legitimate counts and percentages."""
    counts = df["Class"].value_counts().to_dict()
    legit_count = int(counts.get(0, 0))
    fraud_count = int(counts.get(1, 0))
    total = legit_count + fraud_count
    fraud_pct = (fraud_count / total * 100) if total > 0 else 0.0
    return {
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "total_count": total,
        "fraud_percentage": fraud_pct,
        "legit_percentage": 100.0 - fraud_pct,
        "imbalance_ratio": f"{round(legit_count / fraud_count, 1)}:1" if fraud_count > 0 else "N/A"
    }


def compute_amount_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comparative amount summary statistics between classes."""
    stats = []
    for cls, label in [(0, "Legitimate"), (1, "Fraudulent")]:
        subset = df[df["Class"] == cls]["Amount"]
        stats.append({
            "Class": cls,
            "Type": label,
            "Count": len(subset),
            "Mean": subset.mean(),
            "Std": subset.std(),
            "Min": subset.min(),
            "Q25": subset.quantile(0.25),
            "Median": subset.median(),
            "Q75": subset.quantile(0.75),
            "Max": subset.max(),
            "Total_Volume": subset.sum(),
        })
    return pd.DataFrame(stats)


def compute_correlations_with_target(df: pd.DataFrame, target_col: str = "Class") -> pd.Series:
    """Compute Pearson correlation of all features with the target."""
    corr = df.corr()[target_col].drop(target_col).sort_values()
    return corr


def compute_amount_bands_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute transaction counts and fraud rate across defined amount bands."""
    bins = [-0.01, 0.0, 10.0, 50.0, 100.0, 500.0, 1000.0, float("inf")]
    labels = [
        "$0 (Zero-Dollar)",
        "$0.01 - $10",
        "$10.01 - $50",
        "$50.01 - $100",
        "$100.01 - $500",
        "$500.01 - $1,000",
        "> $1,000"
    ]
    df_copy = df[["Amount", "Class"]].copy()
    df_copy["Amount_Band"] = pd.cut(df_copy["Amount"], bins=bins, labels=labels)
    grouped = df_copy.groupby("Amount_Band", observed=False).agg(
        Total_Count=("Class", "count"),
        Fraud_Count=("Class", "sum"),
    ).reset_index()
    grouped["Legit_Count"] = grouped["Total_Count"] - grouped["Fraud_Count"]
    grouped["Fraud_Rate_Pct"] = (grouped["Fraud_Count"] / grouped["Total_Count"] * 100).fillna(0.0)
    return grouped


def plot_class_distribution(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[str, str]:
    """Chart 1: Class distribution with count and percentage annotations."""
    dist = compute_class_distribution(df)

    # 1. Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Raw counts (log-scale)
    categories = ["Legitimate (0)", "Fraudulent (1)"]
    counts = [dist["legit_count"], dist["fraud_count"]]
    colors = ["#2b5c8f", "#d9534f"]
    
    bars = ax1.bar(categories, counts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_title("Transaction Count by Class (Logarithmic Scale)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count (log scale)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{int(h):,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Percentage breakdown
    pcts = [dist["legit_percentage"], dist["fraud_percentage"]]
    bars2 = ax2.bar(categories, pcts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax2.set_title("Class Proportion (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("ULB Credit Card Fraud Class Imbalance (Imbalance Ratio: ~578:1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    mpl_path = output_dir / "01_class_distribution.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly
    fig_plotly = px.bar(
        x=categories,
        y=counts,
        color=categories,
        color_discrete_sequence=["#2b5c8f", "#d9534f"],
        labels={"x": "Class", "y": "Transaction Count"},
        title="Transaction Class Distribution (Log Scale, Fraud = 0.172%)",
        log_y=True,
        text=[f"{c:,} ({p:.3f}%)" for c, p in zip(counts, pcts)]
    )
    fig_plotly.update_layout(showlegend=False, template="plotly_white")
    plotly_path = output_dir / "01_class_distribution.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_amount_distribution(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[str, str]:
    """Chart 2: Transaction amount distribution (Legitimate vs Fraudulent)."""
    legit_amt = df[df["Class"] == 0]["Amount"]
    fraud_amt = df[df["Class"] == 1]["Amount"]

    # 1. Matplotlib (Boxplot and Log-Histogram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot of log(Amount + 1)
    ax1.boxplot([np.log1p(legit_amt), np.log1p(fraud_amt)], tick_labels=["Legitimate", "Fraudulent"], patch_artist=True,
                boxprops=dict(facecolor="#90afc5", color="#2b5c8f"),
                medianprops=dict(color="#d9534f", linewidth=2))
    ax1.set_title("Log(Amount + $1) Comparison", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Log(Amount + 1) [USD]", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlaid Log-Scale Histograms
    bins = np.logspace(0, np.log10(max(df["Amount"].max(), 10)), 50)
    ax2.hist(legit_amt + 1, bins=bins, alpha=0.6, label=f"Legit (Median: ${legit_amt.median():.2f})", color="#2b5c8f", density=True)
    ax2.hist(fraud_amt + 1, bins=bins, alpha=0.7, label=f"Fraud (Median: ${fraud_amt.median():.2f})", color="#d9534f", density=True)
    ax2.set_xscale("log")
    ax2.set_title("Density Histogram of Amounts (Log-Scaled X)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Amount + $1 (USD, Log Scale)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Transaction Amount Distribution: Legitimate vs Fraudulent", fontsize=14, fontweight="bold")
    plt.tight_layout()
    mpl_path = output_dir / "02_amount_distribution_by_class.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly Interactive Violin / Box
    sample_df = pd.concat([
        df[df["Class"] == 0].sample(n=10000, random_state=42),
        df[df["Class"] == 1]
    ]).copy()
    sample_df["Type"] = sample_df["Class"].map({0: "Legitimate", 1: "Fraudulent"})
    sample_df["Log_Amount"] = np.log1p(sample_df["Amount"])

    fig_plotly = px.box(
        sample_df,
        x="Type",
        y="Amount",
        color="Type",
        log_y=True,
        color_discrete_map={"Legitimate": "#2b5c8f", "Fraudulent": "#d9534f"},
        title="Transaction Amount Distribution (Log Scale)",
        points="outliers"
    )
    fig_plotly.update_layout(template="plotly_white")
    plotly_path = output_dir / "02_amount_distribution_by_class.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_transaction_frequency_time(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[str, str]:
    """Chart 3: Transaction frequency and fraud events across 48 hours."""
    df_time = df[["Time", "Class"]].copy()
    df_time["Elapsed_Hour"] = (df_time["Time"] // 3600).astype(int)

    hourly = df_time.groupby("Elapsed_Hour").agg(
        Total=("Class", "count"),
        Fraud=("Class", "sum")
    ).reset_index()
    hourly["Legit"] = hourly["Total"] - hourly["Fraud"]
    hourly["Fraud_Rate_Pct"] = hourly["Fraud"] / hourly["Total"] * 100

    # 1. Matplotlib Dual-Axis Timeline
    fig, ax1 = plt.subplots(figsize=(14, 5))

    color = "#2b5c8f"
    ax1.set_xlabel("Elapsed Time (Hours from Start of Recording)", fontsize=11)
    ax1.set_ylabel("Total Transactions / Hour", color=color, fontsize=11)
    l1 = ax1.plot(hourly["Elapsed_Hour"], hourly["Total"], color=color, linewidth=2.5, label="Total Transactions (Volume)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Secondary axis for fraud counts
    ax2 = ax1.twinx()
    color2 = "#d9534f"
    ax2.set_ylabel("Fraudulent Transactions / Hour", color=color2, fontsize=11)
    l2 = ax2.plot(hourly["Elapsed_Hour"], hourly["Fraud"], color=color2, linewidth=2.5, linestyle="--", marker="o", label="Fraud Events")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title("Transaction Volume vs Fraud Incidents Over 48-Hour Window", fontsize=14, fontweight="bold")
    plt.tight_layout()
    mpl_path = output_dir / "03_transaction_frequency_time.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly Interactive Dual-Axis
    fig_plotly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_plotly.add_trace(
        go.Scatter(x=hourly["Elapsed_Hour"], y=hourly["Total"], name="Total Volume", line=dict(color="#2b5c8f", width=3)),
        secondary_y=False
    )
    fig_plotly.add_trace(
        go.Scatter(x=hourly["Elapsed_Hour"], y=hourly["Fraud"], name="Fraud Count", line=dict(color="#d9534f", width=3, dash="dot"), mode="lines+markers"),
        secondary_y=True
    )
    fig_plotly.update_layout(
        title_text="Transaction Volume & Fraud Frequency Over 48 Hours",
        template="plotly_white",
        xaxis_title="Elapsed Hour (0-47)"
    )
    fig_plotly.update_yaxes(title_text="Total Transactions", secondary_y=False)
    fig_plotly.update_yaxes(title_text="Fraud Count", secondary_y=True)
    plotly_path = output_dir / "03_transaction_frequency_time.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_fraud_rate_by_amount_bands(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[str, str]:
    """Chart 4: Fraud incidence rate across financial amount brackets."""
    bands_df = compute_amount_bands_summary(df)

    # 1. Matplotlib
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = range(len(bands_df))
    bars = ax1.bar(x, bands_df["Fraud_Rate_Pct"], color="#e06666", edgecolor="black", alpha=0.85, width=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands_df["Amount_Band"], rotation=25, ha="right", fontsize=10)
    ax1.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Fraud Rate (%) by Transaction Amount Band", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, fraud_cnt, tot_cnt in zip(bars, bands_df["Fraud_Count"], bands_df["Total_Count"]):
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}%\n({fraud_cnt}/{tot_cnt:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    mpl_path = output_dir / "04_fraud_rate_by_amount_bands.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly
    fig_plotly = px.bar(
        bands_df,
        x="Amount_Band",
        y="Fraud_Rate_Pct",
        text=bands_df.apply(lambda r: f"{r['Fraud_Rate_Pct']:.3f}% ({int(r['Fraud_Count'])}/{int(r['Total_Count']):,})", axis=1),
        title="Fraud Incidence Rate by Transaction Amount Tier",
        labels={"Amount_Band": "Amount Tier", "Fraud_Rate_Pct": "Fraud Rate (%)"},
        color="Fraud_Rate_Pct",
        color_continuous_scale="Reds"
    )
    fig_plotly.update_layout(template="plotly_white")
    plotly_path = output_dir / "04_fraud_rate_by_amount_bands.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_top_feature_correlations(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[str, str]:
    """Chart 5: Top positive and negative correlated features with Class."""
    corr = compute_correlations_with_target(df)
    
    # Top 6 negative and top 6 positive
    top_neg = corr.head(6)
    top_pos = corr.tail(6)
    top_corr = pd.concat([top_neg, top_pos])

    # 1. Matplotlib Horizontal Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d9534f" if v > 0 else "#2b5c8f" for v in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation Coefficient with Target (Class)", fontsize=11)
    ax.set_title("Top 12 Most Correlated Features with Fraud (Class)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        xpos = w + (0.01 if w > 0 else -0.01)
        ha = "left" if w > 0 else "right"
        ax.annotate(f"{w:.3f}", xy=(xpos, bar.get_y() + bar.get_height() / 2),
                    va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.tight_layout()
    mpl_path = output_dir / "05_top_feature_correlations.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly
    corr_df = pd.DataFrame({"Feature": top_corr.index, "Correlation": top_corr.values})
    corr_df["Direction"] = corr_df["Correlation"].apply(lambda x: "Positive" if x > 0 else "Negative")
    fig_plotly = px.bar(
        corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={"Positive": "#d9534f", "Negative": "#2b5c8f"},
        title="Key Features Correlated with Credit Card Fraud",
        text=corr_df["Correlation"].apply(lambda x: f"{x:.3f}")
    )
    fig_plotly.update_layout(template="plotly_white")
    plotly_path = output_dir / "05_top_feature_correlations.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_key_anonymized_features_distribution(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[str, str]:
    """Chart 6: Distribution of key discriminatory features (V14, V17, V12, V10, V4, V11)."""
    key_features = ["V14", "V17", "V12", "V10", "V4", "V11"]

    # 1. Matplotlib Subplots (Boxplots/Violins)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features):
        ax = axes[idx]
        legit_vals = df[df["Class"] == 0][feat]
        fraud_vals = df[df["Class"] == 1][feat]

        ax.boxplot([legit_vals, fraud_vals], tick_labels=["Legit", "Fraud"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#cbe3f2", color="#2b5c8f"),
                   medianprops=dict(color="#d9534f", linewidth=2),
                   showfliers=False)
        ax.set_title(f"Feature {feat} Distribution", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Distribution of Top Discriminatory Features (Outliers suppressed for clarity)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    mpl_path = output_dir / "06_key_anonymized_features_distribution.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly Interactive Multi-facet Box
    sample_df = pd.concat([
        df[df["Class"] == 0].sample(n=5000, random_state=42),
        df[df["Class"] == 1]
    ]).copy()
    sample_df["Type"] = sample_df["Class"].map({0: "Legitimate", 1: "Fraudulent"})
    
    melted = sample_df.melt(id_vars=["Type"], value_vars=key_features, var_name="Feature", value_name="Value")
    fig_plotly = px.box(
        melted,
        x="Feature",
        y="Value",
        color="Type",
        color_discrete_map={"Legitimate": "#2b5c8f", "Fraudulent": "#d9534f"},
        title="Distributions of Top Discriminatory PCA Features",
        points=False
    )
    fig_plotly.update_layout(template="plotly_white")
    plotly_path = output_dir / "06_key_anonymized_features_distribution.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_correlation_heatmap(
    df: pd.DataFrame, output_dir: Path
) -> str:
    """Chart 7: Correlation matrix of top correlated features + Amount + Time + Class."""
    top_corr = compute_correlations_with_target(df)
    selected_cols = list(top_corr.head(5).index) + list(top_corr.tail(5).index) + ["Time", "Amount", "Class"]
    corr_matrix = df[selected_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-0.6, vmax=0.6)
    
    ax.set_xticks(range(len(selected_cols)))
    ax.set_yticks(range(len(selected_cols)))
    ax.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(selected_cols, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Correlation Heatmap of Key Features and Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    mpl_path = output_dir / "07_correlation_heatmap.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    return str(mpl_path)


def generate_all_visualizations(
    df: pd.DataFrame, output_dir: str | Path = "reports/figures"
) -> List[str]:
    """Generate and save all 7 charts (both PNG and HTML where applicable)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated_files = []
    
    p1_png, p1_html = plot_class_distribution(df, out_path)
    p2_png, p2_html = plot_amount_distribution(df, out_path)
    p3_png, p3_html = plot_transaction_frequency_time(df, out_path)
    p4_png, p4_html = plot_fraud_rate_by_amount_bands(df, out_path)
    p5_png, p5_html = plot_top_feature_correlations(df, out_path)
    p6_png, p6_html = plot_key_anonymized_features_distribution(df, out_path)
    p7_png = plot_correlation_heatmap(df, out_path)

    generated_files.extend([
        p1_png, p1_html,
        p2_png, p2_html,
        p3_png, p3_html,
        p4_png, p4_html,
        p5_png, p5_html,
        p6_png, p6_html,
        p7_png
    ])
    return generated_files
