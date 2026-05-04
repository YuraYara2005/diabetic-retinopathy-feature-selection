import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def render_charts(data_1, data_2, algo_1, algo_2):
    # Enforce dark mode styling for the Matplotlib charts to match the theme
    plt.style.use('dark_background')
    TOTAL_FEATURES = 2048

    # --- CHART 1: CONVERGENCE ---
    st.subheader("📈 Convergence Comparison: Fitness Over Generations")
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(data_1["histories"].mean(axis=0), label=algo_1.upper(), linewidth=2.5, color="#4b72fa")
    ax1.plot(data_2["histories"].mean(axis=0), label=algo_2.upper(), linewidth=2.5, color="#7ee081")

    ax1.set_ylabel("Fitness (Accuracy)")
    ax1.set_xlabel("Generations")
    ax1.legend(loc="lower right")

    ax1.grid(color='#3a3d46', linestyle='-', linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    st.pyplot(fig1, transparent=True)
    st.markdown("---")

    # --- FEATURE REDUCTION RATE (NEW & EXPLICIT) ---
    st.subheader("✂️ Feature Reduction Rate")
    col_a, col_b = st.columns(2)

    feat_1 = data_1['n_features'].mean()
    feat_2 = data_2['n_features'].mean()
    red_1 = ((TOTAL_FEATURES - feat_1) / TOTAL_FEATURES) * 100
    red_2 = ((TOTAL_FEATURES - feat_2) / TOTAL_FEATURES) * 100

    with col_a:
        st.metric(f"{algo_1.upper()}", f"{feat_1:.0f} / {TOTAL_FEATURES}", f"-{red_1:.1f}% Reduction",
                  delta_color="inverse")
    with col_b:
        st.metric(f"{algo_2.upper()}", f"{feat_2:.0f} / {TOTAL_FEATURES}", f"-{red_2:.1f}% Reduction",
                  delta_color="inverse")

    fig4, ax4 = plt.subplots(figsize=(10, 2))
    ax4.barh([algo_2.upper(), algo_1.upper()], [feat_2, feat_1], color=['#7ee081', '#4b72fa'], alpha=0.8)
    ax4.axvline(TOTAL_FEATURES, color='white', linestyle='--', label='Original Features (2048)')
    ax4.set_xlabel("Number of Selected Features")
    ax4.legend(loc='lower right')
    for spine in ax4.spines.values(): spine.set_visible(False)
    st.pyplot(fig4, transparent=True)

    st.markdown("---")

    # --- PSO GBEST VS PBEST CHECK (SAFE DYNAMIC LOAD) ---
    base_dir = Path(__file__).resolve().parent.parent.parent
    algo_1_key = data_1.get("algo_name", "")
    algo_2_key = data_2.get("algo_name", "")

    def plot_pbest_if_exists(algo_key, algo_display_name, data):
        # Look for the secret pbest files we injected during the run
        pbest_files = list((base_dir / "results" / algo_key).glob("pbest_run_*.npy"))
        if len(pbest_files) > 0:
            pbests = [np.load(f) for f in pbest_files]
            min_len = min(len(p) for p in pbests)
            pbests = [p[:min_len] for p in pbests]
            mean_pbest = np.mean(pbests, axis=0)
            mean_gbest = data["histories"].mean(axis=0)[:min_len]

            st.subheader(f"🧠 {algo_display_name}: Swarm Learning (gBest vs pBest)")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(mean_gbest, label="Global Best (gBest)", color="#4b72fa", linewidth=2.5)
            ax.plot(mean_pbest, label="Avg Personal Best (pBest)", color="#7ee081", linestyle="--", linewidth=2.5)
            ax.set_ylabel("Fitness Score")
            ax.set_xlabel("Generations")
            ax.legend(loc="lower right")
            ax.grid(color='#3a3d46', linestyle='-', linewidth=0.5)
            for spine in ax.spines.values(): spine.set_visible(False)
            st.pyplot(fig, transparent=True)
            st.markdown("<br>", unsafe_allow_html=True)

    plot_pbest_if_exists(algo_1_key, algo_1, data_1)
    plot_pbest_if_exists(algo_2_key, algo_2, data_2)

    st.markdown("---")

    # --- Create a 2-Column Layout for the older advanced charts ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Fitness Stability")
        st.caption("Distribution across all independent runs")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bp = ax2.boxplot([data_1['fitness'], data_2['fitness']], labels=[algo_1.upper(), algo_2.upper()],
                         patch_artist=True, widths=0.5)
        colors = ['#4b72fa', '#7ee081']
        for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color); patch.set_alpha(0.7)
        for median in bp['medians']: median.set(color='white', linewidth=2)
        ax2.set_ylabel("Fitness Score")
        ax2.grid(axis='y', color='#3a3d46', linestyle='-', linewidth=0.5)
        for spine in ax2.spines.values(): spine.set_visible(False)
        st.pyplot(fig2, transparent=True)

    with col2:
        st.subheader("🎯 Accuracy vs. Feature Count")
        st.caption("Lower left is better for features, Top is better for accuracy")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.scatter(data_1['n_features'], data_1['fitness'], color="#4b72fa", alpha=0.7, edgecolors='white',
                    label=algo_1.upper())
        ax3.scatter(data_2['n_features'], data_2['fitness'], color="#7ee081", alpha=0.7, edgecolors='white',
                    label=algo_2.upper())
        ax3.set_xlabel("Number of Selected Features")
        ax3.set_ylabel("Fitness Score")
        ax3.legend(loc="best")
        ax3.grid(color='#3a3d46', linestyle='-', linewidth=0.5)
        for spine in ax3.spines.values(): spine.set_visible(False)
        st.pyplot(fig3, transparent=True)