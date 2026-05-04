import streamlit as st
import matplotlib.pyplot as plt


def render_charts(data_1, data_2, algo_1, algo_2):
    # Enforce dark mode styling for the Matplotlib charts to match the theme
    plt.style.use('dark_background')

    # --- CHART 1: CONVERGENCE ---
    st.subheader("📈 Convergence Comparison: Fitness Over Generations")
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    # Using the exact colors from the mockup
    ax1.plot(data_1["histories"].mean(axis=0), label=algo_1.upper(), linewidth=2.5, color="#4b72fa")
    ax1.plot(data_2["histories"].mean(axis=0), label=algo_2.upper(), linewidth=2.5, color="#7ee081")

    ax1.set_ylabel("Fitness (Accuracy)")
    ax1.set_xlabel("Generations")
    ax1.legend(loc="lower right")

    # Clean up the grid
    ax1.grid(color='#3a3d46', linestyle='-', linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    st.pyplot(fig1, transparent=True)
    st.markdown("---")

    # --- Create a 2-Column Layout for the new advanced charts ---
    col1, col2 = st.columns(2)

    with col1:
        # --- CHART 2: THE STABILITY BOXPLOT ---
        st.subheader("📦 Fitness Stability")
        st.caption("Distribution across all independent runs")
        fig2, ax2 = plt.subplots(figsize=(5, 4))

        # Create boxplots
        bp = ax2.boxplot([data_1['fitness'], data_2['fitness']],
                         labels=[algo_1.upper(), algo_2.upper()],
                         patch_artist=True,
                         widths=0.5)

        # Color them to match your UI
        colors = ['#4b72fa', '#7ee081']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set(color='white', linewidth=2)

        ax2.set_ylabel("Fitness Score")
        ax2.grid(axis='y', color='#3a3d46', linestyle='-', linewidth=0.5)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        st.pyplot(fig2, transparent=True)

    with col2:
        # --- CHART 3: THE TRADE-OFF SCATTER PLOT ---
        st.subheader("🎯 Accuracy vs. Feature Count")
        st.caption("Lower left is better for features, Top is better for accuracy")
        fig3, ax3 = plt.subplots(figsize=(5, 4))

        # Scatter plot for all runs
        ax3.scatter(data_1['n_features'], data_1['fitness'],
                    color="#4b72fa", alpha=0.7, edgecolors='white', label=algo_1.upper())
        ax3.scatter(data_2['n_features'], data_2['fitness'],
                    color="#7ee081", alpha=0.7, edgecolors='white', label=algo_2.upper())

        ax3.set_xlabel("Number of Selected Features")
        ax3.set_ylabel("Fitness Score")
        ax3.legend(loc="best")

        ax3.grid(color='#3a3d46', linestyle='-', linewidth=0.5)
        for spine in ax3.spines.values():
            spine.set_visible(False)

        st.pyplot(fig3, transparent=True)

    st.markdown("---")

    # --- CHART 4: BAR CHART ---
    st.subheader("📉 Average Feature Reduction")
    fig4, ax4 = plt.subplots(figsize=(10, 2.5))  # Made slightly shorter to fit the screen better

    algos = [algo_1.upper(), algo_2.upper()]
    means = [data_1['n_features'].mean(), data_2['n_features'].mean()]

    ax4.bar(algos, means, color=['#4b72fa', '#7ee081'], width=0.4)
    ax4.set_ylabel("Avg Features Kept")

    ax4.grid(axis='y', color='#3a3d46', linestyle='-', linewidth=0.5)
    for spine in ax4.spines.values():
        spine.set_visible(False)

    st.pyplot(fig4, transparent=True)