import streamlit as st
import matplotlib.pyplot as plt


def render_charts(data_1, data_2, algo_1, algo_2):
    # Enforce dark mode styling for the Matplotlib charts to match the theme
    plt.style.use('dark_background')

    # --- CHART 1: CONVERGENCE ---
    st.subheader("Convergence Comparison: Fitness Over Generations")
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    # Using the exact colors from the mockup
    ax1.plot(data_1["histories"].mean(axis=0), label=algo_1.upper(), linewidth=2.5, color="#4b72fa")
    ax1.plot(data_2["histories"].mean(axis=0), label=algo_2.upper(), linewidth=2.5, color="#7ee081")

    ax1.set_ylabel("Fitness (Accuracy)")
    ax1.legend(loc="lower right")

    # Clean up the grid so it looks like a modern web dashboard, not a textbook
    ax1.grid(color='#3a3d46', linestyle='-', linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    st.pyplot(fig1, transparent=True)  # Transparent keeps the Streamlit background

    st.markdown("---")

    # --- CHART 2: BAR CHART ---
    st.subheader("Final Feature Reduction (Total Selected)")
    fig2, ax2 = plt.subplots(figsize=(10, 3))

    algos = [algo_1.upper(), algo_2.upper()]
    means = [data_1['n_features'].mean(), data_2['n_features'].mean()]

    bars = ax2.bar(algos, means, color=['#4b72fa', '#7ee081'], width=0.4)
    ax2.set_ylabel("Number of Features")

    # Grid cleanup
    ax2.grid(axis='y', color='#3a3d46', linestyle='-', linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    st.pyplot(fig2, transparent=True)