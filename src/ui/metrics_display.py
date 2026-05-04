# src/ui/metrics_display.py
import streamlit as st


def render_scorecards(data_1, data_2, algo_1, algo_2):
    """
    Takes the aggregated result dictionaries and renders top-level metric cards.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🏆 {algo_1.upper()} Results")

        # Calculate means
        fit_mean_1 = data_1['fitness'].mean()
        feat_mean_1 = data_1['n_features'].mean()

        st.metric("Avg Best Fitness", f"{fit_mean_1:.4f}")
        st.metric("Avg Features Kept", f"{feat_mean_1:.1f} / 2048")

    with col2:
        st.subheader(f"🏆 {algo_2.upper()} Results")

        # Calculate means
        fit_mean_2 = data_2['fitness'].mean()
        feat_mean_2 = data_2['n_features'].mean()

        st.metric("Avg Best Fitness", f"{fit_mean_2:.4f}")
        st.metric("Avg Features Kept", f"{feat_mean_2:.1f} / 2048")