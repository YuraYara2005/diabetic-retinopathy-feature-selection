import streamlit as st

def render_scorecards(data_1, data_2, algo_1, algo_2):
    """
    Takes the aggregated result dictionaries and renders top-level metric cards.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🏆 {algo_1.upper()} Results")

        fit_mean_1 = data_1['fitness'].mean()
        feat_mean_1 = data_1['n_features'].mean()

        # ULTRA-SAFE: Checking dictionary keys, NO .columns!
        acc_mean_1 = data_1['accuracy'].mean() if 'accuracy' in data_1 else 0.0

        st.metric("Avg Best Fitness", f"{fit_mean_1 * 100:.2f}%")
        st.metric("Avg Accuracy", f"{acc_mean_1 * 100:.2f}%" if acc_mean_1 > 0 else "N/A")
        st.metric("Avg Features Kept", f"{feat_mean_1:.0f} / 2048")

    with col2:
        st.subheader(f"🏆 {algo_2.upper()} Results")

        fit_mean_2 = data_2['fitness'].mean()
        feat_mean_2 = data_2['n_features'].mean()

        # ULTRA-SAFE: Checking dictionary keys, NO .columns!
        acc_mean_2 = data_2['accuracy'].mean() if 'accuracy' in data_2 else 0.0

        st.metric("Avg Best Fitness", f"{fit_mean_2 * 100:.2f}%")
        st.metric("Avg Accuracy", f"{acc_mean_2 * 100:.2f}%" if acc_mean_2 > 0 else "N/A")
        st.metric("Avg Features Kept", f"{feat_mean_2:.0f} / 2048")