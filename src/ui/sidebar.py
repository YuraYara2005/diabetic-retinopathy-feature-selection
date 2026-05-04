import streamlit as st


def render_sidebar(data_1=None, data_2=None, algo_1_name="", algo_2_name=""):
    st.markdown("""
        <style>
        [data-testid="stSidebar"] div.stButton > button:first-child {
            background-color: #4b72fa !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: bold !important;
        }
        [data-testid="stSidebar"] div.stButton > button:first-child:hover {
            background-color: #3b5bdb !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if data_1 is not None and data_2 is not None:
        acc_val_1 = data_1['accuracy'].mean() if 'accuracy' in data_1 else 0.0
        acc1 = f"{acc_val_1 * 100:.1f}%" if acc_val_1 > 0 else "N/A"

        fit1 = f"{data_1['fitness'].mean() * 100:.1f}%"
        feat1 = f"{data_1['n_features'].mean():.0f}"

        acc_val_2 = data_2['accuracy'].mean() if 'accuracy' in data_2 else 0.0
        acc2 = f"{acc_val_2 * 100:.1f}%" if acc_val_2 > 0 else "N/A"

        fit2 = f"{data_2['fitness'].mean() * 100:.1f}%"
        feat2 = f"{data_2['n_features'].mean():.0f}"

        st.sidebar.markdown("### 📊 Live Results")

        # ABSOLUTELY ZERO INDENTATION ON EVERY SINGLE LINE.
        # This physically prevents Streamlit from turning it into a code block.
        custom_metrics = f"""
<div style="background-color: #1e1e24; padding: 15px; border-radius: 8px; margin-bottom: 25px; border: 1px solid #333;">
<!-- Algorithm A -->
<div style="color: #a0a0a0; font-size: 11px; font-weight: bold; letter-spacing: 1px; margin-bottom: 8px;">{algo_1_name.upper()}</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
<div>
<div style="color: #888; font-size: 10px;">ACCURACY</div>
<div style="color: #ffffff; font-size: 17px; font-weight: bold;">{acc1}</div>
</div>
<div style="text-align: center;">
<div style="color: #888; font-size: 10px;">FITNESS</div>
<div style="color: #ffffff; font-size: 17px; font-weight: bold;">{fit1}</div>
</div>
<div style="text-align: right;">
<div style="color: #888; font-size: 10px;">FEATURES</div>
<div style="color: #4b72fa; font-size: 17px; font-weight: bold;">{feat1}</div>
</div>
</div>
<div style="height: 1px; background-color: #333333; margin-bottom: 12px;"></div>
<!-- Algorithm B -->
<div style="color: #a0a0a0; font-size: 11px; font-weight: bold; letter-spacing: 1px; margin-bottom: 8px;">{algo_2_name.upper()}</div>
<div style="display: flex; justify-content: space-between;">
<div>
<div style="color: #888; font-size: 10px;">ACCURACY</div>
<div style="color: #ffffff; font-size: 17px; font-weight: bold;">{acc2}</div>
</div>
<div style="text-align: center;">
<div style="color: #888; font-size: 10px;">FITNESS</div>
<div style="color: #ffffff; font-size: 17px; font-weight: bold;">{fit2}</div>
</div>
<div style="text-align: right;">
<div style="color: #888; font-size: 10px;">FEATURES</div>
<div style="color: #7ee081; font-size: 17px; font-weight: bold;">{feat2}</div>
</div>
</div>
</div>
"""
        st.sidebar.markdown(custom_metrics, unsafe_allow_html=True)

    st.sidebar.header("🎛️ Primary Controls")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        algo_1 = st.selectbox("Algorithm A", ["dummy", "GA", "PSO"], index=1)
    with col2:
        algo_2 = st.selectbox("Algorithm B", ["dummy", "GA", "PSO"], index=2)

    classifier = st.sidebar.pills("Classifier Engine", ["KNN", "Random Forest", "SVM"], default="Random Forest")

    pop_size = st.sidebar.slider("Population Size", 10, 100, 50, 10)
    generations = st.sidebar.slider("Generations", 10, 200, 100, 10)

    num_runs = st.sidebar.slider("Independent Runs", 1, 30, 3, 1,
                                 help="Set to 1 or 3 for fast UI testing. Use 30 for final statistics.")
    data_subset = st.sidebar.slider("Data Subsample (%)", 10, 100, 100, 10,
                                    help="Use 20-30% for blazing fast testing. Use 100% for final academic runs.")

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    btn_placeholder = st.sidebar.empty()
    run_clicked = btn_placeholder.button("🚀 Run Comparison", use_container_width=True)

    return algo_1, algo_2, classifier, pop_size, generations, num_runs, data_subset, run_clicked, btn_placeholder