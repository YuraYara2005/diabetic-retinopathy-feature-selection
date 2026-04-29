import streamlit as st


def render_sidebar(data_1=None, data_2=None, algo_1_name="", algo_2_name=""):
    # 1. AGGRESSIVE CSS: Forces the primary button to be Blue and targets the sidebar specifically
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

    # 2. THE SLEEK METRICS (Only shows up after the first run)
    if data_1 is not None and data_2 is not None:
        fit1 = f"{data_1['fitness'].mean() * 100:.1f}%"
        feat1 = f"{data_1['n_features'].mean():.0f}"
        fit2 = f"{data_2['fitness'].mean() * 100:.1f}%"
        feat2 = f"{data_2['n_features'].mean():.0f}"

        st.sidebar.markdown("### 📊 Live Results")

        # Custom HTML injected directly into the sidebar
        custom_metrics = f"""
        <div style="background-color: #1e1e24; padding: 15px; border-radius: 8px; margin-bottom: 25px; border: 1px solid #333;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div>
                    <div style="color: #a0a0a0; font-size: 11px; font-weight: bold; letter-spacing: 1px;">{algo_1_name.upper()} FITNESS</div>
                    <div style="color: #ffffff; font-size: 20px; font-weight: bold;">{fit1}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #a0a0a0; font-size: 11px; font-weight: bold; letter-spacing: 1px;">FEATURES</div>
                    <div style="color: #4b72fa; font-size: 20px; font-weight: bold;">{feat1}</div>
                </div>
            </div>
            <div style="height: 1px; background-color: #333333; margin-bottom: 15px;"></div>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="color: #a0a0a0; font-size: 11px; font-weight: bold; letter-spacing: 1px;">{algo_2_name.upper()} FITNESS</div>
                    <div style="color: #ffffff; font-size: 20px; font-weight: bold;">{fit2}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #a0a0a0; font-size: 11px; font-weight: bold; letter-spacing: 1px;">FEATURES</div>
                    <div style="color: #7ee081; font-size: 20px; font-weight: bold;">{feat2}</div>
                </div>
            </div>
        </div>
        """
        st.sidebar.markdown(custom_metrics, unsafe_allow_html=True)

    # 3. EXPERIMENT CONTROLS
    st.sidebar.header("🎛️ Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        algo_1 = st.selectbox("Algorithm A", ["dummy", "GA", "PSO", "ACO"], index=1)
    with col2:
        algo_2 = st.selectbox("Algorithm B", ["dummy", "GA", "PSO", "ACO"], index=2)

    st.sidebar.markdown("---")

    # Sleek segmented tags instead of circles
    classifier = st.sidebar.pills(
        "Classifier Engine",
        ["KNN", "Random Forest", "SVM"],
        default="Random Forest"
    )

    pop_size = st.sidebar.slider("Population Size", 10, 100, 50, 10)
    generations = st.sidebar.slider("Generations", 10, 200, 100, 10)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # 4. THE BUTTON PLACEHOLDER
    btn_placeholder = st.sidebar.empty()
    run_clicked = btn_placeholder.button("🚀 Run Comparison", use_container_width=True)

    return algo_1, algo_2, classifier, pop_size, generations, run_clicked, btn_placeholder