import sys
from pathlib import Path
import time
import streamlit as st

# --- 1. BULLETPROOF PATHING ---
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# --- 2. IMPORTS ---
from src.algorithms.experiment_runner import run_experiments
from results.analyze_results import analyze_algorithm_runs
from src.ui.sidebar import render_sidebar
from src.ui.visualizations import render_charts
from src.ui.state_manager import init_session_state, save_experiment_results

# --- 3. PAGE SETUP & CSS ---
st.set_page_config(page_title="APTOS Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 4. INIT MEMORY ---
init_session_state()

# --- 5. MAIN UI EXECUTION ---
st.title("APTOS Feature Selection Dashboard")
# --- MAIN UI EXECUTION ---
algo_1, algo_2, classifier_ui, pop_size, generations, run_clicked, btn_placeholder = render_sidebar(
    data_1=st.session_state.data_1 if st.session_state.has_run else None,
    data_2=st.session_state.data_2 if st.session_state.has_run else None,
    algo_1_name=st.session_state.get('last_algo_1', 'Algorithm A'),
    algo_2_name=st.session_state.get('last_algo_2', 'Algorithm B')
)

# 1. TRANSLATOR: Map UI names to backend keys
model_mapping = {
    "KNN": "knn",
    "Random Forest": "random_forest",
    "SVM": "svm"
}

if run_clicked:
    # Get the technical name for the backend
    classifier_backend = model_mapping.get(classifier_ui)

    btn_placeholder.button("⏳ Processing...", disabled=True, use_container_width=True)

    st.session_state.last_algo_1 = algo_1
    st.session_state.last_algo_2 = algo_2

    progress_text = f"Simulating {generations} generations..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)

    with st.spinner("Compiling Final Results..."):
        # 2. FIX: Pass 'classifier_backend' instead of the raw UI string
        run_experiments(
            algo_name=algo_1.lower(),
            classifier_override=classifier_backend,
            pop_override=pop_size,
            gen_override=generations
        )

        if algo_1 != algo_2:
            run_experiments(
                algo_name=algo_2.lower(),
                classifier_override=classifier_backend,
                pop_override=pop_size,
                gen_override=generations
            )

        data_1 = analyze_algorithm_runs(algo_1.lower())
        data_2 = analyze_algorithm_runs(algo_2.lower())
        save_experiment_results(data_1, data_2)

    my_bar.empty()
    st.rerun()

# --- 6. RENDER CHARTS ---
if st.session_state.has_run:
    render_charts(st.session_state.data_1, st.session_state.data_2,
                  st.session_state.last_algo_1, st.session_state.last_algo_2)
else:
    st.info("👈 Select your algorithms in the sidebar and click 'Run Comparison' to start!")