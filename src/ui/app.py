import sys
import time
from pathlib import Path

import streamlit as st

# ── Bulletproof path setup ───────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# ── Project imports ──────────────────────────────────────────────────────────
from src.algorithms.experiment_runner import run_experiments
from src.ui.sidebar import render_sidebar
from src.ui.visualizations import render_charts
from src.ui.state_manager import init_session_state, save_experiment_results

# Try to import analyzer — graceful fallback if results don't exist yet
try:
    from results.analyze_results import analyze_algorithm_runs
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "APTOS Feature Selection Dashboard",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Session state ────────────────────────────────────────────────────────────
init_session_state()

# ── Classifier name mapping  (UI label → backend key) ───────────────────────
MODEL_MAP = {
    "KNN":           "knn",
    "Random Forest": "random_forest",
    "SVM":           "svm",
}

# 💥 TRANSLATE UI DROPDOWNS TO SAFE FOLDER NAMES
ALGO_MAP = {
    "Dummy": "dummy",
    "PSO (Linear Decay)": "pso_linear",
    "PSO (High Inertia)": "pso_high",
    "PSO (Low Inertia)": "pso_low",
    "GA (Tournament + Uniform)": "ga_tourn_uni",
    "GA (Roulette + One-Point)": "ga_roul_one",
    "GA (Tournament + One-Point)": "ga_tourn_one",
    "GA (Roulette + Uniform)": "ga_roul_uni"
}

# 💥 SHORT NAMES FOR THE HTML METRICS DISPLAY AND CHARTS
ALGO_DISPLAY = {
    "dummy": "Dummy",
    "pso_linear": "PSO (Linear)",
    "pso_high": "PSO (High)",
    "pso_low": "PSO (Low)",
    "ga_tourn_uni": "GA (Tourn+Uni)",
    "ga_roul_one":  "GA (Roul+One)",
    "ga_tourn_one": "GA (Tourn+One)",
    "ga_roul_uni":  "GA (Roul+Uni)",
}

# ────────────────────────────────────────────────────────────────────────────
# TITLE
# ────────────────────────────────────────────────────────────────────────────
st.title("🧬 APTOS Feature Selection Dashboard")
st.caption("Evolutionary Algorithm Comparison — Diabetic Retinopathy Dataset")

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
(
    algo_1, algo_2,
    classifier_ui,
    pop_size, generations,
    num_runs,
    data_subset,
    run_clicked,
    btn_placeholder,
) = render_sidebar(
    data_1      = st.session_state.data_1 if st.session_state.has_run else None,
    data_2      = st.session_state.data_2 if st.session_state.has_run else None,
    algo_1_name = st.session_state.get("last_algo_1", "Algorithm A"),
    algo_2_name = st.session_state.get("last_algo_2", "Algorithm B"),
)

# ────────────────────────────────────────────────────────────────────────────
# RUN EXPERIMENTS
# ────────────────────────────────────────────────────────────────────────────
if run_clicked:
    classifier_backend = MODEL_MAP.get(classifier_ui, "knn")

    # Safely convert UI string to the backend key using the map
    algo_1_key = ALGO_MAP.get(algo_1, "dummy")
    algo_2_key = ALGO_MAP.get(algo_2, "dummy")

    # Validate against all allowed variations
    valid_algos = {
        "dummy", "pso_linear", "pso_high", "pso_low",
        "ga_tourn_uni", "ga_roul_one", "ga_tourn_one", "ga_roul_uni"
    }
    if algo_1_key not in valid_algos or algo_2_key not in valid_algos:
        st.error(f"Unknown algorithm selected. Choose from: {', '.join(valid_algos)}")
        st.stop()

    # Save names for chart labels
    st.session_state.last_algo_1 = ALGO_DISPLAY.get(algo_1_key, algo_1_key.upper())
    st.session_state.last_algo_2 = ALGO_DISPLAY.get(algo_2_key, algo_2_key.upper())

    # Disable the button while running
    btn_placeholder.button("⏳ Running...", disabled=True, use_container_width=True)

    st.markdown("---")

    # ── Run Algorithm 1 ──────────────────────────────────────────────────
    st.subheader(f"Running {st.session_state.last_algo_1}...")
    progress_bar_1 = st.progress(0, text="Starting runs...")

    def update_progress_1(fraction: float):
        pct  = int(fraction * 100)
        runs_done = round(fraction * num_runs)
        progress_bar_1.progress(pct, text=f"Run {runs_done}/{num_runs} complete")

    try:
        run_experiments(
            algo_name           = algo_1_key,
            config_path         = "experiment_config.yaml",
            classifier_override = classifier_backend,
            pop_override        = pop_size,
            gen_override        = generations,
            runs_override       = num_runs,
            subset_override     = data_subset,
            progress_callback   = update_progress_1,
        )
        progress_bar_1.progress(100, text=f"✅ {st.session_state.last_algo_1} complete!")
    except FileNotFoundError as e:
        progress_bar_1.empty()
        st.error(f"Dataset not found for {st.session_state.last_algo_1}.\n\n{e}")
        st.info("💡 Tip: Use **Dummy** algorithm to test the UI before the dataset is ready.")
        st.stop()
    except Exception as e:
        progress_bar_1.empty()
        st.error(f"Error running {st.session_state.last_algo_1}: {e}")
        st.stop()

    # ── Run Algorithm 2 (skip if same as algo 1) ─────────────────────────
    if algo_1_key != algo_2_key:
        st.subheader(f"Running {st.session_state.last_algo_2}...")
        progress_bar_2 = st.progress(0, text="Starting runs...")

        def update_progress_2(fraction: float):
            pct = int(fraction * 100)
            runs_done = round(fraction * num_runs)
            progress_bar_2.progress(pct, text=f"Run {runs_done}/{num_runs} complete")

        try:
            run_experiments(
                algo_name           = algo_2_key,
                config_path         = "experiment_config.yaml",
                classifier_override = classifier_backend,
                pop_override        = pop_size,
                gen_override        = generations,
                runs_override       = num_runs,
                subset_override     = data_subset,
                progress_callback   = update_progress_2,
            )
            progress_bar_2.progress(100, text=f"✅ {st.session_state.last_algo_2} complete!")
        except FileNotFoundError as e:
            progress_bar_2.empty()
            st.error(f"Dataset not found for {st.session_state.last_algo_2}.\n\n{e}")
            st.info("💡 Tip: Use **Dummy** algorithm to test the UI before the dataset is ready.")
            st.stop()
        except Exception as e:
            progress_bar_2.empty()
            st.error(f"Error running {st.session_state.last_algo_2}: {e}")
            st.stop()

    # ── Load results & rerender ──────────────────────────────────────────
    if not ANALYZER_AVAILABLE:
        st.error(
            "`results/analyze_results.py` not found. "
            "Make sure that file exists and `results/__init__.py` is present."
        )
        st.stop()

    with st.spinner("📊 Compiling results..."):
        try:
            data_1 = analyze_algorithm_runs(algo_1_key)
            data_2 = analyze_algorithm_runs(algo_2_key)
            save_experiment_results(data_1, data_2)
        except Exception as e:
            st.error(f"Failed to analyze results: {e}")
            st.stop()

    st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# CHARTS (shown after first successful run)
# ────────────────────────────────────────────────────────────────────────────
if st.session_state.has_run:
    render_charts(
        st.session_state.data_1,
        st.session_state.data_2,
        st.session_state.last_algo_1,
        st.session_state.last_algo_2,
    )
else:
    st.info("👈 Select two algorithms in the sidebar and click **Run Comparison** to begin.")
    st.markdown("""
    ### 🚀 Quick Start
    - **Dummy vs Dummy** — tests the full UI pipeline instantly, no dataset needed
    - **PSO (Linear) vs GA (Tourn+Uni)** — full comparison once both algorithms and the dataset are ready
    """)