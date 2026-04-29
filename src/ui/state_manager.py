# src/ui/state_manager.py
import streamlit as st


def init_session_state():
    """
    Initializes memory variables so Streamlit doesn't forget our data
    when the user clicks different tabs or buttons.
    """
    # Remembers IF we have run an experiment yet
    if 'has_run' not in st.session_state:
        st.session_state.has_run = False

    # Stores the data so we don't have to keep reading JSONs from the hard drive
    if 'data_1' not in st.session_state:
        st.session_state.data_1 = None

    if 'data_2' not in st.session_state:
        st.session_state.data_2 = None


def save_experiment_results(data_1, data_2):
    """
    Saves the loaded data into the session memory.
    """
    st.session_state.data_1 = data_1
    st.session_state.data_2 = data_2
    st.session_state.has_run = True