from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


SESSION_SELECTED_RUN_ID = "selected_run_id"
SESSION_SELECTED_EXPERIMENT = "selected_experiment"
SESSION_SELECTED_WINDOW = "selected_window"
SESSION_SELECTED_SPLIT = "selected_signal_split"


@dataclass(frozen=True)
class SelectionDefaults:
    run_id: Optional[str] = None
    experiment: Optional[str] = None
    window: Optional[int] = None
    split: Optional[str] = None


def ensure_session_keys() -> None:
    st.session_state.setdefault(SESSION_SELECTED_RUN_ID, None)
    st.session_state.setdefault(SESSION_SELECTED_EXPERIMENT, None)
    st.session_state.setdefault(SESSION_SELECTED_WINDOW, None)
    st.session_state.setdefault(SESSION_SELECTED_SPLIT, None)


def get_selection_defaults() -> SelectionDefaults:
    ensure_session_keys()
    return SelectionDefaults(
        run_id=st.session_state.get(SESSION_SELECTED_RUN_ID),
        experiment=st.session_state.get(SESSION_SELECTED_EXPERIMENT),
        window=st.session_state.get(SESSION_SELECTED_WINDOW),
        split=st.session_state.get(SESSION_SELECTED_SPLIT),
    )


def set_selection(
    *,
    run_id: str | None = None,
    experiment: str | None = None,
    window: int | None = None,
    split: str | None = None,
) -> None:
    ensure_session_keys()
    if run_id is not None:
        st.session_state[SESSION_SELECTED_RUN_ID] = run_id
    if experiment is not None:
        st.session_state[SESSION_SELECTED_EXPERIMENT] = experiment
    if window is not None:
        st.session_state[SESSION_SELECTED_WINDOW] = int(window)
    if split is not None:
        st.session_state[SESSION_SELECTED_SPLIT] = str(split)


def clear_invalid_selection(*, valid_run_ids: set[str] | None = None, valid_experiments: set[str] | None = None) -> None:
    ensure_session_keys()

    if valid_run_ids is not None:
        cur = st.session_state.get(SESSION_SELECTED_RUN_ID)
        if cur is not None and cur not in valid_run_ids:
            st.session_state[SESSION_SELECTED_RUN_ID] = None

    if valid_experiments is not None:
        cur = st.session_state.get(SESSION_SELECTED_EXPERIMENT)
        if cur is not None and cur not in valid_experiments:
            st.session_state[SESSION_SELECTED_EXPERIMENT] = None

