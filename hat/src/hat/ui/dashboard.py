"""Streamlit operator dashboard.

Run with::

    streamlit run src/hat/ui/dashboard.py

Shows: live signal histogram (U / F / N), Neocortex size, last SWS stats.
Stub for now."""

from __future__ import annotations


def main() -> None:  # pragma: no cover - UI
    import streamlit as st

    from .. import __version__
    from ..api.deps import get_loop, get_raw_log

    st.set_page_config(page_title="HAT dashboard", layout="wide")
    st.title(f"HAT operator dashboard — v{__version__}")

    loop = get_loop()
    log = get_raw_log()

    col1, col2, col3 = st.columns(3)
    col1.metric("Neocortex traces", len(loop.neocortex))
    col2.metric("Raw interactions", sum(1 for _ in log))
    col3.metric("Trainer", type(loop.trainer).__name__)


if __name__ == "__main__":  # pragma: no cover
    main()
