from __future__ import annotations

import pandas as pd
import pytest

from execsim.ml.paper.reports import _to_latex


def test_to_latex_preserves_all_rows_when_render_limit_is_low() -> None:
    frame = pd.DataFrame(
        {
            "row": ["row-1", "row-2", "row-3"],
            "value": [11, 22, 33],
        }
    )
    original = frame.copy(deep=True)

    with pd.option_context("styler.render.max_elements", 1):
        rendered = _to_latex(frame)

    assert "..." not in rendered
    for row in frame.itertuples(index=False, name=None):
        assert f"{row[0]} & {row[1]}" in rendered
    pd.testing.assert_frame_equal(frame, original)


def test_to_latex_restores_the_callers_render_limit() -> None:
    frame = pd.DataFrame({"row": ["row-1"], "value": [11]})
    default_limit = pd.get_option("styler.render.max_elements")

    with pd.option_context("styler.render.max_elements", 1):
        _to_latex(frame)
        assert pd.get_option("styler.render.max_elements") == 1

    assert pd.get_option("styler.render.max_elements") == default_limit


def test_to_latex_restores_options_when_serialization_raises(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("serialization failure")

    monkeypatch.setattr(pd.DataFrame, "to_latex", fail)
    with pd.option_context("styler.render.max_elements", 1):
        with pytest.raises(RuntimeError, match="serialization failure"):
            _to_latex(pd.DataFrame({"value": [1, 2, 3]}))
        assert pd.get_option("styler.render.max_elements") == 1
