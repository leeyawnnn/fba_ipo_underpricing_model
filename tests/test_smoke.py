"""
Smoke tests — basic sanity checks that the source modules import and their
core functions behave as documented on trivial inputs.

Run with:
    python -m pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

def test_setup_logging_returns_logger():
    from src.utils import setup_logging
    import logging
    logger = setup_logging("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_safe_divide():
    from src.utils import safe_divide
    assert safe_divide(10, 2) == 5.0
    assert np.isnan(safe_divide(10, 0))
    assert safe_divide(10, 0, default=-1) == -1


def test_retry_succeeds_on_first_attempt():
    from src.utils import retry
    calls = []

    @retry(max_attempts=3)
    def good():
        calls.append(1)
        return 42

    result = good()
    assert result == 42
    assert len(calls) == 1


def test_retry_raises_after_max_attempts():
    from src.utils import retry
    calls = []

    @retry(max_attempts=2, initial_wait=0)
    def always_fails():
        calls.append(1)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Text feature tests
# ---------------------------------------------------------------------------

def test_tokenise_basic():
    from src.text_features import tokenise
    tokens = tokenise("The firm's revenue grew 15% in Q3.")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    # Numbers-only tokens should be removed
    assert "15" not in tokens


def test_gunning_fog_empty():
    from src.text_features import gunning_fog_index
    result = gunning_fog_index("")
    assert np.isnan(result)


def test_gunning_fog_simple():
    from src.text_features import gunning_fog_index
    text = "The company sells widgets. Revenue grew last year."
    result = gunning_fog_index(text)
    assert isinstance(result, float)
    assert result >= 0


def test_compute_lm_ratios_empty():
    from src.text_features import compute_lm_ratios
    lm_dict = {"lm_negative": {"BAD", "WEAK"}, "lm_positive": {"GOOD", "STRONG"}}
    result = compute_lm_ratios("", lm_dict)
    for v in result.values():
        assert np.isnan(v)


def test_compute_lm_ratios_known():
    from src.text_features import compute_lm_ratios
    lm_dict = {"lm_negative": {"BAD", "WEAK"}}
    result = compute_lm_ratios("the company had bad results weak margins", lm_dict)
    assert "lm_negative_ratio" in result
    assert result["lm_negative_ratio"] == pytest.approx(2 / 7, abs=0.01)


def test_prospectus_uniqueness_shape():
    from src.text_features import compute_prospectus_uniqueness
    texts = [
        "The company operates in the technology sector and sells software.",
        "Revenue grew by ten percent last year driven by enterprise sales.",
        "Risk factors include competition and macroeconomic uncertainty.",
        "The firm focuses on healthcare and medical device distribution.",
        "Software revenue increased due to cloud subscription growth.",
    ]
    sectors = ["Tech", "Tech", "Tech", "Health", "Tech"]
    result = compute_prospectus_uniqueness(texts, sectors)
    assert result.shape == (5,)
    assert np.all(result >= 0) and np.all(result <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

def test_add_calendar_features():
    from src.feature_engineering import add_calendar_features
    df = pd.DataFrame({"ipo_date": pd.to_datetime(["2022-03-15", "2023-07-04", "2024-12-31"])})
    result = add_calendar_features(df)
    assert "ipo_year" in result.columns
    assert "ipo_quarter" in result.columns
    assert "is_quarter_end_month" in result.columns
    # March is quarter-end
    assert result.loc[0, "is_quarter_end_month"] == 1
    # July is not quarter-end
    assert result.loc[1, "is_quarter_end_month"] == 0


def test_add_deal_features_log_columns():
    from src.feature_engineering import add_deal_features
    df = pd.DataFrame({
        "offer_size_m": [50.0, 200.0, 1000.0],
        "shares_offered": [5_000_000, 20_000_000, 100_000_000],
        "lead_underwriter": ["Goldman Sachs", "JPMorgan", "Boutique LLC"],
    })
    result = add_deal_features(df)
    assert "log_offer_size" in result.columns
    assert "log_shares_offered" in result.columns
    assert (result["log_offer_size"] > 0).all()


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

def test_missing_value_report():
    from src.preprocessing import missing_value_report
    df = pd.DataFrame({"a": [1, 2, None], "b": [None, None, 3]})
    report = missing_value_report(df)
    assert "pct_missing" in report.columns
    # 'b' should appear first (more missing)
    assert report.iloc[0]["column"] == "b"


def test_winsorise_target():
    from src.preprocessing import winsorise_target
    data = pd.DataFrame({"underpricing": list(range(100)) + [1000, -500]})
    result = winsorise_target(data)
    assert "winsorized_underpricing" in result.columns
    assert result["winsorized_underpricing"].max() < 1000
    assert result["winsorized_underpricing"].min() > -500


def test_drop_incomplete():
    from src.preprocessing import drop_incomplete
    df = pd.DataFrame({
        "ticker": ["A", "B", None],
        "ipo_date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
        "offer_price": [10.0, None, 15.0],
        "first_day_close": [12.0, 14.0, 16.0],
    })
    result = drop_incomplete(df)
    assert len(result) == 1  # only row A has all mandatory columns filled


# ---------------------------------------------------------------------------
# Models tests
# ---------------------------------------------------------------------------

def test_evaluate_perfect_predictions():
    from src.models import evaluate
    y = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = evaluate(y, y, "perfect")
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["spearman_rho"] == pytest.approx(1.0, abs=1e-6)


def test_baseline_median():
    from src.models import baseline_median
    y_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y_test = np.array([0.25, 0.35])
    pred, metrics = baseline_median(y_train, y_test)
    assert np.all(pred == pytest.approx(np.median(y_train)))


def test_select_features_excludes_target():
    from src.models import select_features
    df = pd.DataFrame({
        "underpricing": [0.1], "winsorized_underpricing": [0.1],
        "ticker": ["A"], "log_offer_size": [10.0], "vix_at_pricing": [20.0],
    })
    features = select_features(df)
    assert "winsorized_underpricing" not in features
    assert "underpricing" not in features
    assert "ticker" not in features
    assert "log_offer_size" in features
