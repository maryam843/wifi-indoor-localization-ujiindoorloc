import numpy as np
import pandas as pd

from src.encoding import NO_SIGNAL_DBM, NOT_DETECTED, encode_not_detected, to_positive


def test_not_detected_becomes_weakest_signal():
    df = pd.DataFrame({"WAP001": [NOT_DETECTED, -50], "WAP002": [-104, NOT_DETECTED], "FLOOR": [100, 2]})
    out = encode_not_detected(df, ["WAP001", "WAP002"])
    assert out["WAP001"].tolist() == [NO_SIGNAL_DBM, -50]
    assert out["WAP002"].tolist() == [-104, NO_SIGNAL_DBM]
    assert out["FLOOR"].tolist() == [100, 2]  # columns that are not WAPs are left alone


def test_input_is_not_changed():
    df = pd.DataFrame({"WAP001": [NOT_DETECTED, -70]})
    encode_not_detected(df, ["WAP001"])
    assert df["WAP001"].tolist() == [NOT_DETECTED, -70]


def test_no_signal_is_weaker_than_any_real_reading():
    assert NO_SIGNAL_DBM < -104


def test_to_positive_scale():
    out = to_positive([[NO_SIGNAL_DBM, 0, -52.5]])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.5]])
