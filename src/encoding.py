# Encoding of the UJIIndoorLoc "not detected" value, shared by the notebook and the tests.
import numpy as np

NOT_DETECTED = 100  # value the dataset uses when an access point was not heard in a scan
NO_SIGNAL_DBM = -105  # just weaker than the weakest real reading (-104 dBm)


def encode_not_detected(df, wap_cols):
    # Fixed rule (nothing is learned from data): 100 -> -105 dBm. Returns a copy.
    out = df.copy()
    out[wap_cols] = out[wap_cols].replace(NOT_DETECTED, NO_SIGNAL_DBM)
    return out


def to_positive(X):
    # -105 dBm (not detected) -> 0, 0 dBm (strongest) -> 1. A fixed formula, so nothing is fitted.
    return (np.asarray(X, dtype="float32") - NO_SIGNAL_DBM) / (-NO_SIGNAL_DBM)
