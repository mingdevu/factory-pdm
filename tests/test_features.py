import math

import pandas as pd

from model.features import FEATURE_COLUMNS, add_features, failure_type_label, reading_to_frame

SAMPLE = {"product_type": "M", "air_temp_k": 300.0, "process_temp_k": 310.5,
          "rpm": 1500.0, "torque_nm": 40.0, "tool_wear_min": 100.0}


def test_derived_features_have_physical_meaning():
    row = add_features(pd.DataFrame([SAMPLE])).iloc[0]
    assert row["temp_diff_k"] == 10.5
    assert math.isclose(row["power_w"], 40.0 * 1500 * 2 * math.pi / 60)
    assert row["strain_min_nm"] == 4000.0
    assert (row["type_M"], row["type_L"], row["type_H"]) == (1, 0, 0)


def test_reading_to_frame_column_order_matches_training():
    assert list(reading_to_frame(SAMPLE).columns) == FEATURE_COLUMNS


def test_failure_type_label_priority_and_unknown():
    frame = pd.DataFrame([
        {"machine_failure": 0, "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0},
        {"machine_failure": 1, "TWF": 0, "HDF": 1, "PWF": 1, "OSF": 0, "RNF": 0},
        {"machine_failure": 1, "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0},
    ])
    assert failure_type_label(frame).tolist() == ["NONE", "HDF", "UNKNOWN"]
