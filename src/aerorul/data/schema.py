"""Column names, sensor descriptions, and per-subset constants for the CMAPSS dataset."""

from __future__ import annotations

INDEX_COLS = ["unit_number", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# conditions / fault modes per subset, from the CMAPSS readme
SUBSET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": 1, "train_units": 100, "test_units": 100},
    "FD002": {"conditions": 6, "fault_modes": 1, "train_units": 260, "test_units": 259},
    "FD003": {"conditions": 1, "fault_modes": 2, "train_units": 100, "test_units": 100},
    "FD004": {"conditions": 6, "fault_modes": 2, "train_units": 248, "test_units": 249},
}

SENSOR_DESCRIPTIONS = {
    "s_1": ("Total temperature at fan inlet", "degR"),
    "s_2": ("Total temperature at LPC outlet", "degR"),
    "s_3": ("Total temperature at HPC outlet", "degR"),
    "s_4": ("Total temperature at LPT outlet", "degR"),
    "s_5": ("Pressure at fan inlet", "psia"),
    "s_6": ("Total pressure in bypass-duct", "psia"),
    "s_7": ("Total pressure at HPC outlet", "psia"),
    "s_8": ("Physical fan speed", "rpm"),
    "s_9": ("Physical core speed", "rpm"),
    "s_10": ("Engine pressure ratio (P50/P2)", "-"),
    "s_11": ("Static pressure at HPC outlet", "psia"),
    "s_12": ("Ratio of fuel flow to Ps30", "pps/psia"),
    "s_13": ("Corrected fan speed", "rpm"),
    "s_14": ("Corrected core speed", "rpm"),
    "s_15": ("Bypass ratio", "-"),
    "s_16": ("Burner fuel-air ratio", "-"),
    "s_17": ("Bleed enthalpy", "-"),
    "s_18": ("Demanded fan speed", "rpm"),
    "s_19": ("Demanded corrected fan speed", "rpm"),
    "s_20": ("HPT coolant bleed", "lbm/s"),
    "s_21": ("LPT coolant bleed", "lbm/s"),
}

# Sensors that are constant (or ~constant) in FD001/FD003 (single operating condition) and
# carry no useful signal there. Kept in the loader but dropped by default during feature
# engineering for single-condition subsets. Multi-condition subsets (FD002/FD004) do vary
# on these because operating condition itself changes them, so don't drop there.
LOW_VARIANCE_SENSORS_SINGLE_CONDITION = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
