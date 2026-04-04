"""
Constants for uploaded panel CSV (e.g. data/raw/final_dataset.csv).
Slab pricing/coverage live in config/model_config.py (slab_multipliers, slab_coverage).
"""

# Placeholder when disruption_type is missing in the CSV
# Must be a value the trained LabelEncoders have seen (Heavy_Rain, Cyclone, Extreme_Heat, etc.)
DISRUPTION_NONE = "Heavy_Rain"
