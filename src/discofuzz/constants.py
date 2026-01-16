from enum import Enum

STRATEGIES = [
    None,
    "mean",
    "intersection+mean",
    "intersection+union",
    "intersection+intersection",
    "selective_intersection+mean",
    "selective_intersection+union",
    "selective_intersection+intersection+mean",
]

class SIMILARITY_METRICS(Enum):
    COS="cos"
    W1="wasserstein-1"
    W2="wasserstein-2"
    Q="quantum"