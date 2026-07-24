"""Single source of truth for the backend's vegetable list — mirrors configs/config.yaml's
`vegetables` list, duplicated here rather than parsed from YAML since app/backend/ deliberately
keeps its own dependency set separate from the research pipeline's (see config.py)."""

VEGETABLES = ["carrot", "brinjal", "pumpkin", "cabbage", "snake_gourd", "leeks"]
