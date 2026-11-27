"""Explicit experimental choices; no implicit configuration in the scripts."""
from dataclasses import dataclass

BATTERIES = ("B0005", "B0006", "B0007", "B0018")
SIGNALS = ("Voltage_measured", "Current_measured", "Temperature_measured")
CAPACITY_NOMINAL_AH = 2.0
EOL_FRACTION = 0.80
N_POINTS = 128
FEATURE_NAMES = ("log_duration_s", "cycle_number")


@dataclass(frozen=True)
class Experiment:
    window: int = 10
    epochs: int = 100
    batch: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 15
    seed: int = 42
    alpha_soh: float = 0.5
    hidden: int = 32
    heads: int = 4
    dropout: float = 0.10
    aux_mode: str = "all"  # "all" | "no_cycle" | "no_duration" | "none"

    def validate(self):
        if self.window < 2 or self.epochs < 1 or self.batch < 1:
            raise ValueError("window>=2, epochs>=1 and batch>=1 are required")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if not 0 <= self.alpha_soh <= 1:
            raise ValueError("alpha_soh must be in [0,1]")
        if self.aux_mode not in ("all", "no_cycle", "no_duration", "none"):
            raise ValueError("Invalid aux_mode: " + self.aux_mode)
