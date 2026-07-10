"""Train only the V3 TCN IMU fall model."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from train_v3_tcn_cnn_gru import main as shared_main  # noqa: E402


if __name__ == "__main__":
    if "--models" not in sys.argv:
        sys.argv.extend(["--models", "tcn"])
    shared_main()
