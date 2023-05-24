import os

os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys

from .application import Application

app = Application(sys.argv)
sys.exit(app.exec())
