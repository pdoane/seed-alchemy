import os

os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import uvicorn

start_time = time.perf_counter()
from .main import app

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Startup in {elapsed_time:.6f} seconds")

uvicorn.run(app, host="0.0.0.0", port=8000)
