import os
import logging
from datetime import datetime

print("=== DEBUGGING LOGGER SCRIPT ===")
print("Current directory is:", os.getcwd())

logs_dir = os.path.join(os.getcwd(), "logs")
print("About to create this directory:", logs_dir)
os.makedirs(logs_dir, exist_ok=True)
print("Directory created (or already exists).")

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)
print("This will be your log file path:", LOG_FILE_PATH)

with open(LOG_FILE_PATH, "w") as f:
    f.write("Test log file content!\n")

print("Finished writing the log file.")
logging = logging  # exposes 'logging' for import