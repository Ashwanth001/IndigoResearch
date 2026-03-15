import os
import subprocess
import sys

log_file = "execution_log.txt"

def log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    print(msg)

log("Starting execution...")
log(f"Current directory: {os.getcwd()}")
log(f"Python version: {sys.version}")

try:
    if not os.path.exists("data"):
        os.makedirs("data")
        log("Created data directory.")
    else:
        log("Data directory already exists.")
except Exception as e:
    log(f"Error creating directory: {e}")

log("Execution finished.")
