import wandb

# Login to WandB (if needed)
wandb.login()

# Initialize API
api = wandb.Api()

# Replace with your WandB run path (project/user details)
run = api.run("wandb/run-20250115_174737-mm3fhkh2")

# Get system metrics history
metrics = run.history(stream="events")

# Find GPU power usage column (it may vary, look for similar names)
gpu_power_columns = [col for col in metrics.columns if "gpu.powerWatts" in col]
if gpu_power_columns:
    gpu_power_data = metrics[gpu_power_columns]
    print(gpu_power_data.head())  # Preview the data
else:
    print("No GPU power usage data found.")
