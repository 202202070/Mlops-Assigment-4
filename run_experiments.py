import yaml
import subprocess
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


experiments = [
    {"learning_rate": 0.1,   "epochs": 10, "batch_size": 64},
    {"learning_rate": 0.01,  "epochs": 10, "batch_size": 64},
    {"learning_rate": 0.001, "epochs": 10, "batch_size": 64},
    {"learning_rate": 0.01,  "epochs": 10, "batch_size": 32},
    {"learning_rate": 0.01,  "epochs": 10, "batch_size": 128},
]

config_path = os.path.join(SCRIPT_DIR, "config.yaml")
train_path = os.path.join(SCRIPT_DIR, "train.py")

for i, config in enumerate(experiments):
    print(f"\n{'='*50}")
    print(f"  Running Experiment {i+1} of {len(experiments)}")
    print(f"  Config: {config}")
    print(f"{'='*50}\n")

    # write config to config.yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # run train.py
    result = subprocess.run([sys.executable, train_path], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"Experiment {i+1} failed!")
    else:
        print(f"Experiment {i+1} completed successfully!")

print("\n\nAll experiments finished!")
print("Open http://localhost:5000 to see results in MLflow UI")
