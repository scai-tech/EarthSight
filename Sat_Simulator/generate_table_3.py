import json
import os
import statistics

logs_dir = "logs"
results = {}
power = {}

folders = os.listdir(logs_dir)
folders.sort()

for folder in folders:
    power_path = os.path.join(logs_dir, folder, "power.json")
    if os.path.exists(power_path):
        with open(power_path, 'r') as f:
            power_data = json.load(f)
            # Extracting the average power consumption
            #             {
            #     "power_generation": 76723200000,
            #     "power_consumptions": 104200200000.0,
            #     "compute_seconds": 298110.0,
            #     "total_compute_power": 17886600000.0,
            #     "pct_of_gen_for_cmpt": 0.23313156906906907
            # }
            avg_power = power_data.get("pct_of_gen_for_cmpt", 0)  # Assuming this key exists
            cmpt_seconds = power_data.get("compute_seconds", 0)
            power[folder] = (avg_power, cmpt_seconds)

for folder, val in sorted(power.items()):
    print(f"{folder}: {100*val[0]:.2f}%")

import os
os.makedirs("results", exist_ok=True)

with open("results/table3.txt", "w+") as f:
    for folder, val in sorted(power.items()):
        f.write(f"{folder}: {100*val[0]:.2f}%\n")