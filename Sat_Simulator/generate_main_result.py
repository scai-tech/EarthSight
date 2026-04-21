import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuration
logs_dir = "logs"
output_path = "results/summary_plot.png"
results = {}

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

# 1. Data Extraction
if not os.path.exists(logs_dir):
    print(f"Error: Directory '{logs_dir}' not found.")
    exit()

folders = os.listdir(logs_dir)

for folder in folders:
    summary_path = os.path.join(logs_dir, folder, "summary.json")
    print(summary_path)
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            # 1. Read the file as a raw string
            raw_data = f.read()
            
            # 2. Forcefully remove all NUL bytes and any trailing whitespace
            cleaned_data = raw_data.replace('\x00', '').strip()
            
            # Skip if the file was empty or only contained NUL bytes
            if not cleaned_data:
                print(f"Warning: {summary_path} is empty or only contained NUL bytes.")
                continue
                
            try:
                # 3. Parse the cleaned string
                summary_data = json.loads(cleaned_data)
                
                # Pulling the specific latency metric
                results[folder] = summary_data.get("weighted_avg_p90_priority_gt_5", 0.0)
                
            except json.JSONDecodeError as e:
                # Catching errors so one badly corrupted file doesn't break your entire loop
                print(f"Failed to parse JSON in {summary_path} even after cleaning: {e}")

# for folder in folders:
#     summary_path = os.path.join(logs_dir, folder, "summary.json")
#     print(summary_path)
#     if os.path.exists(summary_path):
#         with open(summary_path, 'r') as f:
#             summary_data = json.load(f)
#             # Pulling the specific latency metric
#             results[folder] = summary_data.get("weighted_avg_p90_priority_gt_5", 0.0)

# 2. Mapping with Individual Bar Validation
mapping = [
    ('Natural Disaster\n(Coral/TPU)', 'log_tpu-naturaldisaster-serval-mtl-48h', 'log_tpu-naturaldisaster-earthsight-stl-48h', 'log_tpu-naturaldisaster-earthsight-mtl-48h'),
    ('Natural Disaster\n(Jetson/GPU)', 'log_gpu-naturaldisaster-serval-mtl-48h', 'log_gpu-naturaldisaster-earthsight-stl-48h', 'log_gpu-naturaldisaster-earthsight-mtl-48h'),
    ('Intelligence\n(Coral/TPU)',     'log_tpu-intelligence-serval-mtl-48h',     'log_tpu-intelligence-earthsight-stl-48h',     'log_tpu-intelligence-earthsight-mtl-48h'),
    ('Intelligence\n(Jetson/GPU)',     'log_gpu-intelligence-serval-mtl-48h',     'log_gpu-intelligence-earthsight-stl-48h',     'log_gpu-intelligence-earthsight-mtl-48h'),
    ('Urban Obs.\n(Coral/TPU)',      'log_tpu-combined-serval-mtl-48h',         'log_tpu-combined-earthsight-stl-48h',         'log_tpu-combined-earthsight-mtl-48h'),
    ('Urban Obs.\n(Jetson/GPU)',      'log_gpu-combined-serval-mtl-48h',         'log_gpu-combined-earthsight-stl-48h',         'log_gpu-combined-earthsight-mtl-48h')
]

labels, baseline_vals, st_vals, mt_vals = [], [], [], []

print(f"{'Folder Key':<50} | {'Value':<10} | {'Status'}")
print("-" * 75)

def get_val_and_log(key):
    """Helper to fetch data and print status for every single bar."""
    if key in results:
        val = results[key]
        print(f"{key:<50} | {val:>10.4f} | OK")
        return val
    else:
        print(f"{key:<50} | {'X':>10} | MISSING")
        return 0.0

for label, b_key, st_key, mt_key in mapping:
    labels.append(label)
    # Check every individual bar key
    baseline_vals.append(get_val_and_log(b_key))
    st_vals.append(get_val_and_log(st_key))
    mt_vals.append(get_val_and_log(mt_key))
    print("-" * 75) # Separator between groups

# 3. Plotting (Same as before but with validated data)
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, baseline_vals, width, label='Baseline (Serval)', color='#1a2e40', edgecolor='white', hatch='///')
ax.bar(x,         st_vals,       width, label='EarthSight ST',     color='#5b9bd5', edgecolor='white')
ax.bar(x + width, mt_vals,       width, label='EarthSight MT',     color='#548235', edgecolor='white')

ax.set_ylabel('Latency (minutes)', fontsize=12)
ax.set_title('90th Percentile Latency across Scenarios and Hardware', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(output_path)
print(f"\nResults saved to {output_path}")