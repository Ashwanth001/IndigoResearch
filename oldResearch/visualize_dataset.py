import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
data_dir = "data"
train_path = os.path.join(data_dir, "train_data.pt")
val_path = os.path.join(data_dir, "val_data.pt")
test_path = os.path.join(data_dir, "test_data.pt")

def get_stats(data_list, name):
    if not isinstance(data_list, list):
        data_list = [data_list]
    
    stats = []
    for item in data_list:
        year = item['year']
        # labels is a dict containing edge_label tensor
        labels = item['labels']['edge_label']
        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        history_years = [year - i for i in range(5)][::-1]
        stats.append({
            'Split': name,
            'Year': year,
            'History': history_years,
            'Positive': n_pos,
            'Negative': n_neg,
            'Total': n_pos + n_neg
        })
    return stats

# Load data
print("Loading PyTorch data objects...")
try:
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Collect stats
all_stats = []
all_stats.extend(get_stats(train_data, "Training"))
all_stats.extend(get_stats(val_data, "Validation"))
all_stats.extend(get_stats(test_data, "Testing"))

df = pd.DataFrame(all_stats)

# --- VISUALIZATION: Plotting ---
plt.figure(figsize=(14, 7))

# 1. Timeline Plot
plt.subplot(1, 2, 1)
colors = {'Training': '#3498db', 'Validation': '#f1c40f', 'Testing': '#e74c3c'}

for i, row in df.iterrows():
    plt.plot(row['History'], [row['Year']] * 5, '|--', color=colors[row['Split']], alpha=0.3)
    plt.scatter(row['Year'], row['Year'], color=colors[row['Split']], s=80, 
                label=row['Split'] if row['Split'] not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Temporal Windows: History vs Prediction Target", fontweight='bold')
plt.xlabel("Calendar Year")
plt.ylabel("Prediction Target Year")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# 2. Label Distribution Plot
plt.subplot(1, 2, 2)
label_sums = df.groupby('Split')[['Positive', 'Negative']].sum()
label_sums.plot(kind='bar', stacked=True, color=['#2ecc71', '#95a5a6'], ax=plt.gca())
plt.title("Sample Counts (Positive vs Negative Labels)", fontweight='bold')
plt.ylabel("Number of Predicted Pairs")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('data_split_visualization.png')
print("\nVisualization saved to 'data_split_visualization.png'")

# Text output for the report
print("\n--- DATASET SUMMARY TABLE ---")
summary_table = df.copy()
summary_table['History Range'] = summary_table['History'].apply(lambda x: f"{x[0]}-{x[-1]}")
print(summary_table[['Split', 'Year', 'History Range', 'Positive', 'Negative', 'Total']].to_string(index=False))
