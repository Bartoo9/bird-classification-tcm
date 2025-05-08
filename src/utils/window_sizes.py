import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def analyze_window_sizes(X_windows, y_labels, class_names, output_dir=None, model_name=None):

    species_windows = {}
    for i, class_name in enumerate(class_names):
        species_windows[class_name] = []
    
    for i in range(len(X_windows)):
        label_idx = y_labels[i]
        species = class_names[label_idx]
        window_size = len(X_windows[i])
        species_windows[species].append(window_size)
    
    summary_data = []
    for species, sizes in species_windows.items():
        if sizes:
            summary_data.append({
                'Species': species,
                'Count': len(sizes),
                'Mean Size': np.mean(sizes),
                'Min Size': min(sizes),
                'Max Size': max(sizes),
                'Median Size': np.median(sizes)
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nWindow Size Analysis by Species:")
    print("--------------------------------")
    print(f"{'Species':40} | {'Count':6} | {'Avg':5} | {'Min':4} | {'Max':4}")
    print("-" * 65)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Species']:40} | {row['Count']:6} | {row['Mean Size']:5.1f} | {row['Min Size']:4} | {row['Max Size']:4}")
    
    if output_dir and model_name:
        summary_df.to_csv(os.path.join(output_dir, f"{model_name}_window_sizes.csv"), index=False)
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        summary_df = summary_df.sort_values('Mean Size')
        
        sns.barplot(x='Mean Size', y='Species', data=summary_df, 
                   palette='viridis', orient='h')
        
        plt.title('Average Window Size by Species', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_window_sizes.png"), dpi=300)
        plt.close()
    
    return summary_df