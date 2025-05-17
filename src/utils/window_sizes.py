import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

#window size distributions 
def analyze_window_sizes(X_windows, y_encoded, class_names, output_dir=None):

    annotations_path = "../dataset/dataset_ctx_multi/embedding_annotations_multi.csv"
    embeddings_dir = "../dataset/dataset_ctx_multi/embeddings_multi"
    
    if os.path.exists(annotations_path):
        try:
            df = pd.read_csv(annotations_path)
            print(f"- CSV loaded successfully with {len(df)} rows")
            print(f"- Columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"- First row species: {df['species'].iloc[0]}")
        except Exception as e:
            print(f"- Error reading CSV: {e}")

    if isinstance(y_encoded, np.ndarray):
        y_encoded_array = y_encoded
    elif len(y_encoded) > 0:  
        y_encoded_array = np.array(y_encoded)
    else:
        y_encoded_array = np.array([])
    
    is_multilabel = (len(y_encoded_array.shape) > 1 and 
                     y_encoded_array.shape[0] > 0 and 
                     y_encoded_array.shape[1] > 1)
    
    window_sizes = [len(window) for window in X_windows]
    
    species_window_sizes = defaultdict(list)
    
    for i, (window, label) in enumerate(zip(X_windows, y_encoded)):
        if is_multilabel:
            positive_classes = np.where(label == 1)[0]
            for cls_idx in positive_classes:
                if cls_idx < len(class_names): 
                    species = class_names[cls_idx]
                    species_window_sizes[species].append(len(window))
        else:
            species = class_names[label]  
            species_window_sizes[species].append(len(window))
    
    species_stats = []
    for species, sizes in species_window_sizes.items():
        species_stats.append({
            'species': species,
            'count': len(sizes),
            'avg_size': np.mean(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes)
        })
    
    species_stats_df = pd.DataFrame(species_stats).sort_values('count', ascending=False)
    
    for _, row in species_stats_df.head(5).iterrows():
        print(f"  - {row['species']}: {row['count']} windows, avg size {row['avg_size']:.2f}")
    
    if output_dir:
        species_stats_df.to_csv(os.path.join(output_dir, 'window_size_stats.csv'), index=False)
    
        plt.figure(figsize=(10, 6))
        plt.hist(window_sizes, bins=20)
        plt.title('Distribution of Window Sizes')
        plt.xlabel('Window Size')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'window_size_histogram.png'), dpi=300)
        
        plt.figure(figsize=(12, 8))
        plot_data = []
        labels = []
        
        for _, row in species_stats_df.head(10).iterrows():
            species = row['species']
            sizes = species_window_sizes[species]
            plot_data.append(sizes)
            labels.append(f"{species} (n={len(sizes)})")
        
        if plot_data:  
            plt.violinplot(plot_data, showmeans=True)
            plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
            plt.title('Window Sizes by Top 10 Species')
            plt.ylabel('Window Size')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'window_size_by_species.png'), dpi=300)
        
        if is_multilabel:
            species_cooccurrence = np.zeros((len(class_names), len(class_names)))
            
            for label in y_encoded:
                positive_classes = np.where(label == 1)[0]
                for i in positive_classes:
                    for j in positive_classes:
                        species_cooccurrence[i, j] += 1
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(species_cooccurrence, xticklabels=class_names, 
                        yticklabels=class_names, cmap='viridis')
            plt.title('Species Co-occurrence in Windows')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'species_cooccurrence.png'), dpi=300)
    
    return species_stats_df