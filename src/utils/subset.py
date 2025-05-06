import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
        

def analyze_vocalization_durations(stats_file, annotations_path, output_dir=None):
    df = pd.read_csv(stats_file)
    
    annotations = pd.read_csv(annotations_path)
    embedding_counts = annotations['label'].value_counts().to_dict()

    df['embedding_count'] = df['species'].map(embedding_counts).fillna(0).astype(int)
    
    print("\nSpecies by embedding count (top 20):")
    top_species = df.sort_values('embedding_count', ascending=False).head(20)
    print(top_species[['species', 'embedding_count', 'mean', 'std']])
    
    short_species = df[df['mean'] < 2]['species'].tolist()
    medium_species = df[(df['mean'] >= 2) & (df['mean'] < 5)]['species'].tolist()
    long_species = df[df['mean'] >= 5]['species'].tolist()
    
    print("\nVocalization Length Groups:")
    print(f"Short vocalizations (<2s): {len(short_species)} species")
    print(f"Medium vocalizations (2-5s): {len(medium_species)} species")
    print(f"Long vocalizations (>5s): {len(long_species)} species")
    print("\nShort vocalization species (with embedding counts):")
    short_df = df[df['species'].isin(short_species)]
    print(short_df[['species', 'mean', 'embedding_count']].sort_values('embedding_count', ascending=False))
    
    print("\nMedium vocalization species (with embedding counts):")
    medium_df = df[df['species'].isin(medium_species)]
    print(medium_df[['species', 'mean', 'embedding_count']].sort_values('embedding_count', ascending=False))
    
    print("\nLong vocalization species (with embedding counts):")
    long_df = df[df['species'].isin(long_species)]
    print(long_df[['species', 'mean', 'embedding_count']].sort_values('embedding_count', ascending=False))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))
        bins = np.linspace(0, 16, 33) 
        plt.hist(df['mean'], bins=bins, alpha=0.7)
        plt.axvline(x=2, color='r', linestyle='--', label='Short/Medium Boundary (2s)')
        plt.axvline(x=5, color='g', linestyle='--', label='Medium/Long Boundary (5s)')
        plt.xlabel('Mean Vocalization Duration (seconds)')
        plt.ylabel('Number of Species')
        plt.title('Distribution of Bird Species by Mean Vocalization Duration')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'vocalization_duration_distribution.png'))

        plt.figure(figsize=(12, 8))
        sizes = df['embedding_count'].clip(upper=5000) / 50 + 20  
        
        scatter = plt.scatter(df['mean'], df['std'], 
                   s=sizes,  
                   alpha=0.7)
        
        for i, row in df.iterrows():
            if row['mean'] < 16 and row['std'] < 16 and row['embedding_count'] > 300:
                plt.annotate(f"{row['species']} ({row['embedding_count']})", 
                             (row['mean'], row['std']), 
                             fontsize=8, alpha=0.7)
        
        plt.axvline(x=2, color='r', linestyle='--', label='Short/Medium Boundary (2s)')
        plt.axvline(x=5, color='g', linestyle='--', label='Medium/Long Boundary (5s)')
        plt.xlabel('Mean Vocalization Duration (seconds)')
        plt.ylabel('Standard Deviation of Duration (seconds)')
        plt.title('Bird Species by Duration (point size = embedding count)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(0, 16)
        plt.ylim(0, 16)
        plt.savefig(os.path.join(output_dir, 'vocalization_duration_scatter_sized.png'))
        
        plt.figure(figsize=(12, 8))
        plt.scatter(df['mean'], df['embedding_count'], alpha=0.7)
        for i, row in df.iterrows():
            if row['embedding_count'] > 300: 
                plt.annotate(row['species'], 
                             (row['mean'], row['embedding_count']),
                             fontsize=8, alpha=0.7)
                
        plt.axvline(x=2, color='r', linestyle='--', label='Short/Medium Boundary (2s)')
        plt.axvline(x=5, color='g', linestyle='--', label='Medium/Long Boundary (5s)')
        plt.xlabel('Mean Vocalization Duration (seconds)')
        plt.ylabel('Number of Embeddings')
        plt.title('Embedding Count vs Vocalization Duration by Species')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'duration_vs_count.png'))
    

    min_embeddings = 300 
    
    well_represented_short = df[(df['species'].isin(short_species)) & 
                               (df['embedding_count'] >= min_embeddings)]
    well_represented_medium = df[(df['species'].isin(medium_species)) & 
                                (df['embedding_count'] >= min_embeddings)]
    well_represented_long = df[(df['species'].isin(long_species)) & 
                              (df['embedding_count'] >= min_embeddings)]
    
    print(f"\nWell-represented species (≥{min_embeddings} embeddings):")
    print(f"Short: {len(well_represented_short)} species")
    print(f"Medium: {len(well_represented_medium)} species")
    print(f"Long: {len(well_represented_long)} species")
    
    top_short = well_represented_short.sort_values('embedding_count', ascending=False).head(8)['species'].tolist()
    top_medium = well_represented_medium.sort_values('embedding_count', ascending=False).head(8)['species'].tolist()
    top_long = well_represented_long.sort_values('embedding_count', ascending=False).head(8)['species'].tolist()
    
    print("\nRecommended species for experiment:")
    
    print(f"Short vocalizations (<2s): {len(top_short)} species")
    for s in top_short:
        row = df[df['species'] == s].iloc[0]
        print(f"  - {s}: {int(row['embedding_count'])} embeddings, {row['mean']:.2f}s mean duration")
    
    print(f"Medium vocalizations (2-5s): {len(top_medium)} species")
    for s in top_medium:
        row = df[df['species'] == s].iloc[0]
        print(f"  - {s}: {int(row['embedding_count'])} embeddings, {row['mean']:.2f}s mean duration")
    
    print(f"Long vocalizations (>5s): {len(top_long)} species")
    for s in top_long:
        row = df[df['species'] == s].iloc[0]
        print(f"  - {s}: {int(row['embedding_count'])} embeddings, {row['mean']:.2f}s mean duration")
    
    if output_dir:
        df.to_csv(os.path.join(output_dir, 'species_stats_with_counts.csv'), index=False)
    
    return top_short, top_medium, top_long, df


if __name__ == "__main__":
    stats_file = "../dataset/filtered_dataset/duration_stats.csv"
    annotations_path = "../dataset/_embedding_annotations.csv"
    output_dir = "../dataset/vocalization_analysis"
    
    short_species, medium_species, long_species, stats_df = analyze_vocalization_durations(
        stats_file, annotations_path, output_dir)