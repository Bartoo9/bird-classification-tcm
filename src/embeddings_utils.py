import os
import pandas as pd
from collections import Counter

def check_embeddings_data(embeddings_dir, annotation_file):
    annotations = pd.read_csv(annotation_file)

    # per class distribution
    print("\nClass Distribution:")
    class_counts = Counter(annotations["label"])
    for label, count in class_counts.items():
        print(f"{label}: {count} samples")

    #check if some are below 50 samples
    underrepresented_classes = [label for label, count in class_counts.items() if count < 50]
    print("\nUnderrepresented Classes (fewer than 10 samples):")
    print(underrepresented_classes)

    #null cases
    print("\nChecking for missing or null values...")
    missing_embeddings = annotations["embedding name"].isnull().sum()
    missing_labels = annotations["label"].isnull().sum()
    print(f"Missing embedding names: {missing_embeddings}")
    print(f"Missing labels: {missing_labels}")

    # duplicates
    print("\nChecking for duplicate entries...")
    duplicate_rows = annotations.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")

    #missing embeddings
    print("\nChecking for missing embeddings in the directory...")
    missing_files = []
    for embedding_name in annotations["embedding name"]:
        embedding_path = os.path.join(embeddings_dir, embedding_name)
        if not os.path.exists(embedding_path):
            missing_files.append(embedding_name)

    print(f"Number of missing embeddings: {len(missing_files)}")
    if missing_files:
        print("Missing embeddings:")
        print(missing_files[:10])

    # Summary
    print("\nSummary:")
    print(f"Total classes: {len(class_counts)}")
    print(f"Total samples: {len(annotations)}")
    print(f"Underrepresented classes: {len(underrepresented_classes)}")
    print(f"Missing embeddings: {len(missing_files)}")
    print(f"Duplicate rows: {duplicate_rows}")
    print("\nLowest amount of samples per class:")
    print(class_counts.most_common()[-1])
    print("\nHighest amount of samples per class:")
    print(class_counts.most_common()[0])

def analyze_embeddings_distribution(embedding_annotations_path, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    annotations = pd.read_csv(embedding_annotations_path)
    
    species_counts = annotations['label'].value_counts().reset_index()
    species_counts.columns = ['species', 'count']
    
    species_counts.to_csv(os.path.join(output_dir, 'species_distribution.csv'), index=False)
    
    stats = {
        'total_embeddings': len(annotations),
        'total_species': len(species_counts),
        'min_count': species_counts['count'].min(),
        'max_count': species_counts['count'].max(),
        'median_count': species_counts['count'].median(),
        'mean_count': species_counts['count'].mean(),
        'std_count': species_counts['count'].std(),
        'underrepresented_species': len(species_counts[species_counts['count'] < 40])
    }
    
    with open(os.path.join(output_dir, 'embedding_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    plt.figure(figsize=(15, 8))
    sns.histplot(species_counts['count'], bins=30, kde=True)
    plt.title('Distribution of Embeddings per Species')
    plt.xlabel('Number of Embeddings')
    plt.ylabel('Number of Species')
    plt.savefig(os.path.join(output_dir, 'embeddings_distribution.png'))
    
    plt.figure(figsize=(15, 10))
    top_species = species_counts.sort_values('count', ascending=False).head(30)
    sns.barplot(x='count', y='species', data=top_species)
    plt.title('Top 30 Species by Number of Embeddings')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_species_counts.png'))
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return species_counts, stats

if __name__ == "__main__":
    embeddings_dir = "../dataset/embeddings"
    annotation_file = "../dataset/_embedding_annotations.csv"

    # check_embeddings_data(embeddings_dir, annotation_file)
    analyze_embeddings_distribution(annotation_file, "../dataset/embeddings_analysis")