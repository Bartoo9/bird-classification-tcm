def tsne_embeddings_selected_species(embeddings_dir, annotations_path, perplexity=30, random_state=1):
    import pandas as pd
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    short_species = [
        'Emberiza calandra', 'Cyanistes caeruleus', 'Regulus regulus',
        'Certhia brachydactyla', 'Emberiza cirlus'
    ]

    medium_species = [
        'Fringilla coelebs', 'Turdus merula', 'Sylvia atricapilla',
        'Erithacus rubecula', 'Luscinia megarhynchos', 'Parus major',
        'Phylloscopus collybita', 'Turdus philomelos'
    ]

    long_species = [
        'Periparus ater', 'Alauda arvensis', 'Cuculus canorus',
        'Acrocephalus scirpaceus', 'Acrocephalus arundinaceus', 
        'Columba palumbus', 'Spinus spinus'
    ]
    
    selected_species = short_species + medium_species + long_species
    
    annotations = pd.read_csv(annotations_path)
    
    filtered_annotations = annotations[annotations['label'].isin(selected_species)]
    print(f"Found {len(filtered_annotations)} embeddings for the 20 selected species")
    
    embeddings = []
    labels = []
    categories = []
    
    for _, row in filtered_annotations.iterrows():
        embedding_path = os.path.join(embeddings_dir, row['embedding name'])
        if os.path.exists(embedding_path):
            data = np.load(embedding_path)
            embeddings.append(data)
            labels.append(row['label'])

            if row['label'] in short_species:
                categories.append('short')
            elif row['label'] in medium_species:
                categories.append('medium')
            else:
                categories.append('long')
    
    embeddings = np.array(embeddings)
    
    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)

    tsne_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'species': labels,
        'category': categories
    })
    
    plt.figure(figsize=(14, 12))
    
    sns.scatterplot(
        data=tsne_df, x='x', y='y', 
        hue='species',  
        palette='tab20',  
        s=50, alpha=0.7
    )
    
    plt.title("t-SNE of Bird Embeddings (20 Selected Species)")
    plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tsne_by_species.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    g = sns.FacetGrid(tsne_df, col="species", col_wrap=5, height=3)
    g.map(sns.scatterplot, "x", "y", alpha=0.7)
    g.add_legend()
    g.savefig("tsne_facet_by_species.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    embeddings_dir = "../dataset/augmented_embeddings"
    embedding_annotations = "../dataset/_augmented_embedding_annotations.csv"
    
    tsne_embeddings_selected_species(embeddings_dir, embedding_annotations)