import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter

# # 38074 - segments
# # 41462 - annotrations in the csv file
# # 41188 - spectrograms

# 31480 - embeddings
# Embedding shape: (1024,)

def annotations_from_spectrograms(spectrogram_dir, output_csv="mel_spectrogram_annotations.csv"):
    
    if not os.path.exists(spectrogram_dir):
        print(f"Spectrogram directory not found: {spectrogram_dir}")
        return

    spectrogram_data = []
    for label in os.listdir(spectrogram_dir):
        label_dir = os.path.join(spectrogram_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(".png"):
                    spectrogram_data.append({
                        "segment name": file,
                        "label": label,
                        "Begin Time (s)": 0, 
                        "End Time (s)": 3    
                    })

    annotations = pd.DataFrame(spectrogram_data)
    print(f"Total spectrograms found: {len(annotations)}")

    annotations.to_csv(output_csv, index=False)
    print(f"Annotation file generated: {output_csv}")

def check_embeddings(embeddings_dir):

    total_files = 0
    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            total_files += 1

    data = np.load(os.path.join(embeddings_dir, os.listdir(embeddings_dir)[0]))
    print(f"Embedding shape: {data.shape}")
    print(f"Embedding example: {data[:5]}")

    print(f"Total embeddings found: {total_files}")

def load_embeddings(embeddings_dir):
    embeddings = []
    file_names = []
    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            data = np.load(os.path.join(embeddings_dir, file))
            embeddings.append(data)
            file_names.append(file)
    return np.array(embeddings), file_names

def tsne_embeddings(embeddings, file_names, preplexity=30, random_state=1):
    tsne = TSNE(n_components=2, perplexity=preplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], edgecolors="blue",
                facecolors="none",
                alpha=0.7, s=20)
    plt.title("t-SNE of Embeddings")
    plt.savefig("tsne_embeddings.png")
    plt.show()

if __name__ == "__main__":
    spectrogram_dir = "../dataset/mel_spectrograms"
    output_csv = "../dataset/mel_spectrogram_annotations.csv"
    embeddings_dir = "../dataset/embeddings"
    segment_annotations = "../dataset/_segment_annotations.csv"
    embedding_annotations = "../dataset/_embedding_annotations.csv"
    segments_dir = "../dataset/3s_segments"
    # annotations_from_spectrograms(spectrogram_dir, output_csv)
    # check_embeddings(embeddings_dir)
    # embeddings, file_names = load_embeddings(embeddings_dir)
    # #they look quite okay 
    # tsne_embeddings(embeddings, file_names)

    
