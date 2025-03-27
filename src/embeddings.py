import bioacoustics_model_zoo as bmz
import numpy as np
import os
import pandas as pd 
import shutil

def get_embeddings(audio_dir, output_dir):
    birdnet = bmz.BirdNET()

    audio_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.wav')]
    embeddings = birdnet.embed(audio_files, return_dfs = False, return_preds = False)

    for i, audio_file in enumerate(audio_files):
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_path = os.path.join(output_dir, f"{file_name}.npy")
        np.save(output_path, embeddings[i])

def embedding_annotation_file(embeddings_dir, annotation_file, output_csv, missing_csv, sorted_dir):
    annotations = pd.read_csv(annotation_file)

    embedding_files = [os.path.splitext(file)[0] for file in 
                       os.listdir(embeddings_dir) if file.endswith('.npy')]
    embedding_annotations = annotations[
        annotations["segment name"].str.replace(".wav", "").isin(embedding_files)
    ]
    embedding_annotations["embedding name"] = embedding_annotations["segment name"].str.replace(
        ".wav", ".npy"
    )
    embedding_annotations[["embedding name", "label"]].to_csv(output_csv, index=False)

    missing_embeddings = set(embedding_files) - set(
        annotations["segment name"].str.replace(".wav", "")
    )

    if missing_embeddings:
        pd.DataFrame({"missing embedding": list(missing_embeddings)}).to_csv(
            missing_csv, index=False
        )
    
    if not os.path.exists(sorted_dir):
        os.makedirs(sorted_dir)
    
    for _, row in embedding_annotations.iterrows():
        embedding_name = row["embedding name"]
        label = row["label"]
        label_dir = os.path.join(sorted_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        src_path = os.path.join(embeddings_dir, embedding_name)
        dst_path = os.path.join(label_dir, embedding_name)
        shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    audio_dir = "../dataset/3s_segments"
    output_dir = "../dataset/embeddings"
    embeddings_dir = output_dir
    annotation_file = "../dataset/_segment_annotations.csv"
    output_csv = "../dataset/_embedding_annotations.csv"
    missing_csv = "../dataset/_missing_embeddings.csv"
    sorted_dir = "../dataset/sorted_embeddings"

    # get_embeddings(audio_dir, output_dir)
    embedding_annotation_file(embeddings_dir, annotation_file,
                               output_csv, missing_csv, sorted_dir)