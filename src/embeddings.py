import bioacoustics_model_zoo as bmz
import numpy as np
import os
import pandas as pd 
import shutil

#citation https://opensoundscape.org/en/latest/tutorials/training_birdnet_and_perch.html
def get_embeddings(audio_dir, output_dir):
    birdnet = bmz.BirdNET()

    os.makedirs(output_dir, exist_ok=True)

    audio_files_all = [file for file in os.listdir(audio_dir) if file.endswith('.wav')]

    existing_embeddings = [os.path.splitext(file)[0] 
             + '.wav' for file in os.listdir(output_dir) if file.endswith('.npy')]
    
    missing_files = [file for file in audio_files_all if os.path.basename(file) 
                     not in existing_embeddings]
    
    if not missing_files:
        print("All audio files already processed.")
        return
    
    print(f"Processing {len(missing_files)} audio files...")

    missing_paths = [os.path.join(audio_dir, file) for file in missing_files]
    embeddings = birdnet.embed(missing_paths,return_dfs = False, return_preds = False)

    for i, audio_file in enumerate(missing_files):
        file_name = os.path.splitext(audio_file)[0]
        output_path = os.path.join(output_dir, f"{file_name}.npy")
        np.save(output_path, embeddings[i])

def embedding_annotation_file(embeddings_dir, annotation_file, output_csv, missing_csv=None, sorted_dir=None):
    embedding_files = [file for file in os.listdir(embeddings_dir) if file.endswith('.npy')]
    print(f"Total embeddings found: {len(embedding_files)}")
    
    annotations_df = pd.read_csv(annotation_file)
    print(f"Total annotation entries: {len(annotations_df)}")
    
    annotations_df['embedding_file'] = annotations_df['segment name'].str.replace('.wav', '.npy')
    
    #had quite a lot of duplicates for some reason
    dupes = annotations_df['embedding_file'].duplicated(keep=False)
    duplicate_count = dupes.sum()
    print(f"Found {duplicate_count} duplicate annotation entries")
    
    unique_annotation_files = annotations_df['embedding_file'].unique()
    print(f"Unique annotation files: {len(unique_annotation_files)}")
    
    valid_embeddings = annotations_df[annotations_df['embedding_file'].isin(embedding_files)]
    print(f"Valid embeddings with annotations: {len(valid_embeddings)}")
    print(f"Missing embeddings: {len(annotations_df) - len(valid_embeddings)}")
    
    embedding_data = []
    processed_embeddings = set()
    
    for _, row in valid_embeddings.iterrows():
        if row["embedding_file"] not in processed_embeddings:
            embedding_data.append({
                "embedding name": row["embedding_file"],
                "label": row["label"]
            })
            processed_embeddings.add(row["embedding_file"])
    
    unlabeled_embeddings = set(embedding_files) - set(valid_embeddings['embedding_file'])
    if unlabeled_embeddings:
        print(f"Found {len(unlabeled_embeddings)} embeddings without annotation entries")
        for file_name in unlabeled_embeddings:
            base_name = os.path.splitext(file_name)[0]
            label = base_name.split('_')[0] if '_' in base_name else 'unknown'
            embedding_data.append({
                "embedding name": file_name,
                "label": label
            })
    
    embedding_df = pd.DataFrame(embedding_data)
    embedding_df.to_csv(output_csv, index=False)
    print(f"Created annotation file with {len(embedding_df)} entries (should match total embeddings)")
    
    if missing_csv:
        missing_annotations = set(annotations_df['embedding_file']) - set(embedding_files)
        pd.DataFrame({"missing_file": list(missing_annotations)}).to_csv(missing_csv, index=False)
        print(f"Saved {len(missing_annotations)} missing files to {missing_csv}")
    
    if sorted_dir and not os.path.exists(sorted_dir):
        os.makedirs(sorted_dir, exist_ok=True)
        
        label_counts = embedding_df['label'].value_counts()
        print(f"Found {len(label_counts)} unique labels")
        
        #if i need to sort them later
        for _, row in embedding_df.iterrows():
            embedding_name = row["embedding name"]
            label = row["label"]
            label_dir = os.path.join(sorted_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            src_path = os.path.join(embeddings_dir, embedding_name)
            dst_path = os.path.join(label_dir, embedding_name)
            shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    audio_dir = "../processed_dataset/3s_segments_ctx"
    output_dir = "../dataset/embeddings"
    embeddings_dir = output_dir
    annotation_file = "../dataset/_segment_annotations.csv"
    output_csv = "../dataset/_embedding_annotations.csv"
    missing_csv = "../dataset/_missing_embeddings.csv"
    sorted_dir = "../dataset/sorted_embeddings"

    # get_embeddings(audio_dir, output_dir)
    embedding_annotation_file(embeddings_dir, annotation_file,
                               output_csv)