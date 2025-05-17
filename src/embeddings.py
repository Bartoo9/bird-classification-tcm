import bioacoustics_model_zoo as bmz
import numpy as np
import os
import pandas as pd 
import shutil
import ast

#GENERATE THE EMBEDDINGS FROM THE SEGMENTS DIR
def get_embeddings(audio_dir, output_dir, batch_size=1000, force_regenerate=False):

    birdnet = bmz.BirdNET()
    os.makedirs(embeddings_dir, exist_ok=True)

    audio_files_all = [file for file in os.listdir(audio_dir) if file.endswith('.wav')]
    print(f"Found {len(audio_files_all)} total audio files")

    if not force_regenerate:
        existing_embeddings = set(
            os.path.splitext(file)[0] for file in os.listdir(output_dir) if file.endswith('.npy')
        )
        
        missing_files = [
            file for file in audio_files_all 
            if os.path.splitext(os.path.basename(file))[0] not in existing_embeddings
        ]
        
        print(f"Found {len(existing_embeddings)} existing embeddings")
        print(f"Need to process {len(missing_files)} new audio files")
    else:
        missing_files = audio_files_all
        print(f"Force regenerating all {len(missing_files)} audio files")
    
    if not missing_files:
        print("No new files to process. All embeddings are already available.")
        return
    
    total_processed = 0
    for i in range(0, len(missing_files), batch_size):
        batch_files = missing_files[i:min(i+batch_size, len(missing_files))]
        print(f"Processing batch {i//batch_size + 1}/{(len(missing_files)+batch_size-1)//batch_size}: {len(batch_files)} files")
        
        missing_paths = [os.path.join(audio_dir, file) for file in batch_files]
        try:
            embeddings = birdnet.embed(missing_paths, return_dfs=False, return_preds=False)
            
            if len(embeddings) != len(batch_files):
                print(f"Warning: Got {len(embeddings)} embeddings for {len(batch_files)} files")
                batch_files = batch_files[:len(embeddings)]
                
            for j, audio_file in enumerate(batch_files):
                file_name = os.path.splitext(audio_file)[0]
                output_path = os.path.join(output_dir, f"{file_name}.npy")
                np.save(output_path, embeddings[j])
                total_processed += 1
                
            print(f"Completed batch. Total processed: {total_processed}/{len(missing_files)}")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            print("Will continue with next batch")
    
    print(f"Embedding generation complete. Generated {total_processed} new embeddings.")

def embedding_annotation_file_multi(embeddings_dir, annotation_file, output_csv, missing_csv=None, sorted_dir=None):
    embedding_files = [file for file in os.listdir(embeddings_dir) if file.endswith('.npy')]
    print(f"Total embeddings found: {len(embedding_files)}")
    
    annotations_df = pd.read_csv(annotation_file)
    print(f"Total annotation entries: {len(annotations_df)}")
    
    if isinstance(annotations_df['species'].iloc[0], str):
        annotations_df['species'] = annotations_df['species'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    annotations_df['embedding_name'] = annotations_df['segment_name'].str.replace('.wav', '.npy')
    
    dupes = annotations_df['embedding_name'].duplicated(keep=False)
    duplicate_count = dupes.sum()
    print(f"Found {duplicate_count} duplicate annotation entries")
    
    unique_annotation_files = annotations_df['embedding_name'].unique()
    print(f"Unique annotation files: {len(unique_annotation_files)}")
    
    valid_embeddings = annotations_df[annotations_df['embedding_name'].isin(embedding_files)].copy()
    print(f"Valid embeddings with annotations: {len(valid_embeddings)}")
    print(f"Missing embeddings: {len(annotations_df) - len(valid_embeddings)}")
    
    valid_embeddings['multiple_species'] = valid_embeddings['species'].apply(lambda x: len(x) > 1)
    
    valid_embeddings.to_csv(output_csv, index=False)
    print(f"Created multi-species annotation file with {len(valid_embeddings)} entries")
    print(f"Covering {len(valid_embeddings['embedding_name'].unique())} unique embeddings")
    
    if missing_csv:
        missing_annotations = set(annotations_df['embedding_name']) - set(embedding_files)
        pd.DataFrame({"missing_file": list(missing_annotations)}).to_csv(missing_csv, index=False)
        print(f"Saved {len(missing_annotations)} missing files to {missing_csv}")
    
    if sorted_dir:
        os.makedirs(sorted_dir, exist_ok=True)
        
        all_species = set()
        for species_list in valid_embeddings['species']:
            all_species.update(species_list)
        
        print(f"Found {len(all_species)} unique species")
        
        for species in all_species:
            if species == "Unknown":
                continue
                
            species_dir = os.path.join(sorted_dir, species.replace("/", "_"))
            os.makedirs(species_dir, exist_ok=True)
            
            species_embeddings = valid_embeddings[valid_embeddings['species'].apply(lambda x: species in x)]['embedding_name'].unique()
            
            for embedding_name in species_embeddings:
                src_path = os.path.join(embeddings_dir, embedding_name)
                dst_path = os.path.join(species_dir, embedding_name)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
    
        print(f"Organized embeddings by {len(all_species)} species")
    
    return valid_embeddings

if __name__ == '__main__':
    audio_dir = "../processed_dataset/3s_segments_multi"
    output_dir = "../dataset/dataset_ctx_multi"
    embeddings_dir = f'{output_dir}/embeddings_multi'
    annotation_file = "../processed_dataset/segment_metadata_multi.csv"
    output_csv = "../dataset/dataset_ctx_multi/embedding_annotations_multi.csv"

    sorted_dir = "../dataset/dataset_ctx_multi/sorted_embeddings_multi"
    
    # Generate embeddings if needed (uncomment to run)
    # get_embeddings(audio_dir, embeddings_dir, batch_size=1000, force_regenerate=False)
    
    # Create annotation file
    embedding_annotation_file_multi(embeddings_dir, annotation_file, 
                                   output_csv, sorted_dir)