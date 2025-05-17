import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
from collections import defaultdict

#original 1 minute audio segmentation into 3 second segments, used for creating the embeddings with BirdNET
#gives out the new metadata file containing segmented audio files with identifying timestamps, present species and information about the order within the file
#used for splitting the data, creating annotations
def segment_audio(input_dir, output_dir, segment_length=3000, labels_file="../src/BirdNET_GLOBAL_6K_V2.4_Labels_af.txt"):

    segments_dir = os.path.join(output_dir, "3s_segments_multi")

    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    
    species_mapping = birdnet_species_mapping(labels_file)
    segment_metadata = []
    
    segment_counters = defaultdict(int)

    for site in os.listdir(input_dir):
        site_path = os.path.join(input_dir, site)
        print(f'Checking {site_path}')

        if not os.path.isdir(site_path):
            print(f'Skipping non-dir item: {site}')
            continue

        recordings_path = os.path.join(site_path, "Recordings")
        annotations_path = os.path.join(site_path, "Raven Pro annotations")

        if not os.path.exists(recordings_path) or not os.path.exists(annotations_path):
            print(f"Skipping {site} - missing recordings or annotations")
            continue
        
        for annotation_file in os.listdir(annotations_path):
            if not annotation_file.endswith(".txt"):
                continue
                
            annotation_path = os.path.join(annotations_path, annotation_file)
            print(f'Processing {annotation_path}')
            
            try:
                annotation_data = pd.read_csv(annotation_path, delimiter="\t")
                
                if annotation_data.empty:
                    print(f"No annotations found for {annotation_file}")
                    continue

                wav_file = os.path.join(recordings_path, annotation_file.replace('.txt', '.wav'))
                
                if not os.path.exists(wav_file):
                    print(f"No wav file found for {annotation_file}")
                    continue
                
                audio = AudioSegment.from_file(wav_file)
                audio_duration_ms = len(audio)
                
                vocalizations = []
                for _, row in annotation_data.iterrows():
                    scientific_name = row['Species']
                    birdnet_species = species_mapping.get(scientific_name, scientific_name)

                    vocalizations.append({
                        'start_time': int(row['Begin Time (s)'] * 1000),
                        'end_time': int(row['End Time (s)'] * 1000),
                        'species': birdnet_species
                    })

                vocalizations.sort(key=lambda x: x['start_time'])
                
                if not vocalizations:
                    continue
                
                for segment_start in range(0, audio_duration_ms, segment_length//3):  #1 sec overlap
                    segment_end = min(segment_start + segment_length, audio_duration_ms)
                    
                    if segment_end - segment_start < 2000:
                        continue

                    overlapping_vocs = []
                    for voc in vocalizations:
                        if voc['start_time'] < segment_end and voc['end_time'] > segment_start:
                            overlapping_vocs.append(voc)
                    
                    if not overlapping_vocs:
                        if segment_counters['background'] < 100 and np.random.random() < 0.1: 
                            segment_type = 'background'
                            species = ['Noise_Noise']
                        else:
                            continue
                    else:
                        species = [voc['species'] for voc in overlapping_vocs]
                        
                        if segment_start == 0:
                            segment_type = 'file_start'
                        elif segment_end == audio_duration_ms:
                            segment_type = 'file_end'
                        else:
                            segment_type = 'file_middle'
                    
                    segment = audio[segment_start:segment_end]
                    
                    if len(segment) < segment_length:
                        padding = AudioSegment.silent(duration=segment_length - len(segment))
                        segment = segment + padding
                    
                    base_name = os.path.splitext(annotation_file)[0]
                    segment_counter = segment_counters[f"{base_name}_{segment_type}"]
                    segment_counters[f"{base_name}_{segment_type}"] += 1
                    
                    segment_name = f"{base_name}_{segment_start}_{segment_end}_{segment_type}_{segment_counter}.wav"
                    segment_path = os.path.join(segments_dir, segment_name)
                    
                    segment.export(segment_path, format="wav")
                    
                    record = {
                        'segment_name': segment_name,
                        'species': species,
                        'site': site,
                        'original_file': annotation_file.replace('.txt', '.wav'),
                        'segment_type': segment_type,
                        'segment_start_ms': segment_start,
                        'segment_end_ms': segment_end,
                    }
                    
                    if segment_type != 'background':
                        voc_start_ms = max(min(voc['start_time'] for voc in overlapping_vocs), segment_start)
                        voc_end_ms = min(max(voc['end_time'] for voc in overlapping_vocs), segment_end)
                        
                        record.update({
                            'voc_start_ms': voc_start_ms,
                            'voc_end_ms': voc_end_ms,
                            'voc_relative_start_ms': voc_start_ms - segment_start,
                            'voc_relative_end_ms': voc_end_ms - segment_start,
                            'pre_context_ms': voc_start_ms - segment_start,
                            'post_context_ms': segment_end - voc_end_ms,
                        })
                    else:
                        record.update({
                            'voc_start_ms': None,
                            'voc_end_ms': None,
                            'voc_relative_start_ms': None,
                            'voc_relative_end_ms': None,
                            'pre_context_ms': None,
                            'post_context_ms': None,
                        })
                    
                    segment_metadata.append(record)
                    
            except Exception as e:
                print(f"Error processing {annotation_file}: {e}")
                continue
            
    metadata_df = pd.DataFrame(segment_metadata)
    metadata_file = os.path.join(output_dir, "segment_metadata_multi.csv")
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Created {len(segment_metadata)} segments:")
    print(metadata_df['segment_type'].value_counts())
    
    all_species = []
    for species_list in metadata_df['species']:
        if isinstance(species_list, list):
            all_species.extend(species_list)
        else:
            all_species.append(species_list)
    
    species_counts = pd.Series(all_species).value_counts()
    species_counts.to_csv(os.path.join(output_dir, "segment_species_distribution.csv"), index=True, header=["count"])
    
    if 'species' in metadata_df.columns:
        overlap_counts = metadata_df['species'].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        ).value_counts().sort_index()
        
        print("\nOverlap Statistics:")
        for count, num_segments in overlap_counts.items():
            print(f"  {count} species: {num_segments} segments")
    
    return metadata_df

def birdnet_species_mapping(labels_file):
    species_mapping = {}
    
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                    
                parts = line.split('_', 1)  
                if len(parts) == 2:
                    scientific_name = parts[0]
                    birdnet_format = line  
                    
                    species_mapping[scientific_name] = birdnet_format
                    species_mapping[scientific_name.lower()] = birdnet_format
        
        print(f"Loaded {len(species_mapping)} species mappings")
        
        species_mapping['Background'] = 'Noise_Noise'
        species_mapping['Noise'] = 'Noise_Noise'
        
        return species_mapping
    except Exception as e:
        print(f"Error loading BirdNet labels: {e}")
        return {'Background': 'Noise_Noise'}
    
if __name__ == "__main__":
    segment_audio(
        input_dir="../dataset/filtered_dataset",
        output_dir="../processed_dataset",
        segment_length=3000)