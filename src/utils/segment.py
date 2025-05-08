import os 
from pydub import AudioSegment
import pandas as pd 

#util to get scientific_common names for labels
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
        print(f"Sample mappings: {list(species_mapping.items())[:3]}")
        
        species_mapping['Background'] = 'Noise_Noise'
        species_mapping['Noise'] = 'Noise_Noise'
        
        return species_mapping
    except Exception as e:
        print(f"Error loading BirdNet labels: {e}")
        return {'Background': 'Noise_Noise'}
    
    
#BirdNet requires 3 second audio segments
#vocalisation segments are extracted from the files with the surrounding context noise
#these segments are then saved into a new dir - containing 3s segments along with new annotations labeling each segment

def segment_audio(input_dir, output_dir, segment_length=3000, labels_file="..\src\BirdNET_GLOBAL_6K_V2.4_Labels_af.txt"):
    segments_dir = os.path.join(output_dir, "3s_segments_ctx")

    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    
    species_mapping = birdnet_species_mapping(labels_file)
    segment_metadata = []

    for site in os.listdir(input_dir):
        site_path = os.path.join(input_dir, site)
        print(f'checking {site_path}')

        if not os.path.isdir(site_path):
            print(F'skipping non-dir item: {site}')
            continue

        recordings_path = os.path.join(site_path, "Recordings")
        annotations_path = os.path.join(site_path, "Raven Pro annotations")

        if not os.path.exists(recordings_path) or not os.path.exists(annotations_path):
            print(f"Skipping {site} - missing recordings or annotations")
            continue
        
        for annotation_file in os.listdir(annotations_path):
            if annotation_file.endswith(".txt"):
                annotation_path = os.path.join(annotations_path, annotation_file)
                print(f'checking {annotation_path}')
                annotation_data = pd.read_csv(annotation_path, delimiter="\t")

                if annotation_data.empty:
                    print(f"No annotations found for {annotation_file}")
                    continue

                wav_file = os.path.join(recordings_path, annotation_file.replace('.txt', '.wav'))
                
                if not os.path.exists(wav_file):
                    print(f"No wav file found for {annotation_file}")
                    continue
                
                #load audio file
                audio = AudioSegment.from_file(wav_file)
                audio_duration_ms = len(audio)\
                
                vocalizations = []
                for _, row in annotation_data.iterrows():
                    scientific_name = row['Species']
                    birdnet_species = species_mapping.get(scientific_name, scientific_name)

                    vocalizations.append({
                    'start_time': int(row['Begin Time (s)'] * 1000),
                    'end_time' : int(row['End Time (s)'] * 1000),
                    'species': birdnet_species,
                    'processed': False
                    })

                vocalizations.sort(key=lambda x: x['start_time'])

                for i,voc in enumerate(vocalizations):

                    if voc['processed']:
                        continue

                    start_time = voc['start_time']
                    end_time = voc['end_time']
                    species = voc['species']
                    voc_duration = end_time - start_time

                    print(f"Segmenting from {start_time} to {end_time}... {species}")

                    voc['processed'] = True

                    #short vocalizations
                    if voc_duration <= segment_length:
                        #center short vocalizations within the context
                        center_time = (start_time + end_time) // 2
                        segment_start = max(0, center_time - segment_length // 2)
                        segment_end = min(audio_duration_ms, segment_start + segment_length)

                        if segment_end - segment_start < segment_length:
                            segment_start = max(0, segment_end - segment_length)

                        segment = audio[segment_start:segment_end]

                        if len(segment) < segment_length:
                            padding = AudioSegment.silent(duration=segment_length - len(segment))
                            segment = segment + padding
                        
                        segment_name = f"{os.path.splitext(annotation_file)[0]}_centered.wav"
                        segment_path = os.path.join(segments_dir, segment_name)
                        segment.export(segment_path, format="wav")

                        segment_metadata.append({
                            'segment_name': segment_name,
                            'species': species,
                            'site': site,
                            'original_file': annotation_file.replace('.txt', '.wav'),
                            'segment_type': 'centered',
                            'segment_start_ms': segment_start,
                            'segment_end_ms': segment_end,
                            'voc_start_ms': start_time,
                            'voc_end_ms': end_time,
                            'voc_relative_start_ms': start_time - segment_start,
                            'voc_relative_end_ms': end_time - segment_start,
                            'pre_context_ms': start_time - segment_start,
                            'post_context_ms': segment_end - end_time})
                        
                    else:
                        first_segment_start = max(0, start_time - segment_length // 2)
                        first_segment_end = first_segment_start + segment_length
                        segment = audio[first_segment_start:first_segment_end]
                        segment_name = f"{os.path.splitext(annotation_file)[0]}_start.wav"
                        segment_path = os.path.join(segments_dir, segment_name)
                        segment.export(segment_path, format="wav")

                        segment_metadata.append({
                            'segment_name': segment_name,
                            'species': species,
                            'site': site,
                            'original_file': annotation_file.replace('.txt', '.wav'),
                            'segment_type': 'long_start',
                            'segment_start_ms': first_segment_start,
                            'segment_end_ms': first_segment_end,
                            'voc_start_ms': start_time,
                            'voc_end_ms': end_time,
                            'voc_relative_start_ms': start_time - first_segment_start,
                            'voc_relative_end_ms': min(first_segment_end, end_time) - first_segment_start,
                            'pre_context_ms': start_time - first_segment_start,
                            'post_context_ms': 0,
                        })

                        #middle segments with overlap 
                        if voc_duration > segment_length * 1.5:
                        
                            step = segment_length // 2
                            
                            for mid_start in range(start_time + step, end_time - step, step):
                                mid_end = mid_start + segment_length
                                
                                if mid_end > end_time:
                                    break 
                                
                                segment = audio[mid_start:mid_end]
                                segment_name = f"{os.path.splitext(annotation_file)[0]}_mid.wav"
                                segment_path = os.path.join(segments_dir, segment_name)
                                segment.export(segment_path, format="wav")
                                
                                segment_metadata.append({
                                    'segment_name': segment_name,
                                    'species': species,
                                    'site': site,
                                    'original_file': annotation_file.replace('.txt', '.wav'),
                                    'segment_type': 'long_middle',
                                    'segment_start_ms': mid_start,
                                    'segment_end_ms': mid_end,
                                    'voc_start_ms': start_time,
                                    'voc_end_ms': end_time,
                                    'voc_relative_start_ms': 0,
                                    'voc_relative_end_ms': segment_length,
                                    'pre_context_ms': 0,
                                    'post_context_ms': 0,
                                })
                        
                        last_segment_end = min(audio_duration_ms, end_time + segment_length // 2)
                        last_segment_start = last_segment_end - segment_length
                        
                        if last_segment_start < start_time + segment_length // 2:
                            continue
                        
                        segment = audio[last_segment_start:last_segment_end]
                        segment_name = f"{os.path.splitext(annotation_file)[0]}_end.wav"
                        segment_path = os.path.join(segments_dir, segment_name)
                        segment.export(segment_path, format="wav")
                        
                        segment_metadata.append({
                            'segment_name': segment_name,
                            'species': species,
                            'site': site,
                            'original_file': annotation_file.replace('.txt', '.wav'),
                            'segment_type': 'long_end',
                            'segment_start_ms': last_segment_start,
                            'segment_end_ms': last_segment_end,
                            'voc_start_ms': start_time,
                            'voc_end_ms': end_time,
                            'voc_relative_start_ms': max(0, start_time - last_segment_start),
                            'voc_relative_end_ms': end_time - last_segment_start,
                            'pre_context_ms': 0,
                            'post_context_ms': last_segment_end - end_time,
                        })
                
                #background segments
                if len(vocalizations) > 0:
                    gaps = []
                    
                    if vocalizations[0]['start_time'] >= segment_length:
                        gaps.append({
                            'start': 0,
                            'end': vocalizations[0]['start_time']
                        })
                    
                    for i in range(len(vocalizations) - 1):
                        gap_start = vocalizations[i]['end_time']
                        gap_end = vocalizations[i + 1]['start_time']
                        
                        if gap_end - gap_start >= segment_length:
                            gaps.append({
                                'start': gap_start,
                                'end': gap_end
                            })
                    
                    if audio_duration_ms - vocalizations[-1]['end_time'] >= segment_length:
                        gaps.append({
                            'start': vocalizations[-1]['end_time'],
                            'end': audio_duration_ms
                        })
                    
                    for i, gap in enumerate(gaps):
                        gap_len = gap['end'] - gap['start']
                        num_segments = min(3, max(1, gap_len // segment_length))
                        
                        for j in range(num_segments):
                            if num_segments == 1:
                                bg_center = (gap['start'] + gap['end']) // 2
                                bg_start = max(0, bg_center - segment_length // 2)
                            else:
                                portion = gap_len / num_segments
                                bg_start = int(gap['start'] + portion * j + portion / 2 - segment_length / 2)
                                bg_start = max(0, bg_start)
                            
                            bg_end = min(audio_duration_ms, bg_start + segment_length)
                            
                            if bg_end - bg_start < segment_length:
                                bg_start = max(0, bg_end - segment_length)
                            
                            bg_segment = audio[bg_start:bg_end]
                            
                            if len(bg_segment) < segment_length:
                                padding = AudioSegment.silent(duration=segment_length - len(bg_segment))
                                bg_segment = bg_segment + padding
                            
                            bg_name = f"{os.path.splitext(annotation_file)[0]}_Background.wav"
                            bg_path = os.path.join(segments_dir, bg_name)
                            
                            bg_segment.export(bg_path, format="wav")
                            
                            # Add metadata
                            segment_metadata.append({
                                'segment_name': bg_name,
                                'species': 'Noise_Noise',  
                                'site': site,
                                'original_file': annotation_file.replace('.txt', '.wav'),
                                'segment_type': 'background',
                                'segment_start_ms': bg_start,
                                'segment_end_ms': bg_end,
                                'voc_start_ms': None,
                                'voc_end_ms': None,
                                'voc_relative_start_ms': None,
                                'voc_relative_end_ms': None,
                                'pre_context_ms': None,
                                'post_context_ms': None,
                            })
    
    #save data
    metadata_df = pd.DataFrame(segment_metadata)
    metadata_file = os.path.join(output_dir, "segment_metadata.csv")
    metadata_df.to_csv(metadata_file, index=False)
    
    #summary
    print(f"Created {len(segment_metadata)} segments:")
    print(metadata_df['segment_type'].value_counts())
    print(f"\nMetadata saved to {metadata_file}")

    species_counts = metadata_df['species'].value_counts()
    species_counts.to_csv(os.path.join(output_dir, "segment_distribution.csv"), index=True, header=["count"])
    
    return metadata_df
                

                    
if __name__ == "__main__":
    segment_audio(
        input_dir="../dataset/filtered_dataset",
        output_dir="../processed_dataset",
        segment_length=3000)
    