import os 
from pydub import AudioSegment
import pandas as pd 

#BirdNet requires 3 second audio segments
#longer segments will be split into 3 second segments with padding if necessary
#these segments are then saved into a new dir - containing 3s segments along with new annotations labeling each segment

def segment_audio(input_dir, output_dir, segment_length=3000):
    segments_dir = os.path.join(output_dir, "3s_segments")

    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    
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

                audio = AudioSegment.from_file(wav_file)
            
                for _, row in annotation_data.iterrows():
                    start_time = int(row['Begin Time (s)'] * 1000)
                    end_time = int(row['End Time (s)'] * 1000)
                    label = row['Species']

                    print(f"Segmenting from {start_time} to {end_time}... {label}")

                    vocalization = audio[start_time:end_time]

                    if len(vocalization) < segment_length:
                        padding = AudioSegment.silent(duration=segment_length - len(vocalization))
                        vocalization = vocalization + padding
                        
                
                    for i in range(0, len(vocalization), segment_length):
                        segment = vocalization[i:i + segment_length]

                        if len(segment) < segment_length:
                            padding = AudioSegment.silent(duration=segment_length - len(segment))
                            segment = segment + padding
                        
                        segment_name = f"{os.path.splitext(annotation_file)[0]}_segment_{start_time//1000}_{i//segment_length}.wav"
                        segment_path = os.path.join(segments_dir, segment_name)
                        segment.export(segment_path, format="wav")
                        print(f"Segment saved: {segment_path}")

                        segment_metadata.append({
                            'segment name': segment_name,
                            'label': label,
                            'Begin Time (s)': start_time + i,
                            'End Time (s)': start_time + i + len(segment)
                        })

    annotation_file_path = os.path.join(output_dir, "3s_segment_annotations.csv")
    pd.DataFrame(segment_metadata).to_csv(annotation_file_path, index=False)

segment_audio(input_dir="../dataset/filtered_dataset",
              output_dir="../dataset",
              segment_length=3000)