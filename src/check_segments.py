import pandas as pd
import os 

#util to check segment count makes sense
def check_segments(annotation_file):
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return

    df = pd.read_csv(annotation_file)
    print(f"Total segments: {len(df)}")

    required_columns = ['segment name', 'label', 'Begin Time (s)', 'End Time (s)']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns in the annotation file. Found columns: {df.columns.tolist()}")
        return

    species_counts = df.groupby('label').size().reset_index(name='segment count')
    print("\nSegment counts by species:")
    print(species_counts)

    print("\nSpecies with fewer than 5 segments:")
    print(species_counts[species_counts['segment count'] < 5])

annotation_file = "../dataset/3s_segment_annotations.csv"

check_segments(annotation_file)