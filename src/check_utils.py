# # check how many files are 3s_segments folder 
# # and how many files are in 3s_segment_annotations.csv
# # and check if they are same or not
# import os 
# import pandas as pd 

# print(len(os.listdir("../dataset/3s_segments")))

# df = pd.read_csv("../dataset/3s_segment_annotations.csv")
# print(len(df))

# # check how many spectrograms are in the spectrograms folder
# # need to go into every folder and check how many files are there

# for folder in os.listdir("../dataset/mel_spectrograms"):
#     total = sum([len(os.listdir(f"../dataset/mel_spectrograms/{folder}")) for folder in os.listdir("../dataset/mel_spectrograms")])

# print(total)

# # 38d074
# # 41462
# # 41188

import pandas as pd
import os

import os
import pandas as pd

def check_spectrogram_annotations(annotation_file, spectrogram_dir, output_mismatch_file="spectrogram_annotation_mismatches.csv"):
    """
    Checks if the spectrograms in the spectrograms folder match the annotations in the CSV file.

    Args:
        annotation_file (str): Path to the 3s_segment_annotations.csv file.
        spectrogram_dir (str): Path to the spectrograms folder.
        output_mismatch_file (str): Path to save the mismatch CSV file.
    """
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return

    if not os.path.exists(spectrogram_dir):
        print(f"Spectrogram directory not found: {spectrogram_dir}")
        return

    # Load the annotation file
    df = pd.read_csv(annotation_file)
    annotation_segments = set(df['segment name'].str.replace(".wav", ""))  # Remove .wav extension
    print(f"Total annotated segments: {len(annotation_segments)}")

    # Get the list of spectrogram files
    spectrogram_files = set()
    for folder in os.listdir(spectrogram_dir):
        folder_path = os.path.join(spectrogram_dir, folder)
        if os.path.isdir(folder_path):
            spectrogram_files.update([os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith(".png")])
    print(f"Total spectrograms found: {len(spectrogram_files)}")

    # Compare the two sets
    missing_spectrograms = annotation_segments - spectrogram_files
    extra_spectrograms = spectrogram_files - annotation_segments

    # Output concise results
    if len(missing_spectrograms) == 0 and len(extra_spectrograms) == 0:
        print("All spectrograms match the annotation file.")
    else:
        print("Mismatch detected!")
        print(f"Missing spectrograms: {len(missing_spectrograms)}")
        print(f"Extra spectrograms: {len(extra_spectrograms)}")

        # Save mismatches to a CSV file
        with open(output_mismatch_file, mode='w', newline='') as file:
            file.write("Type,Name\n")
            for spectrogram in missing_spectrograms:
                file.write(f"Missing,{spectrogram}\n")
            for spectrogram in extra_spectrograms:
                file.write(f"Extra,{spectrogram}\n")
        print(f"Mismatches saved to: {output_mismatch_file}")

if __name__ == "__main__":
    annotation_file = "../dataset/3s_segment_annotations.csv"
    segments_dir = "../dataset/3s_segments"

    print("Checking segments...")
    check_spectrogram_annotations(annotation_file, segments_dir)