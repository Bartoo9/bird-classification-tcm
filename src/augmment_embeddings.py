import bioacoustics_model_zoo as bmz
import numpy as np
import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy import signal

#used this before but wasnt used in the final evaluations
def augment_and_embed_selected_species(audio_dir, output_dir, annotation_file, 
                                       target_samples_per_class=None, plot_distribution=True, 
                                       max_augmentations_per_file=50, min_augmentations_per_file=5):
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    annotations_df = pd.read_csv(annotation_file)
    print(f"Loaded {len(annotations_df)} annotations")
    
    filtered_df = annotations_df[annotations_df['label'].isin(selected_species)]
    
    if len(filtered_df) == 0:
        raise ValueError("No annotations found for the selected species!")
    
    class_counts = filtered_df['label'].value_counts()
    print("Original class distribution:")
    for species in selected_species:
        count = class_counts.get(species, 0)
        if species in short_species:
            category = "short"
        elif species in medium_species:
            category = "medium"
        else:
            category = "long"
        print(f"  {species} ({category}): {count} samples")
    
    if plot_distribution:
        plt.figure(figsize=(15, 8))
        plot_data = []
        for species in selected_species:
            count = class_counts.get(species, 0)
            if species in short_species:
                category = "Short"
            elif species in medium_species:
                category = "Medium"
            else:
                category = "Long"
            plot_data.append((species, count, category))
        
        plot_df = pd.DataFrame(plot_data, columns=['Species', 'Count', 'Category'])
        plot_df = plot_df.sort_values('Count')
        
        colors = {'Short': 'skyblue', 'Medium': 'orange', 'Long': 'green'}
        ax = plt.barh(plot_df['Species'], plot_df['Count'], 
                     color=[colors[cat] for cat in plot_df['Category']])
        plt.xlabel('Number of Samples')
        plt.title('Original Class Distribution')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[cat], label=cat) for cat in colors]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'original_distribution.png'))
    
    birdnet = bmz.BirdNET()
    
    files_to_process = []
    file_labels = []
    augmentation_info = []

    for class_name in selected_species:
        class_files = filtered_df[filtered_df['label'] == class_name]['segment name'].tolist()
        if not class_files:
            print(f"Warning: No files found for {class_name}")
            continue
            
        count = len(class_files)
        
        for file in class_files:
            file_path = os.path.join(audio_dir, file)
            if os.path.exists(file_path):
                files_to_process.append(file_path)
                file_labels.append(class_name)
                augmentation_info.append("original")
        
        if count < target_samples_per_class:
            augmentations_needed = target_samples_per_class - count
            
            augmentation_intensity = "medium"
            if class_name in short_species:
                augmentation_intensity = "high"
            elif class_name in long_species:
                augmentation_intensity = "low"
            
            print(f"Class {class_name}: Need {augmentations_needed} augmentations " 
                  f"(intensity: {augmentation_intensity})")
            
            aug_per_file = min(
                max_augmentations_per_file,  
                max(
                    min_augmentations_per_file,  
                    int(np.ceil(augmentations_needed / count) * 1.1)  
                )
            )
            
            print(f"  Creating up to {aug_per_file} augmentations per file")
            
            augmented_count = 0
            files_processed = 0
            
            while augmented_count < augmentations_needed:
                random.shuffle(class_files)
                
                for file in class_files:
                    file_path = os.path.join(audio_dir, file)
                    if not os.path.exists(file_path):
                        continue
                    
                    remaining = augmentations_needed - augmented_count
                    augs_for_this_file = min(aug_per_file, remaining)
                    
                    for i in range(augs_for_this_file):
                        aug_id = f"{files_processed}_{i}"
                        
                        augmented_file = create_augmented_audio(
                            file_path, 
                            os.path.join(output_dir, f"aug_{os.path.basename(file).replace('.wav', '')}_{aug_id}.wav"),
                            intensity=augmentation_intensity
                        )
                        
                        files_to_process.append(augmented_file)
                        file_labels.append(class_name)
                        augmentation_info.append(f"augmented_{aug_id}")
                        augmented_count += 1
                    
                    files_processed += 1
                    
                    if augmented_count >= augmentations_needed:
                        break
                
                if augmented_count < augmentations_needed:
                    
                    if files_processed > count * 3:
                        break
    
    
    batch_size = 32
    for i in tqdm(range(0, len(files_to_process), batch_size)):
        batch_files = files_to_process[i:i+batch_size]
        batch_labels = file_labels[i:i+batch_size]
        batch_info = augmentation_info[i:i+batch_size]
        
        embeddings = birdnet.embed(batch_files, return_dfs=False, return_preds=False)
        
        for j, (emb, label, info) in enumerate(zip(embeddings, batch_labels, batch_info)):
            file_name = os.path.splitext(os.path.basename(batch_files[j]))[0]
            output_path = os.path.join(output_dir, f"{file_name}.npy")
            np.save(output_path, emb)
    
    embedding_files = [file for file in os.listdir(output_dir) if file.endswith('.npy')]
    embedding_data = []
    
    for i, file in enumerate(files_to_process):
        file_name = os.path.splitext(os.path.basename(file))[0]
        embedding_name = f"{file_name}.npy"
        if embedding_name in embedding_files:
            species_category = "unknown"
            if file_labels[i] in short_species:
                species_category = "short"
            elif file_labels[i] in medium_species:
                species_category = "medium"
            elif file_labels[i] in long_species:
                species_category = "long"
                
            embedding_data.append({
                "embedding name": embedding_name,
                "label": file_labels[i],
                "augmentation": augmentation_info[i],
                "category": species_category
            })
    
    embedding_df = pd.DataFrame(embedding_data)
    output_csv = os.path.join(output_dir, "_augmented_embedding_annotations.csv")
    embedding_df.to_csv(output_csv, index=False)
    
    if plot_distribution:
        new_class_counts = embedding_df['label'].value_counts()
        plt.figure(figsize=(15, 8))
        
        plot_data = []
        for species in selected_species:
            count = new_class_counts.get(species, 0)
            if species in short_species:
                category = "Short"
            elif species in medium_species:
                category = "Medium"
            else:
                category = "Long"
            plot_data.append((species, count, category))
        
        plot_df = pd.DataFrame(plot_data, columns=['Species', 'Count', 'Category'])
        plot_df = plot_df.sort_values('Count')
        
        colors = {'Short': 'skyblue', 'Medium': 'orange', 'Long': 'green'}
        plt.barh(plot_df['Species'], plot_df['Count'], 
                color=[colors[cat] for cat in plot_df['Category']])
        plt.xlabel('Number of Samples')
        plt.title('Augmented Class Distribution')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[cat], label=cat) for cat in colors]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'augmented_distribution.png'))
        
    for species in selected_species:
        count = new_class_counts.get(species, 0)
        if species in short_species:
            category = "short"
        elif species in medium_species:
            category = "medium"
        else:
            category = "long"
        print(f"  {species} ({category}): {count} samples")
    
    return output_csv


def create_augmented_audio(input_file, output_file, intensity="medium"):
    from scipy import signal
    
    audio, sr = librosa.load(input_file, sr=None)
    
    if intensity == "low":
        pitch_range = (-1, 1)  # Less pitch shift
        stretch_range = (0.9, 1.1)  
        noise_range = (0.0005, 0.002)  
        gain_range = (0.8, 1.2)  
        filter_prob = 0.4  
    elif intensity == "high":
        pitch_range = (-3, 3)  # More pitch shift
        stretch_range = (0.8, 1.2)  
        noise_range = (0.001, 0.008)  
        gain_range = (0.7, 1.4) 
        filter_prob = 0.7  
    else:  
        pitch_range = (-2, 2)
        stretch_range = (0.85, 1.15)
        noise_range = (0.001, 0.005)
        gain_range = (0.75, 1.3)
        filter_prob = 0.5
    
    augmented = audio.copy()
    
    if random.random() > 0.3:
        pitch_shift = random.uniform(*pitch_range)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch_shift)
    
    #stretch time
    if random.random() > 0.3:
        stretch_factor = random.uniform(*stretch_range)
        augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
        if len(augmented) > len(audio):
            augmented = augmented[:len(audio)]
        elif len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
    
    #noise
    if random.random() > 0.4:
        noise_level = random.uniform(*noise_range)
        noise = np.random.normal(0, noise_level, size=len(augmented))
        augmented = augmented + noise
    
    #gain
    if random.random() > 0.5:
        gain = random.uniform(*gain_range)
        augmented = augmented * gain
    
    #bandpass filtering
    if random.random() > (1 - filter_prob):
        if random.random() > 0.5:
            cutoff = random.uniform(200, 500)
            nyq = 0.5 * sr
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(4, normal_cutoff, btype='highpass')
            augmented = signal.filtfilt(b, a, augmented)
        else:
            cutoff = random.uniform(3000, 6000)
            nyq = 0.5 * sr
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(4, normal_cutoff, btype='lowpass')
            augmented = signal.filtfilt(b, a, augmented)
    
    augmented = librosa.util.normalize(augmented)
    

    sf.write(output_file, augmented, sr)
    
    return output_file


if __name__ == '__main__':
    audio_dir = "../dataset/3s_segments"
    output_dir = "../dataset/augmented_embeddings"
    annotation_file = "../dataset/_segment_annotations.csv"
    
    augmented_annotation_file = augment_and_embed_selected_species(
        audio_dir=audio_dir,
        output_dir=output_dir,
        annotation_file=annotation_file,
        target_samples_per_class=1000  
    )