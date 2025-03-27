import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd 

def compute_mel_spectrogram(audio_path, sample_rate = 48000, fft_window_ms = 10.7, 
                            hop_length_ms = 8, n_mels = 64, fmin = 150, fmax = 15000):

    y, sr = librosa.load(audio_path, sr=sample_rate)
    n_fft = int(sample_rate * fft_window_ms / 1000) # 512 samples
    hop_length = int(sample_rate * hop_length_ms / 1000) # 384 samples

    mel_spectrogram = librosa.feature.melspectrogram(
        y = y, sr=sr , n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax
    )

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram

def save_mel_spectrogram(mel_spect, output_path):

    plt.figure(figsize=(2.24, 2.24))
    plt.axis('off')
    librosa.display.specshow(mel_spect, sr=48000, x_axis='time', y_axis='mel', cmap='viridis')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_audio_files(annotations_path, audio_dir, output_dir): 
    
    segment_annotations = pd.read_csv(annotations_path)

    for _,row in segment_annotations.iterrows():
        segment_name = row['segment name']
        label = row['label']

        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        audio_path = os.path.join(audio_dir, segment_name)

        if os.path.exists(audio_path):
            mel_spectrogram = compute_mel_spectrogram(audio_path)
            output_path = os.path.join(label_dir, f"{os.path.splitext(segment_name)[0]}.png")
            save_mel_spectrogram(mel_spectrogram, output_path)
            print(f"Saved {output_path}")
        else:
            print(f"Audio file not found: {audio_path}")


if __name__ == '__main__':
    annotations_path = "../dataset/3s_segment_annotations.csv"
    audio_dir = "../dataset/3s_segments"
    output_dir = "../dataset/mel_spectrograms"

    process_audio_files(annotations_path, audio_dir, output_dir)