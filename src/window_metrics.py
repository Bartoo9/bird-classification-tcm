import pandas as pd
from temporal_models import train_model
import os
import wandb

def analyze_window_sizes():
    window_sizes = [3, 5, 7]
    output_base = "../results/window_analysis_unidirectional"
    cache_dir = "../cached_data"
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    wandb.login()

    common_params = {
        'model_type': 'gru',
        'embeddings_dir': "../dataset/augmented_embeddings",
        'annotations_path': "../dataset/_augmented_embedding_annotations.csv",
        'hidden_dim': 128,
        'num_layers': 1,
        'subset': True,
        'return_detailed_metrics': True,
        'epochs': 50,
        'use_smote': True, 
        'save_resampled': True,
        'resampled_cache_dir': cache_dir
    }

    results = []
    
    for window_size in window_sizes:
        print(f"\n=== Testing Window Size {window_size} ===")
        output_dir = os.path.join(output_base, f"window_{window_size}")
        os.makedirs(output_dir, exist_ok=True)
        
        #
        model, acc, metrics, val_loader = train_model(
            **common_params,
            window_size=window_size,
            output_dir=output_dir, 
        )
        
        #results
        results.append({
            'window_size': window_size,
            'accuracy': acc,
            'metrics': metrics
        })

        pd.DataFrame([{
            'window_size': window_size,
            'accuracy': acc,
            'mAP': metrics['mAP'].mean() if isinstance(metrics, pd.DataFrame) else metrics['mAP'],            
            'macro_f1': metrics['macro_f1'].mean() if isinstance(metrics, pd.DataFrame) else metrics['macro_f1'],  
            'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        

if __name__ == "__main__":
    analyze_window_sizes()