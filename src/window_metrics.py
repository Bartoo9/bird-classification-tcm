import pandas as pd
from temporal_models import train_model
import os
import wandb

WANDB_INITIALIZED = False

#calls and logs the model training for different window sizes
def analyze_window_sizes(ctx_type='bidirectional'):
    global WANDB_INITIALIZED

    if not WANDB_INITIALIZED:
        wandb.init(project="bird-classification-tcm", 
                  name=f"window_analysis_{ctx_type}",
                  config={
                      "context_type": ctx_type,
                      "window_sizes": [1, 3, 5, 7, 9]
                  })
        WANDB_INITIALIZED = True
        print("wandb initialized successfully!")

    window_sizes = [1, 3, 5, 7, 9]
    output_base = f"../results/ctx_results/window_analysis_{ctx_type}"
    os.makedirs(output_base, exist_ok=True)

    common_params = {
        'model_type': 'gru',
        'embeddings_dir': "../dataset/dataset_ctx_multi/embeddings_multi",
        'annotations_path': "../dataset/dataset_ctx_multi/embedding_annotations_multi.csv",
        'hidden_dim': 64,
        'num_layers': 1,
        'subset': True,
        'return_detailed_metrics': True,
        'epochs': 50,
        'ctx_type': ctx_type,
        'downsample': False,
        'dropout': 0.5,
        'threshold': 3000,
        'sigmoid': 0.9,
        'use_wandb': False
    }

    results = []
    window_sizes_list = []
    map_scores = []
    f1_scores = []

    for window_size in window_sizes:
        print(f"\n=== Testing Window Size {window_size} ===")
        output_dir = os.path.join(output_base, f"window_{window_size}")
        os.makedirs(output_dir, exist_ok=True)
        annotations_path = common_params['annotations_path']
        print(f"Checking annotations file: {annotations_path}")
        print(f"File exists: {os.path.exists(annotations_path)}")
    
        model, acc, metrics, val_loader = train_model(
            **common_params,
            window_size=window_size,
            output_dir=output_dir, 
        )

        window_sizes_list.append(window_size)
        map_score = metrics.get('mAP', 0)
        f1_score = metrics.get('macro_f1', 0)
        map_scores.append(map_score)
        f1_scores.append(f1_score)

        try:
            wandb.log({
                "window_size": window_size,
                "test_map": map_score,
                "test_f1": f1_score,
                "test_accuracy": acc
            })

        except Exception as e:
            if not WANDB_INITIALIZED:
                wandb.init(project="bird-classification-tcm", name=f"window_analysis_{ctx_type}_retry")
                WANDB_INITIALIZED = True
                wandb.log({
                    "window_size": window_size,
                    "test_map": map_score,
                    "test_f1": f1_score,
                    "test_accuracy": acc
                })
            
    try:
        wandb.log({
            "window_size_map_curve": wandb.Table(
                columns=["window_size", "mAP"],
                data=[[w, m] for w, m in zip(window_sizes_list, map_scores)]
            ),
            "window_size_f1_curve": wandb.Table(
                columns=["window_size", "F1"],
                data=[[w, f] for w, f in zip(window_sizes_list, f1_scores)]
            )
        })
    except Exception as e:
        print(f"Error logging summary to wandb: {e}")

    #results
    results.append({
        'window_size': window_size,
        'accuracy': acc,
        'metrics': metrics
    })

    pd.DataFrame([{
        'window_size': window_size,
        'accuracy': acc,
        'mAP': metrics.get('mAP', metrics.get('ap', 0)) if isinstance(metrics, pd.DataFrame) else metrics.get('mAP', metrics.get('mean_ap', 0)),           
        'f1': metrics['f1'].mean() if isinstance(metrics, pd.DataFrame) else metrics['macro_f1'],  
        'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    

if __name__ == "__main__":
    analyze_window_sizes()