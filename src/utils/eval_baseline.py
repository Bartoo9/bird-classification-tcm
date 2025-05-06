import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    accuracy_score
)

def evaluate_baseline_model(
    model, X_test, y_test, encoder, label_encoder, 
    device, output_dir=None, model_name="model", 
    subset=False, short_species=None, medium_species=None, long_species=None
):
    
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
    
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    metrics_df = []
    class_names = label_encoder.classes_
    
    for i, class_name in enumerate(class_names):
        # Per-class metrics
        true_binary = y_test_onehot[:, i]
        pred_proba = y_pred_probs[:, i]
        pred_binary = (pred_proba > 0.5).astype(int)
        
        if subset:
            if class_name in short_species:
                category = 'short'
            elif class_name in medium_species:
                category = 'medium'
            elif class_name in long_species:
                category = 'long'
            else:
                category = 'unknown'
        else:
            category = 'all'
        
        class_metrics = {
            'class': class_name,
            'auroc': roc_auc_score(true_binary, pred_proba),
            'ap': average_precision_score(true_binary, pred_proba),
            'f1': f1_score(true_binary, pred_binary),
            'support': np.sum(true_binary),
            'category': category
        }
        metrics_df.append(class_metrics)
    
    metrics_df = pd.DataFrame(metrics_df)
    
    results = {
        'accuracy': accuracy_score(y_test, np.argmax(y_pred_probs, axis=1)),
        'mAP': average_precision_score(y_test_onehot, y_pred_probs, average='macro'),
        'macro_f1': f1_score(y_test_onehot, (y_pred_probs > 0.5).astype(int), average='macro'),
        'macro_auroc': roc_auc_score(y_test_onehot, y_pred_probs, average='macro'),
        'num_parameters': num_params
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_df.to_csv(os.path.join(output_dir, f'{model_name}_metrics.csv'), index=False)
        
        with open(os.path.join(output_dir, f'{model_name}_summary.txt'), 'w') as f:
            f.write(f"Number of parameters: {num_params:,}\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"mAP: {results['mAP']:.4f}\n")
            f.write(f"macro F1: {results['macro_f1']:.4f}\n")
            f.write(f"macro AUROC: {results['macro_auroc']:.4f}\n")
        
        create_baseline_figures(metrics_df, output_dir, model_name)
    
    return results, metrics_df

def create_baseline_figures(metrics_df, output_dir, model_name):
    """Create clean, publication-quality figures focusing on mAP scores."""
    plt.style.use('seaborn-v0_8-paper')
    
    colors = {
        'short': '#2ecc71',    
        'medium': '#3498db',  
        'long': '#e74c3c'      
    }
    
    plt.figure(figsize=(10, 6))
    for category in ['short', 'medium', 'long']:
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['support'], cat_data['ap'],
                   label=f'{category.title()} vocalizations',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('number of training samples', fontsize=8)
    plt.ylabel('mean average precision', fontsize=8)
    plt.title('mAP vs. sample Size', fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_mAP_perf_vs_samples.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    for category in ['short', 'medium', 'long']:
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['support'], cat_data['f1'],
                   label=f'{category.title()} vocalizations',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('number of training samples', fontsize=8)
    plt.ylabel('f1-score', fontsize=12)
    plt.title('f1 vs. sample Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_f1_perf_vs_samples.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Category-wise Performance
    plt.figure(figsize=(10, 6))
    
    # Ensure correct category order
    ordered_categories = ['short', 'medium', 'long']
    category_means = metrics_df.groupby('category')[['f1', 'ap', 'auroc']].mean()
    category_means = category_means.reindex(ordered_categories)
    
    x = np.arange(len(ordered_categories))
    width = 0.20
    
    # Define colors for metrics
    metric_colors = {
        'f1': '#2ecc71',    # Green
        'ap': '#3498db',    # Blue
        'auroc': '#e74c3c'  # Red
    }
    
    # Plot bars for each metric
    metrics = [('f1', 'F1'), ('ap', 'AP'), ('auroc', 'AUROC')]
    bars = []
    
    for i, (metric_col, metric_name) in enumerate(metrics):
        data = category_means[metric_col]
        bar = plt.bar(x + (i-1)*width, data, width,
                     label=metric_name, alpha=0.7,
                     color=metric_colors[metric_col])
        bars.append(bar)
    
    plt.xlabel('vocalisation length', fontsize=8)
    plt.ylabel('Score', fontsize=8)
    plt.title('metrics by vocalisation length', fontsize=8)
    plt.xticks(x, [cat.title() for cat in ordered_categories])
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_category_performance.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()