from sklearn.metrics import accuracy_score,auc, roc_curve, f1_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import pandas as pd 

def softmax(x, axis=1):

    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def evaluate_temporal_model(y_proba, y_true, class_names, output_dir=None, model_name=None, 
                   subset=False, short_species=None, medium_species=None, long_species=None,
                   train_counts=None):
    
    warnings.filterwarnings("ignore", message='no positive samples')
    warnings.filterwarnings("ignore", message='no positive class')

    if len(y_true) != len(y_proba):
        raise ValueError("Length mismatch between y_true and y_proba")
    
    y_true_onehot = np.zeros((len(y_true), len(class_names)))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    
    y_pred = np.argmax(y_proba, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    mAP = average_precision_score(y_true_onehot, y_proba, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    class_metrics = []
    for i, class_name in enumerate(class_names):
        true_binary = np.equal(y_true, i)  
        true_binary = true_binary.astype(int)
        pred_proba = y_proba[:, i]
        pred_binary = (pred_proba >= 0.5).astype(int)  
        
        #metrics
        f1 = f1_score(true_binary, pred_binary)
        ap = average_precision_score(true_binary, pred_proba)
        fpr, tpr, _ = roc_curve(true_binary, pred_proba)
        auroc = auc(fpr, tpr)
        support = np.sum(true_binary)
        
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
        
        train_support = train_counts[class_name] if train_counts else support

        class_metrics.append({
            'class': class_name,
            'macro_f1': f1,
            'mAP': ap,
            'auroc': auroc,
            'support': support,
            'category': category,
            'train_support': train_support
        })
    
    detailed_metrics_df = pd.DataFrame(class_metrics)
    
    if output_dir and model_name:
        detailed_metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), 
                                 index=False)
        figures(detailed_metrics_df, output_dir, model_name)

        summary_metrics = {
            'accuracy': accuracy,
            'mAP': mAP,
            'macro_f1': macro_f1,
        }

        pd.DataFrame([summary_metrics]).to_csv(
            os.path.join(output_dir, f"{model_name}_summary_metrics.csv"), 
            index=False
        )

    return detailed_metrics_df, {
        'accuracy': accuracy,
        'mAP': mAP,
        'macro_f1': macro_f1
    }

def figures(metrics_df, output_dir, model_name):
    """Create publication-quality figures matching baseline model style."""
    plt.style.use('seaborn-v0_8-paper')

    colors = {
        'short': '#2ecc71',    
        'medium': '#3498db',   
        'long': '#e74c3c'      
    }
    #map vs samples
    plt.figure(figsize=(10, 6))
    for category in ['short', 'medium', 'long']:
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['train_support'], cat_data['mAP'],
                   label=f'{category.title()} Vocalizations',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('Mean Average Precision', fontsize=12)
    plt.title('Performance vs. Sample Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_mAP_vs_samples.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    #f1 vs samples
    plt.figure(figsize=(10, 6))
    for category in ['short', 'medium', 'long']:
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['train_support'], cat_data['macro_f1'],
                   label=f'{category.title()} Vocalizations',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Performance vs. Sample Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_f1_vs_samples.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    #bars per category
    plt.figure(figsize=(10, 6))
    
    ordered_categories = ['short', 'medium', 'long']
    category_means = metrics_df.groupby('category')[['macro_f1', 'mAP', 'auroc']].mean()
    category_means = category_means.reindex(ordered_categories)
    
    x = np.arange(len(ordered_categories))
    width = 0.20
    
    metric_colors = {
        'macro_f1': '#2ecc71',    
        'mAP': '#3498db',    
        'auroc': '#e74c3c'  
    }
    
    metrics = [('f1', 'F1'), ('ap', 'AP'), ('auroc', 'AUROC')]
    bars = []
    
    for i, (metric_col, metric_name) in enumerate(metrics):
        data = category_means[metric_col]
        bar = plt.bar(x + (i-1)*width, data, width,
                     label=metric_name, alpha=0.7,
                     color=metric_colors[metric_col])
        bars.append(bar)
    
    plt.xlabel('Vocalization Length', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics by Category', fontsize=14)
    plt.xticks(x, [cat.title() for cat in ordered_categories])
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_category_performance.pdf"),
                dpi=300, bbox_inches='tight')
    plt.close()

    summary_stats = metrics_df.groupby('category').agg({
        'f1': ['mean', 'std', 'min', 'max'],
        'ap': ['mean', 'std'],
        'auroc': ['mean', 'std'],
        'support': ['sum', 'mean']
    }).round(3)
    
    summary_stats.columns = [f'{col[0]}_{col[1]}' for col in summary_stats.columns]
    summary_stats.to_csv(os.path.join(output_dir, f"{model_name}_summary_stats.csv"))

    return summary_stats