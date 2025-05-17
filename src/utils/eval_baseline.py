#function for evaluating baseline models for the multilabel classification
#bunch fo graphs, its a mess 

def evaluate_baseline_model(
    model, X_test, y_test, mlb, 
    device, output_dir=None, model_name="model", threshold=0.5,
    subset=False, short_species=None, medium_species=None, long_species=None
):
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, f1_score, 
        precision_score, recall_score
    )
    
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
    
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    metrics_df = []
    class_names = mlb.classes_
    
    #per class metrics
    for i, class_name in enumerate(class_names):
        true_binary = y_test[:, i]
        pred_proba = y_pred_probs[:, i]
        pred_binary = y_pred_binary[:, i]
        
        if np.sum(true_binary) == 0:
            continue
        
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
        
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        
        class_metrics = {
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'auroc': roc_auc_score(true_binary, pred_proba) if np.sum(true_binary) > 0 and np.sum(true_binary) < len(true_binary) else 0.5,
            'ap': average_precision_score(true_binary, pred_proba),
            'f1': f1_score(true_binary, pred_binary, zero_division=0),
            'support': np.sum(true_binary),
            'category': category
        }
        metrics_df.append(class_metrics)
    
    metrics_df = pd.DataFrame(metrics_df)
    
    #overall metrics for the model
    results = {
        'samples_f1': f1_score(y_test, y_pred_binary, average='samples', zero_division=0),
        'micro_f1': f1_score(y_test, y_pred_binary, average='micro', zero_division=0),
        'macro_f1': f1_score(y_test, y_pred_binary, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test, y_pred_binary, average='weighted', zero_division=0),
        'mAP': average_precision_score(y_test, y_pred_probs, average='macro'),
        'micro_auroc': roc_auc_score(y_test, y_pred_probs, average='micro'),
        'macro_auroc': roc_auc_score(y_test, y_pred_probs, average='macro'),
        'num_parameters': num_params
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_df.to_csv(os.path.join(output_dir, f'{model_name}_metrics.csv'), index=False)
        
        with open(os.path.join(output_dir, f'{model_name}_summary.txt'), 'w') as f:
            f.write(f"Number of parameters: {num_params:,}\n")
            f.write(f"Samples F1: {results['samples_f1']:.4f}\n")
            f.write(f"Micro F1: {results['micro_f1']:.4f}\n")
            f.write(f"Macro F1: {results['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {results['weighted_f1']:.4f}\n")
            f.write(f"mAP: {results['mAP']:.4f}\n")
            f.write(f"Micro AUROC: {results['micro_auroc']:.4f}\n")
            f.write(f"Macro AUROC: {results['macro_auroc']:.4f}\n")
        
        create_multi_baseline_figures(metrics_df, output_dir, model_name)
        
        #co occurence matrix
        if len(y_test) > 0:
            create_cooccurrence_matrix(y_test, y_pred_binary, class_names, output_dir, model_name)
    
    return results, metrics_df

#make some figures when testing the model
def create_multi_baseline_figures(metrics_df, output_dir, model_name):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("pastel")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    colors = {
        'short': '#a6cee3',    
        'medium': '#b2df8a',   
        'long': '#fdbf6f',     
        'unknown': '#95a5a6',  
        'all': '#9b59b6'       
    }
    
    #check whether the sample size is skewing the results (mAP)
    plt.figure(figsize=(10, 6))
    
    categories = metrics_df['category'].unique()
    category_order = {'short': 0, 'medium': 1, 'long': 2}
    categories = sorted(categories, key=lambda x: category_order.get(x, 99))

    for category in categories:
        if category not in colors:
            continue
            
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['support'], cat_data['ap'],
                   label=f'{category.title()}',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('Number of positive samples', fontsize=12)
    plt.ylabel('Average Precision', fontsize=12)
    plt.title('Average Precision vs. Sample Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_mAP_perf_vs_samples.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    #f1 scores vs sample sizes
    plt.figure(figsize=(10, 6))
    
    for category in categories:
        if category not in colors:
            continue
            
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['support'], cat_data['f1'],
                   label=f'{category.title()}',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('Number of positive samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs. Sample Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_f1_perf_vs_samples.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    #precision/recall
    plt.figure(figsize=(10, 6))
    
    for category in categories:
        if category not in colors:
            continue
            
        cat_data = metrics_df[metrics_df['category'] == category]
        plt.scatter(cat_data['precision'], cat_data['recall'],
                   label=f'{category.title()}',
                   color=colors[category],
                   alpha=0.7, s=100)
    
    plt.xlabel('Precision', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Precision vs. Recall by Category', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_precision_recall.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    #top and bottom performers
    metrics_df_sorted = metrics_df.sort_values('f1', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_df = metrics_df_sorted.head(15)
    plt.barh(top_df['class'], top_df['f1'], color=[colors.get(c, '#9b59b6') for c in top_df['category']])
    plt.xlabel('F1 Score', fontsize=12)
    plt.title('Top 15 Species by F1 Score', fontsize=14)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_top_species.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    if len(metrics_df) > 15:
        plt.figure(figsize=(12, 8))
        bottom_df = metrics_df_sorted.tail(15)
        plt.barh(bottom_df['class'], bottom_df['f1'], color=[colors.get(c, '#9b59b6') for c in bottom_df['category']])
        plt.xlabel('F1 Score', fontsize=12)
        plt.title('Bottom 15 Species by F1 Score', fontsize=14)
        plt.xlim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.6, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_bottom_species.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    #category wise performances 
    if 'unknown' in metrics_df['category'].unique():
        metrics_df = metrics_df[metrics_df['category'] != 'unknown']

    category_means = metrics_df.groupby('category')[['f1', 'ap', 'auroc']].mean()
    
    plt.figure(figsize=(10, 6))
    category_means[['f1', 'ap', 'auroc']].plot(kind='bar', rot=0, figsize=(10, 6), 
                                                color=['#a6cee3', '#b2df8a', '#fdbf6f'])
    plt.title('Average Performance by Category', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.legend(['F1 Score', 'Average Precision', 'AUROC'])
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_category_performance.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_cooccurrence_matrix(y_true, y_pred, class_names, output_dir, model_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    
    true_cooccur = np.dot(y_true.T, y_true) / y_true.shape[0]
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(true_cooccur, dtype=bool))
    sns.heatmap(true_cooccur, mask=mask, cmap='viridis', 
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, annot=False, square=True)
    plt.title('Species Co-occurrence in Ground Truth', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_true_cooccurrence.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    pred_cooccur = np.dot(y_pred.T, y_pred) / y_pred.shape[0]
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(pred_cooccur, dtype=bool))
    sns.heatmap(pred_cooccur, mask=mask, cmap='viridis', 
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, annot=False, square=True)
    plt.title('Species Co-occurrence in Predictions', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_pred_cooccurrence.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    diff_cooccur = pred_cooccur - true_cooccur
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(diff_cooccur, dtype=bool))
    sns.heatmap(diff_cooccur, mask=mask, cmap='coolwarm', 
                xticklabels=class_names, yticklabels=class_names,
                vmin=-1, vmax=1, center=0, annot=False, square=True)
    plt.title('Difference in Co-occurrence (Predictions - Ground Truth)', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_cooccurrence_diff.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()