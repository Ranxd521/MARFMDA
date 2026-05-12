import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score,
    confusion_matrix
)

def evaluate_all():
    res_dir = 'ablations/results_gpt'
    metrics_list = []
    
    for f in os.listdir(res_dir):
        if f.endswith('.json') and f.startswith('res_'):
            with open(os.path.join(res_dir, f), 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    results = data.get('results', [])
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    continue
                
            if not results:
                continue
                
            y_true = []
            y_scores = []
            
            for item in results:
                y_true.append(item.get('label', 0))
                y_scores.append(item.get('predicted_score', 0.5))
                
            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            y_pred = (y_scores >= 0.5).astype(int)
            
            try:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, y_scores)
                aupr = average_precision_score(y_true, y_scores)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
            except Exception as e:
                # Fallback if an experiment entirely failed and predicted the same score
                acc, prec, rec, f1, spec, auc, aupr = 0, 0, 0, 0, 0, 0, 0
                
            # Read original timings/review rate if available
            orig_metrics = data.get('metrics', {})
            review_rate = orig_metrics.get('review_rate', 0.0)
            time_sec = orig_metrics.get('time_seconds', 0.0)
                
            exp_name = f.replace('res_', '').replace('.json', '')
            
            metrics_list.append({
                'exp_name': exp_name,
                'acc': acc,
                'prec': prec,
                'rec': rec,
                'f1': f1,
                'spec': spec,
                'auc': auc,
                'aupr': aupr,
                'time': time_sec,
                'rev_rate': review_rate
            })
            
    metrics_list.sort(key=lambda x: x['exp_name'])
    
    print(f"\n{'Experiment Name':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Spec':<6} | {'AUC':<6} | {'AUPR':<6} | {'Time(s)':<7} | {'RevRate':<7}")
    print("-" * 105)
    for m in metrics_list:
        print(f"{m['exp_name']:<25} | "
              f"{m['acc']:.4f} | "
              f"{m['prec']:.4f} | "
              f"{m['rec']:.4f} | "
              f"{m['f1']:.4f} | "
              f"{m['spec']:.4f} | "
              f"{m['auc']:.4f} | "
              f"{m['aupr']:.4f} | "
              f"{m['time']:<7.1f} | "
              f"{m['rev_rate']:.1%}")

if __name__ == "__main__":
    evaluate_all()
