import numpy as np
from models import compute_all_metrics
import itertools
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_all_masks(args, model,data_module, wandb_logger):
    
    device = next(model.parameters()).device
    X_valid = torch.tensor(data_module.X_valid, dtype=torch.float32).to(device)
    X_test = torch.tensor(data_module.X_test, dtype=torch.float32).to(device)
    
    for mask in model.masks:
        model.mask.data = torch.from_numpy(mask).type_as(model.mask.data)
        
        y_pred_valid = model.forward(X_valid, test_time=True)[0]
        y_pred_valid = torch.softmax(y_pred_valid, dim=1).cpu().detach().numpy()
        y_pred_valid = np.argmax(y_pred_valid, axis=1)
        
        y_pred_test = model.forward(X_test, test_time=True)[0]
        y_pred_test = torch.softmax(y_pred_test, dim=1).cpu().detach().numpy()
        y_pred_test = np.argmax(y_pred_test, axis=1)
        
        valid_metrics = compute_all_metrics(args, data_module.y_valid, y_pred_valid)
        test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test)

        # Log the results for this permutation
        
        int_list = model.necessary_features().int().tolist()
        # Convert list of integers to a string of 0s and 1s
        mask_as_string_of_ones_and_zeros = ''.join(str(i) for i in int_list)

      
        wandb_logger.log_metrics({
            'masks/valid_metrics': valid_metrics,
            'masks/test_metrics': test_metrics,
            'masks/mask':mask_as_string_of_ones_and_zeros,
            'masks/mask_probs': model.mask.data
        })
        
def get_labels_lists(outputs):
	all_y_true, all_y_pred = [], []
	for output in outputs:
		all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
		all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

	return all_y_true, all_y_pred

def evaluate(model, dataloader):
    outputs = [model.test_step(batch, batch_idx) for batch_idx, batch in enumerate(dataloader)]
    avg_losses = {loss: np.mean([output['losses'][loss].item() for output in outputs]) for loss in ['total','cross_entropy']}
    
    metrics = {f"{loss}_loss": avg_losses[loss] for loss in avg_losses}
    y_true, y_pred = get_labels_lists(outputs)
    metrics.update({
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
    })
    
    if model.args.num_classes == 2:
        metrics['AUROC_weighted'] = roc_auc_score(y_true, y_pred, average='weighted')
    
    return metrics

def evaluate_test_time_interventions(model, data_module, args, wandb_logger):
    model.args.test_time_interventions_in_progress = True
    num_necessary_features = int(model.necessary_features().float().sum().item())
    end = args.num_features - num_necessary_features
    
    max_steps = 10
    num_steps = min(max_steps, end + 1)
    
    for k in np.linspace(0, end, num_steps, dtype=int):
        model.args.num_additional_features = k
        valid_metrics = evaluate(model, data_module.val_dataloader()[0])
        test_metrics = evaluate(model, data_module.test_dataloader())
        
        wandb_logger.log_metrics({
            'num_additional_features': k,
            'tti_valid_metrics': valid_metrics,
            'tti_test_metrics': test_metrics
        })


def assist_test_time_interventions(model, data_module, args, wandb_logger):
    '''
    Generate data for plot of varying num necessary features for the sigmoidal mask. 
    '''
    
    device = next(model.parameters()).device
    
    feature_permutations = []
    feature_permutation_descriptions = []
    for k in range(args.num_features):
        necessary_features = model.necessary_features(k=k)
    

        # Identify indices of unnecessary features
        unnecessary_indices = torch.where(necessary_features == False)[0].cpu().numpy()
        num_unnecessary_features = len(unnecessary_indices)
        
        X_valid_permuted = data_module.X_valid.copy()
        X_test_permuted = data_module.X_test.copy()
        X_valid_permuted[:, unnecessary_indices] = torch.nan
        X_test_permuted[:, unnecessary_indices] = torch.nan
        
        feature_permutations.append(
            {
                'X_valid': torch.tensor(X_valid_permuted, dtype=torch.float32).to(device), 
                'X_test': torch.tensor(X_test_permuted, dtype=torch.float32).to(device), 
                'y_valid': data_module.y_valid, 
                'y_test': data_module.y_test
            }
        )
        feature_permutation_descriptions.append(
            {
                'num_nans':num_unnecessary_features,
                'nan_indices': unnecessary_indices
            }
        )

    # Evaluate each permutation
    for i, feature_permutation in enumerate(feature_permutations):
        y_pred_valid = model.inference(feature_permutation['X_valid']).cpu().detach().numpy()
        y_pred_valid = np.argmax(y_pred_valid, axis=1)
        y_pred_test = model.inference(feature_permutation['X_test']).cpu().detach().numpy()
        y_pred_test = np.argmax(y_pred_test, axis=1)
        
        valid_metrics = compute_all_metrics(args, feature_permutation['y_valid'], y_pred_valid)
        test_metrics = compute_all_metrics(args, feature_permutation['y_test'], y_pred_test)

        # Log the results for this permutation
        feature_permutation_description = feature_permutation_descriptions[i]
        nan_indices = feature_permutation_description['nan_indices']
        num_nans = feature_permutation_description['num_nans']
        
        wandb_logger.log_metrics({
            'nan_indices': nan_indices,
            'num_nans': num_nans,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics
        })
