from dataset import *
from models import *
import itertools
import torch

def evaluate_test_time_interventions(model, data_module, args, wandb_logger):
    necessary_features = model.necessary_features()
    
    device = next(model.parameters()).device

    # Identify indices of unnecessary features
    unnecessary_indices = torch.where(necessary_features == False)[0]
    num_unnecessary_features = len(unnecessary_indices)

    feature_permutations = []
    feature_permutation_descriptions = []
    for num_nans in range(num_unnecessary_features + 1):
        for indices in itertools.combinations(unnecessary_indices, num_nans):
            X_valid_permuted = data_module.X_valid.copy()
            X_test_permuted = data_module.X_test.copy()
            
            for idx in indices:
                X_valid_permuted[:, idx] = torch.nan
                X_test_permuted[:, idx] = torch.nan
            
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
                    'num_nans':num_nans,
                    'nan_indices': indices
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
