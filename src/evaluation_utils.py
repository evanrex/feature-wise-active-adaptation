import numpy as np
from models import compute_all_metrics
import itertools
import torch

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
        

def evaluate_test_time_interventions(trainer, model, data_module, args, checkpoint_path):
    
    model.args.test_time_interventions = True
    
    trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)
    necessary_features = model.necessary_features()
    
    for k in range(0, args.num_features-necessary_features):
        model.args.num_additional_features = k
        
        model.log_test_key = 'bestmodel_tti_valid'
        trainer.test(model, dataloaders=data_module.val_dataloader()[0], ckpt_path=checkpoint_path)
        
        model.log_test_key = 'bestmodel_tti_test'
        trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)
 



# def evaluate_test_time_interventions(model, data_module, args, wandb_logger):
#     necessary_features = model.necessary_features(k=0)
    
#     device = next(model.parameters()).device

#     # Identify indices of unnecessary features
#     unnecessary_indices = torch.where(necessary_features == False)[0]
#     num_unnecessary_features = len(unnecessary_indices)

#     feature_permutations = []
#     feature_permutation_descriptions = []
#     for num_nans in range(num_unnecessary_features + 1):
#         for indices in itertools.combinations(unnecessary_indices, num_nans):
#             X_valid_permuted = data_module.X_valid.copy()
#             X_test_permuted = data_module.X_test.copy()
            
#             for idx in indices:
#                 X_valid_permuted[:, idx] = torch.nan
#                 X_test_permuted[:, idx] = torch.nan
            
#             feature_permutations.append(
#                 {
#                     'X_valid': torch.tensor(X_valid_permuted, dtype=torch.float32).to(device), 
#                     'X_test': torch.tensor(X_test_permuted, dtype=torch.float32).to(device), 
#                     'y_valid': data_module.y_valid, 
#                     'y_test': data_module.y_test
#                 }
#             )
#             feature_permutation_descriptions.append(
#                 {
#                     'num_nans':num_nans,
#                     'nan_indices': indices
#                 }
#             )

#     # Evaluate each permutation
#     for i, feature_permutation in enumerate(feature_permutations):
#         y_pred_valid = model.inference(feature_permutation['X_valid']).cpu().detach().numpy()
#         y_pred_valid = np.argmax(y_pred_valid, axis=1)
#         y_pred_test = model.inference(feature_permutation['X_test']).cpu().detach().numpy()
#         y_pred_test = np.argmax(y_pred_test, axis=1)
        
#         valid_metrics = compute_all_metrics(args, feature_permutation['y_valid'], y_pred_valid)
#         test_metrics = compute_all_metrics(args, feature_permutation['y_test'], y_pred_test)

#         # Log the results for this permutation
#         feature_permutation_description = feature_permutation_descriptions[i]
#         nan_indices = feature_permutation_description['nan_indices']
#         num_nans = feature_permutation_description['num_nans']
        
#         wandb_logger.log_metrics({
#             'nan_indices': nan_indices,
#             'num_nans': num_nans,
#             'valid_metrics': valid_metrics,
#             'test_metrics': test_metrics
#         })



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
