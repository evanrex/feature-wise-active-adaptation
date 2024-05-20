import numpy as np
from models import compute_all_metrics
import itertools
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error
from tqdm import tqdm


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


def evaluate_imputation_PBMC(model, data_module, args, wandb_logger, feature_importance=None,logging_key=""):

    imputation_methods = ['mean']
    
    for fraction in tqdm([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 1.0]):
        for missingness_type in ['MCAR', 'MNAR']:
            if missingness_type == 'MNAR' and feature_importance is not None:
                data_module.gen_MNAR_datasets(feature_importance, fraction, replace_val=np.nan)
            elif missingness_type == 'MCAR':
                data_module.gen_MCAR_datasets(fraction, replace_val=np.nan)
            else:
                continue
            
            for imputation_method in imputation_methods:
                datasets = {'valid': (data_module.X_valid_missing, data_module.y_valid, 'validation'), 
                            'test': (data_module.X_test_missing, data_module.y_test, 'test')}
                
                metrics = {'test':{}, 'valid': {}}

                for dataset_name, (dataset, labels, label_text) in datasets.items():
                    X_imputed = dataset
                    missing_mask = np.isnan(dataset)
                    X_imputed[missing_mask] = 0
                    if args.model in ['lasso', 'rf', 'xgboost']:
                        y_pred = model.predict(X_imputed)
                        metrics[dataset_name] = compute_all_metrics(args, labels, y_pred)
                    else:
                        metrics[dataset_name] = evaluate(model, data_module.missing_dataloader(X_imputed, labels))

                    if fraction > 0:
                        mse = mean_squared_error(data_module.X_valid[missing_mask], X_imputed[missing_mask]) if dataset_name == 'valid' else mean_squared_error(data_module.X_test[missing_mask], X_imputed[missing_mask])
                        metrics[dataset_name]['imputation_mse'] = mse

                valid_metrics = metrics['valid']
                test_metrics = metrics['test']
                
                log_metrics = {'fraction_missing_features'+logging_key: fraction}
                if valid_metrics:
                    log_metrics[imputation_method+'_'+missingness_type+'_imputation_valid_metrics'+logging_key] = valid_metrics
                if test_metrics:
                    log_metrics[imputation_method+'_'+missingness_type+'_imputation_test_metrics'+logging_key] = test_metrics
                
                wandb_logger.log_metrics(log_metrics)           


def evaluate_imputation(model, data_module, args, wandb_logger, feature_importance=None,logging_key=""):
    if args.dataset == 'PBMC':
        evaluate_imputation_PBMC(model, data_module, args, wandb_logger, feature_importance=feature_importance,logging_key=logging_key)
        return
    from hyperimpute.plugins.imputers import Imputers
    imputers = Imputers()
    
    mean_imputer = imputers.get('mean', random_state=args.seed_model_init)
    ice_imputer = imputers.get('sklearn_ice', random_state=args.seed_model_init)
    missforest_imputer = imputers.get('sklearn_missforest', random_state=args.seed_model_init)
    
    mean_imputer = mean_imputer.fit(data_module.X_train)
    ice_imputer = ice_imputer.fit(data_module.X_train)
    missforest_imputer = missforest_imputer.fit(data_module.X_train)
    
    imputation_methods = {
        'mean': mean_imputer,
        'ice': ice_imputer,
        'missforest': missforest_imputer
    }
    
    for fraction in tqdm([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]):
        for missingness_type in ['MCAR', 'MNAR']:
            if missingness_type == 'MNAR' and feature_importance is not None:
                data_module.gen_MNAR_datasets(feature_importance, fraction, replace_val=np.nan)
            elif missingness_type == 'MCAR':
                data_module.gen_MCAR_datasets(fraction, replace_val=np.nan)
            else:
                continue
            
            for imputation_method in imputation_methods:
                datasets = {'valid': (data_module.X_valid_missing, data_module.y_valid, 'validation'), 
                            'test': (data_module.X_test_missing, data_module.y_test, 'test')}
                
                metrics = {'test':{}, 'valid': {}}

                for dataset_name, (dataset, labels, label_text) in datasets.items():
                    try:
                        X_imputed = imputation_methods[imputation_method].transform(dataset).to_numpy()
                        if args.model in ['lasso', 'rf', 'xgboost']:
                            y_pred = model.predict(X_imputed)
                            metrics[dataset_name] = compute_all_metrics(args, labels, y_pred)
                        else:
                            metrics[dataset_name] = evaluate(model, data_module.missing_dataloader(X_imputed, labels))

                        if fraction > 0:
                            missing_mask = np.isnan(dataset)
                            mse = mean_squared_error(data_module.X_valid[missing_mask], X_imputed[missing_mask]) if dataset_name == 'valid' else mean_squared_error(data_module.X_test[missing_mask], X_imputed[missing_mask])
                            metrics[dataset_name]['imputation_mse'] = mse

                    except Exception as e:
                        print(f"Imputation failed for {imputation_method} on {label_text} split for fraction {fraction} with Error: {e}")

                valid_metrics = metrics['valid']
                test_metrics = metrics['test']
                
                log_metrics = {'fraction_missing_features'+logging_key: fraction}
                if valid_metrics:
                    log_metrics[imputation_method+'_'+missingness_type+'_imputation_valid_metrics'+logging_key] = valid_metrics
                if test_metrics:
                    log_metrics[imputation_method+'_'+missingness_type+'_imputation_test_metrics'+logging_key] = test_metrics
                
                wandb_logger.log_metrics(log_metrics)           


def evaluate_MCAR_imputation(model, data_module, args, wandb_logger, logging_key=""):
    
    
    for fraction in tqdm([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]):
        removed_features = data_module.gen_MCAR_datasets(fraction)
            
        if args.model in ['lasso', 'rf', 'xgboost']:
            y_pred_valid = model.predict(data_module.X_valid_missing)
            y_pred_test = model.predict(data_module.X_test_missing)

            valid_metrics = compute_all_metrics(args, data_module.y_valid, y_pred_valid)
            test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test)
        else:
            if args.model == "fwal":
                if not args.hierarchical:
                    raise ValueError("Feature selection is only supported for hierarchical models")
                model.update_masks(removed_features)
            valid_metrics = evaluate(model, data_module.missing_val_dataloader())
            test_metrics = evaluate(model, data_module.missing_test_dataloader())
        
        wandb_logger.log_metrics({
            'fraction_missing_features_MCAR'+logging_key: fraction,
            'MCAR_imputation_valid_metrics'+logging_key: valid_metrics,
            'MCAR_imputation_test_metrics'+logging_key: test_metrics
        })
    
    if args.model == "fwal":
        if not args.hierarchical:
            raise ValueError("Feature selection is only supported for hierarchical models")
        model.update_masks(None)
        
    
    
def evaluate_feature_selection(model, feature_importance, data_module, args, wandb_logger, logging_key=""):
    
    for fraction in tqdm([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]):
        removed_features = data_module.gen_MNAR_datasets(feature_importance, fraction)
            
        if args.model in ['lasso', 'rf', 'xgboost']:
            y_pred_valid = model.predict(data_module.X_valid_missing)
            y_pred_test = model.predict(data_module.X_test_missing)

            valid_metrics = compute_all_metrics(args, data_module.y_valid, y_pred_valid)
            test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test)
        else:
            if args.model == "fwal":
                if not args.hierarchical:
                    raise ValueError("Feature selection is only supported for hierarchical models")
                model.update_masks(removed_features)
            valid_metrics = evaluate(model, data_module.missing_val_dataloader())
            test_metrics = evaluate(model, data_module.missing_test_dataloader())
        
        wandb_logger.log_metrics({
            'fraction_missing_features_MNAR'+logging_key: fraction,
            'feature_selection_valid_metrics'+logging_key: valid_metrics,
            'feature_selection_test_metrics'+logging_key: test_metrics
        })
    
    if args.model == "fwal":
        if not args.hierarchical:
            raise ValueError("Feature selection is only supported for hierarchical models")
        model.update_masks(None)
        
    
    
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
