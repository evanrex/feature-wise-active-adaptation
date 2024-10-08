import collections
from dataclasses import dataclass
from pickletools import optimize
from statistics import mode
from pkg_resources import evaluate_marker
import pytorch_lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from lassonet import LassoNetClassifier

import wandb
import json
import pprint
import warnings
import sklearn
import logging

from dataset import *
from models import *
from _config import DATA_DIR
from evaluation_utils import *
import os
import glob

os.environ["WANDB__SERVICE_WAIT"] = "300"


def get_run_name(args):
    if args.model == "dnn":
        run_name = "mlp"
    elif args.model == "dietdnn":
        run_name = "mlp_wpn"
    else:
        run_name = args.model

    if args.sparsity_type == "global":
        run_name += "_SPN_global"
    elif args.sparsity_type == "local":
        run_name += "_SPN_local"

    return run_name


def create_wandb_logger(args):
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        save_dir=args.data_dir,
        group=args.group,
        job_type=args.job_type,
        tags=args.tags,
        notes=args.notes,
        entity=WANDB_ENTITY,
        # reinit=True,
        log_model=args.wandb_log_model,
        settings=wandb.Settings(start_method="thread"),
    )
    trainer = pytorch_lightning.Trainer(logger=wandb_logger)
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(args)  # add configuration file

    return wandb_logger


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, pretrain_epochs, **kwargs):
        super().__init__(**kwargs)
        self.pretrain_epochs = pretrain_epochs

    def on_validation_end(self, trainer, pl_module):
        """
        Only activates once the pre-training is over
        """
        # Check if the current epoch is less than pretrain_epochs
        if trainer.current_epoch < self.pretrain_epochs:
            # Skip the early stopping check
            return
        # Otherwise, proceed with the normal early stopping logic
        super().on_validation_end(trainer, pl_module)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, pretrain_epochs, **kwargs):
        super().__init__(**kwargs)
        self.num_pretrain_epochs = pretrain_epochs

    def on_validation_end(self, trainer, pl_module):
        # Override to skip checkpointing if the current epoch is less than num_pretrain_epochs
        if trainer.current_epoch < self.num_pretrain_epochs:
            return
        # Call the superclass method to continue normal operation
        super().on_validation_end(trainer, pl_module)

    def _save_model(self, trainer, filepath):
        # This method is called to save the model. You might need to handle the case
        # where the monitored metric is missing. This is just a placeholder implementation.
        # The real implementation might need adjustments based on your specific requirements.
        metrics = trainer.callback_metrics
        if (
            self.monitor not in metrics
            and trainer.current_epoch < self.num_pretrain_epochs
        ):
            # Skip saving if the monitored metric is missing during pretraining
            return
        super()._save_model(trainer, filepath)


def run_experiment(args):
    args.suffix_wand_run_name = f"repeat-{args.repeat_id}__test-{args.test_split}"

    #### Load dataset
    print(f"\nInside training function")
    print(f"\nLoading data {args.dataset}...")
    data_module = create_data_module(args)

    print(
        f"Train/Valid/Test splits of sizes {args.train_size}, {args.valid_size}, {args.test_size}"
    )
    print(f"Num of features: {args.num_features}")

    #### Intialize logging
    wandb_logger = create_wandb_logger(args)
    trainer = pytorch_lightning.Trainer(logger=wandb_logger)
    if trainer.global_rank == 0:
        wandb.run.name = (
            f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"
        )

    #### Scikit-learn training
    if args.model in ["lasso", "rf", "lgb", "tabnet", "lassonet", "xgboost"]:
        # scikit-learn expects class_weights to be a dictionary
        class_weights = {}
        for i, val in enumerate(args.class_weights):
            class_weights[i] = val

        class_weights_list = [class_weights[i] for i in range(len(class_weights))]

        if args.model == "lasso":
            model = LogisticRegression(
                penalty="elasticnet",
                C=args.lasso_C,
                l1_ratio=args.lasso_l1_ratio,
                class_weight=class_weights,
                max_iter=10000,
                random_state=args.seed_model_init,
                solver="saga",
                verbose=True,
            )
            model.fit(data_module.X_train, data_module.y_train)
            importance = (
                model.coef_
            )  # This is a 2D array of shape (n_classes, n_features)
            # Log each class's feature importances as a separate entry
            for class_index in range(importance.shape[0]):
                importance_dict = {
                    f"feature_{i}": importance[class_index, i]
                    for i in range(importance.shape[1])
                }
                wandb.log({f"class_{class_index}_importances": importance_dict})

        elif args.model == "rf":
            model = RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                min_samples_leaf=args.rf_min_samples_leaf,
                max_depth=args.rf_max_depth,
                class_weight=class_weights,
                max_features="sqrt",
                random_state=args.seed_model_init,
                verbose=True,
            )
            model.fit(data_module.X_train, data_module.y_train)
            importance = (
                model.feature_importances_
            )  # This is a 1D array of shape (n_features,)
            # Create a dictionary of feature importances
            importance_dict = {
                f"feature_{i}": importance[i] for i in range(len(importance))
            }
            # Log the dictionary of feature importances to wandb
            wandb.log({"feature_importances": importance_dict})

        elif args.model == "lgb":
            params = {
                "max_depth": args.lgb_max_depth,
                "learning_rate": args.lgb_learning_rate,
                "min_data_in_leaf": args.lgb_min_data_in_leaf,
                "class_weight": class_weights,
                "n_estimators": 200,
                "objective": "cross_entropy",
                "num_iterations": 10000,
                "device": "gpu",
                "feature_fraction": "0.3",
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                data_module.X_train,
                data_module.y_train,
                eval_set=[(data_module.X_valid, data_module.y_valid)],
                callbacks=[lgb.early_stopping(stopping_rounds=100)],
            )

        elif args.model == "tabnet":
            model = TabNetClassifier(
                n_d=8,
                n_a=8,  # The TabNet implementation says "Bigger values gives more capacity to the model with the risk of overfitting"
                n_steps=3,
                gamma=1.5,
                n_independent=2,
                n_shared=2,  # default values
                momentum=0.3,
                clip_value=2.0,
                lambda_sparse=args.tabnet_lambda_sparse,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=args.lr),  # the paper sugests 2e-2
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"gamma": 0.95, "step_size": 20},
                seed=args.seed_training,
            )

            class WeightedCrossEntropy(Metric):
                def __init__(self):
                    self._name = "cross_entropy"
                    self._maximize = False

                def __call__(self, y_true, y_score):
                    aux = (
                        F.cross_entropy(
                            input=torch.tensor(y_score, device="cuda"),
                            target=torch.tensor(y_true, device="cuda"),
                            weight=torch.tensor(args.class_weights, device="cuda"),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    return float(aux)

            virtual_batch_size = 5
            if args.dataset == "lung":
                virtual_batch_size = 6  # lung has training of size 141. With a virtual_batch_size of 5, the last batch is of size 1 and we get an error because of BatchNorm

            batch_size = args.train_size
            model.fit(
                data_module.X_train,
                data_module.y_train,
                eval_set=[(data_module.X_valid, data_module.y_valid)],
                eval_metric=[WeightedCrossEntropy],
                loss_fn=torch.nn.CrossEntropyLoss(
                    torch.tensor(args.class_weights, device="cuda")
                ),
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                max_epochs=5000,
                patience=100,
            )

        elif args.model == "lassonet":
            model = LassoNetClassifier(
                lambda_start=args.lassonet_lambda_start,
                gamma=args.lassonet_gamma,
                gamma_skip=args.lassonet_gamma,
                M=args.lassonet_M,
                n_iters=args.lassonet_epochs,
                optim=partial(torch.optim.AdamW, lr=1e-4, betas=[0.9, 0.98]),
                hidden_dims=(100, 100, 10),
                class_weight=class_weights_list,  # use weighted loss
                dropout=0.2,
                batch_size=8,
                backtrack=True,  # if True, ensure the objective is decreasing
                # random_state = 42, # seed for validation set,
                # no need to use because we provide validation set
            )

            model.path(
                data_module.X_train,
                data_module.y_train,
                X_val=data_module.X_valid,
                y_val=data_module.y_valid,
            )

        elif args.model == "xgboost":
            import xgboost as xgb

            if args.num_classes == 2:
                eval_metric = "logloss"
            else:
                eval_metric = "mlogloss"
            model = xgb.XGBClassifier(
                eval_metric=eval_metric,
                use_label_encoder=False,
                random_state=args.seed_model_init,
                verbose=True,
                early_stopping_rounds=int(
                    args.patience_early_stopping * args.val_check_interval
                ),
                device="cuda",
                eta=args.xgb_eta,
                max_depth=args.xgb_max_depth,
            )
            model.fit(
                data_module.X_train,
                data_module.y_train,
                eval_set=[(data_module.X_valid, data_module.y_valid)],
                verbose=True,
            )
            importance = model.feature_importances_
            importance_dict = {
                f"feature_{i}": importance[i] for i in range(len(importance))
            }
            # Log the dictionary of feature importances to wandb
            wandb.log({"feature_importances": importance_dict})

        #### Log metrics
        y_pred_train = model.predict(data_module.X_train)
        y_pred_valid = model.predict(data_module.X_valid)
        y_pred_test = model.predict(data_module.X_test)

        train_metrics = compute_all_metrics(args, data_module.y_train, y_pred_train)
        valid_metrics = compute_all_metrics(args, data_module.y_valid, y_pred_valid)
        test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test)

        for metrics, dataset_name in zip(
            [train_metrics, valid_metrics, test_metrics],
            ["bestmodel_train", "bestmodel_valid", "bestmodel_test"],
        ):
            for metric_name, metric_value in metrics.items():
                wandb.run.summary[f"{dataset_name}/{metric_name}"] = metric_value

        if args.evaluate_imputation:
            if args.model == "lasso":
                feature_importance = np.mean(
                    np.abs(model.coef_), axis=0
                )  # taking mean feature importance
            elif args.model in ["rf", "xgboost"]:
                feature_importance = model.feature_importances_
            evaluate_imputation(
                model,
                data_module,
                args,
                wandb_logger,
                feature_importance=feature_importance,
            )

        if args.evaluate_MCAR_imputation:
            # mean imputation
            evaluate_MCAR_imputation(model, data_module, args, wandb_logger)

        if args.evaluate_feature_selection:
            if args.model == "lasso":
                feature_importance = np.mean(
                    np.abs(model.coef_), axis=0
                )  # taking mean feature importance
            elif args.model in ["rf", "xgboost"]:
                feature_importance = model.feature_importances_
            else:
                raise Exception(
                    f"evaluate_feature_selection not supported for model: {args.model}"
                )
            evaluate_imputation(
                model,
                data_module,
                args,
                wandb_logger,
                feature_importance=feature_importance,
            )

    #### Pytorch lightning training
    else:

        ############################## setting model args ##############################

        #### Set embedding size if it wasn't provided
        if args.wpn_embedding_size == -1:
            args.wpn_embedding_size = args.train_size
        if args.sparsity_gene_embedding_size == -1:
            args.sparsity_gene_embedding_size = args.train_size
        if args.pretrain and (args.num_pretrain_steps != -1):
            # compute the upper rounded number of epochs to training (used for lr scheduler in DKL)
            steps_per_epoch = np.floor(args.train_size / args.batch_size)
            args.max_pretrain_epochs = int(
                np.ceil((args.num_pretrain_steps) / steps_per_epoch)
            )
            print(f"Pre-Training for max_pretrain_epochs = {args.max_pretrain_epochs}")
        if args.max_steps != -1:
            # compute the upper rounded number of epochs to training (used for lr scheduler in DKL)
            steps_per_epoch = np.floor(args.train_size / args.batch_size)
            args.max_epochs = int(np.ceil((args.max_steps) / steps_per_epoch))
            print(f"Training for max_epochs = {args.max_epochs}")
        ############################## done setting model args ##############################

        #### Loading model if run name is provided
        if args.load_trained_model_run_name is not None:

            def get_epoch_model_paths(directory_path):
                epoch_files = []
                for filename in os.listdir(directory_path):
                    if filename.startswith("epoch") and filename.endswith(".ckpt"):
                        full_path = os.path.join(directory_path, filename)
                        epoch_files.append(full_path)
                return epoch_files

            def get_most_recent_file(file_paths):
                most_recent_path = None
                latest_time = 0

                for path in file_paths:
                    modification_time = os.path.getmtime(path)
                    if modification_time > latest_time:
                        latest_time = modification_time
                        most_recent_path = path

                return most_recent_path

            ckpt_path = os.path.join(
                args.data_dir, "fwal", args.load_trained_model_run_name, "checkpoints"
            )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Directory not found at {ckpt_path}")
            ckpts = get_epoch_model_paths(ckpt_path)
            if len(ckpts) == 0:
                raise FileNotFoundError(
                    f"No .ckpt files starting eith 'epoch' found at {ckpt_path}"
                )
            checkpoint_path = get_most_recent_file(ckpts)
            model = create_model(args, data_module, checkpoint=checkpoint_path)

        #### Training model
        else:
            #### Create model
            model = create_model(args, data_module)

            if args.pretrain:
                pretrainer, pre_checkpoint_callback = pre_train_model(
                    args, model, data_module, wandb_logger
                )
                trainer, checkpoint_callback = train_model(
                    args,
                    model,
                    data_module,
                    wandb_logger,
                    pre_trained_ckpt=pre_checkpoint_callback.best_model_path,
                )
            else:
                trainer, checkpoint_callback = train_model(
                    args, model, data_module, wandb_logger
                )

            if args.train_on_full_data:
                checkpoint_path = checkpoint_callback.last_model_path
            else:
                checkpoint_path = checkpoint_callback.best_model_path

                print(f"\n\nBest model saved on path {checkpoint_path}\n\n")
                wandb.log(
                    {
                        "bestmodel/step": checkpoint_path.split("step=")[1].split(
                            ".ckpt"
                        )[0]
                    }
                )

            #### Compute metrics for the best model
            model.log_test_key = "bestmodel_train"
            trainer.test(
                model,
                dataloaders=data_module.train_dataloader(),
                ckpt_path=checkpoint_path,
            )

            model.log_test_key = "bestmodel_valid"
            trainer.test(
                model,
                dataloaders=data_module.val_dataloader()[0],
                ckpt_path=checkpoint_path,
            )

            model.log_test_key = "bestmodel_test"
            trainer.test(
                model,
                dataloaders=data_module.test_dataloader(),
                ckpt_path=checkpoint_path,
            )

            if args.model == "fwal":
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["state_dict"])
                # Convert boolean tensor to tensor of 0s and 1s
                int_list = model.necessary_features().int().tolist()
                # Convert list of integers to a string of 0s and 1s
                mask_as_string_of_ones_and_zeros = "".join(str(i) for i in int_list)

                if args.hierarchical and not args.as_MLP_baseline:
                    wandb.log({"best_mask_0_parameters": model.mask_0.data})
                    wandb.log({"best_mask_1_parameters": model.mask_1.data})
                    wandb.log({"best_mask": mask_as_string_of_ones_and_zeros})
                    wandb_logger.log_metrics(
                        {
                            "best_mask": mask_as_string_of_ones_and_zeros,
                            "best_mask_0_histogram": model.mask_0.data,
                            "best_mask_1_histogram": model.mask_1.data,
                            "best_mask_0_parameters": model.mask_0.data.tolist(),
                            "best_mask_1_parameters": model.mask_1.data.tolist(),
                        }
                    )
                else:
                    wandb.log({"best_mask": mask_as_string_of_ones_and_zeros})
                    wandb.log({"best_mask_parameters": model.mask.data})
                    wandb_logger.log_metrics(
                        {
                            "best_mask": mask_as_string_of_ones_and_zeros,
                            "best_mask_parameters": model.mask.data,
                        }
                    )

            elif args.model in ["cae", "superivsed_cae"]:
                model.eval()
                int_list = model.necessary_features().int().tolist()
                mask_as_string_of_ones_and_zeros = "".join(str(i) for i in int_list)
                wandb.log({"best_mask": mask_as_string_of_ones_and_zeros})
                wandb.log({"best_mask_parameters": int_list})
                wandb_logger.log_metrics(
                    {
                        "best_mask": mask_as_string_of_ones_and_zeros,
                        "best_mask_parameters": int_list,
                    }
                )
            elif args.model == "SEFS":
                model.eval()
                int_list = model.necessary_features().int().tolist()
                mask_as_string_of_ones_and_zeros = "".join(str(i) for i in int_list)
                wandb.log({"best_mask": mask_as_string_of_ones_and_zeros})
                wandb.log({"best_mask_parameters": int_list})
                wandb_logger.log_metrics(
                    {
                        "best_mask": mask_as_string_of_ones_and_zeros,
                        "best_mask_parameters": model.mask_module.pi_logit.data,
                    }
                )

        if args.evaluate_imputation:
            if (args.model == "fwal" and args.as_MLP_baseline) or (
                args.model in ["cae", "supervised_cae"]
            ):
                feature_importance = None
            elif (args.model == "fwal" and args.hierarchical) or (args.model == "SEFS"):
                feature_importance = model.feature_importance()
            else:
                raise ValueError(
                    f"Feature importance is only supported for hierarchical F-Act, SEFS, cae, supervised_cae & MLP for Pytorch models. Not supported for the pytorch model {args.model}"
                )
            evaluate_imputation(
                model,
                data_module,
                args,
                wandb_logger,
                feature_importance=feature_importance,
            )
        if args.evaluate_MCAR_imputation:
            evaluate_MCAR_imputation(model, data_module, args, wandb_logger)
        if args.evaluate_feature_selection:

            if (args.model != "fwal" and not args.hierarchical) and (
                args.model != "SEFS"
            ):
                raise ValueError(
                    "Feature selection is only supported for hierarchical F-Act and SEFS"
                )
            feature_importance = model.feature_importance()
            evaluate_feature_selection(
                model, feature_importance, data_module, args, wandb_logger
            )
        if args.test_time_interventions == "evaluate_test_time_interventions":
            # We have just loaded the best model weights for fwal in the prev if statement
            evaluate_test_time_interventions(model, data_module, args, wandb_logger)
        elif args.test_time_interventions == "assist_test_time_interventions":
            assist_test_time_interventions(model, data_module, args, wandb_logger)

    wandb.finish()

    print("\nExiting from train function..")


def pre_train_model(args, model, data_module, wandb_logger=None):
    """
    Return
    - Pytorch Lightening Trainer
    - checkpoint callback
    """

    ##### Train
    if args.saved_checkpoint_name:
        wandb_artifact_path = f"andreimargeloiu/low-data/{args.saved_checkpoint_name}"
        print(f"\nDownloading artifact: {wandb_artifact_path}...")

        artifact = wandb.use_artifact(wandb_artifact_path, type="model")
        artifact_dir = artifact.download()
        model_checkpoint = torch.load(os.path.join(artifact_dir, "model.ckpt"))
        weights = model_checkpoint["state_dict"]
        print("Artifact downloaded")

        if args.load_model_weights:
            print(f"\nLoading pretrained weights into model...")
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            print(f"Missing keys: \n")
            print(missing_keys)

            print(f"Unexpected keys: \n")
            print(unexpected_keys)

    mode_metric = (
        "max" if args.pretrain_metric_model_selection == "balanced_accuracy" else "min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=f"pre_valid/{args.pretrain_metric_model_selection}",
        mode=mode_metric,
        save_last=True,
        verbose=True,
    )
    callbacks = [checkpoint_callback, RichProgressBar()]

    if args.pretrain_patience_early_stopping and args.train_on_full_data == False:
        callbacks.append(
            EarlyStopping(
                monitor=f"pre_valid/{args.pretrain_metric_model_selection}",
                mode=mode_metric,
                patience=args.pretrain_patience_early_stopping,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    pl.seed_everything(args.seed_training, workers=True)
    trainer = pl.Trainer(
        # Training
        max_steps=args.num_pretrain_steps,
        gradient_clip_val=2.5,
        # logging
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        # miscellaneous
        accelerator="auto",
        devices="auto",
        detect_anomaly=(not args.hpc_run),
        overfit_batches=args.overfit_batches,
        deterministic=args.deterministic,
    )
    # train
    trainer.fit(model, data_module)
    args.pretrain = (
        False  # update flag so future training is normal and not pretraining
    )
    model.finish_pretraining()

    return trainer, checkpoint_callback


def train_model(args, model, data_module, wandb_logger=None, pre_trained_ckpt=None):
    """
    Return
    - Pytorch Lightening Trainer
    - checkpoint callback
    """

    ##### Train
    if args.saved_checkpoint_name:
        wandb_artifact_path = f"andreimargeloiu/low-data/{args.saved_checkpoint_name}"
        print(f"\nDownloading artifact: {wandb_artifact_path}...")

        artifact = wandb.use_artifact(wandb_artifact_path, type="model")
        artifact_dir = artifact.download()
        model_checkpoint = torch.load(os.path.join(artifact_dir, "model.ckpt"))
        weights = model_checkpoint["state_dict"]
        print("Artifact downloaded")

        if args.load_model_weights:
            print(f"\nLoading pretrained weights into model...")
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            print(f"Missing keys: \n")
            print(missing_keys)

            print(f"Unexpected keys: \n")
            print(unexpected_keys)

    mode_metric = "max" if args.metric_model_selection == "balanced_accuracy" else "min"
    checkpoint_callback = ModelCheckpoint(
        monitor=f"valid/{args.metric_model_selection}",
        mode=mode_metric,
        save_last=True,
        verbose=True,
    )
    callbacks = [checkpoint_callback, RichProgressBar()]

    if args.patience_early_stopping and args.train_on_full_data == False:
        callbacks.append(
            EarlyStopping(
                monitor=f"valid/{args.metric_model_selection}",
                mode=mode_metric,
                patience=args.patience_early_stopping,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    pl.seed_everything(args.seed_training, workers=True)
    trainer = pl.Trainer(
        # Training
        max_steps=args.max_steps,
        gradient_clip_val=2.5,
        # logging
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        # miscellaneous
        accelerator="auto",
        devices="auto",
        detect_anomaly=(not args.hpc_run),
        overfit_batches=args.overfit_batches,
        deterministic=args.deterministic,
    )
    # train
    trainer.fit(model, data_module)

    return trainer, checkpoint_callback


def load_feature_importance(run):
    # Load the feature importance
    history = run.history()

    max_index = max(
        int(col.split("_")[-1])
        for col in history.columns
        if "feature_importances.feature" in col
    )
    feature_importances = np.full(max_index + 1, np.nan)

    # Iterate over each column and place the non-NaN value in the corresponding position in the array
    for col in history.columns:
        if "feature_importances.feature" in col:
            # Extract the feature index from the column name
            feature_index = int(col.split("_")[-1])
            # Find the non-NaN value in this column
            non_nan_value = (
                history[col].dropna().values[0]
            )  # assuming there is exactly one non-NaN value per feature
            feature_importances[feature_index] = non_nan_value
    return feature_importances


def load_feature_importance_lasso(run):

    history = run.history()

    # Determine the maximum indices for classes and features
    max_class_index = max(
        int(col.split("_")[1]) for col in history.columns if "class_" in col
    )
    max_feature_index = max(
        int(col.split("_")[-1]) for col in history.columns if "feature" in col
    )

    # Initialize a 2D array of NaNs
    coef = np.full((max_class_index + 1, max_feature_index + 1), np.nan)

    # Iterate over each column and assign values to the coef array
    for col in history.columns:
        if "class_" in col and "feature" in col:
            parts = col.split("_")
            class_index = int(parts[1])
            feature_index = int(parts[-1])
            # Find the non-NaN value in this column
            non_nan_value = (
                history[col].dropna().values[0]
            )  # assuming there is exactly one non-NaN value per feature and class
            coef[class_index, feature_index] = non_nan_value

    # Now 'coef' is your reconstructed 2D array similar to 'model.coef_' in sklearn Lasso for multi-class problems
    feature_importance = np.mean(np.abs(coef), axis=0)
    return feature_importance


def validate_args(args):
    """
    Validate the arguments passed to the script against the run config.
    """

    api = wandb.Api()
    run = api.run(f"{'evangeorgerex'}/{'fwal'}/{args.load_trained_model_run_name}")
    # Load the configuration
    config = run.config

    args.seed_model_init = config.get("seed_model_init", args.seed_model_init)
    args.sparsity_regularizer_hyperparam = config.get(
        "sparsity_regularizer_hyperparam", args.sparsity_regularizer_hyperparam
    )
    args.pretrain = config.get("pretrain", args.pretrain)
    args.num_pretrain_steps = config.get("num_pretrain_steps", args.num_pretrain_steps)
    args.lr = config.get("lr", args.lr)
    args.as_MLP_baseline = config.get("as_MLP_baseline", args.as_MLP_baseline)

    args.dataset = config.get("dataset", args.dataset)

    args.hierarchical = config.get("hierarchical", args.hierarchical)
    args.num_hidden = config.get("num_hidden", args.num_hidden)

    if args.model == "fwal":
        pass
    elif args.model == "cae":
        args.CAE_neurons_ratio = config.get("CAE_neurons_ratio", args.CAE_neurons_ratio)
        args.num_CAE_neurons = config.get("num_CAE_neurons", args.num_CAE_neurons)
    elif args.model == "supervised_cae":
        args.CAE_neurons_ratio = config.get("CAE_neurons_ratio", args.CAE_neurons_ratio)
        args.num_CAE_neurons = config.get("num_CAE_neurons", args.num_CAE_neurons)
    elif args.model == "SEFS":
        pass
    elif args.model == "xgboost":
        args.xgb_eta = config.get("xgb_eta", args.xgb_eta)
        args.xgb_max_depth = config.get("xgb_max_depth", args.xgb_max_depth)
    elif args.model == "rf":
        args.rf_max_depth = config.get("rf_max_depth", args.rf_max_depth)
    elif args.model == "lasso":
        args.lasso_C = config.get("lasso_C", args.lasso_C)
        args.lasso_l1_ratio = config.get("lasso_l1_ratio", args.lasso_l1_ratio)

    if args.retrain_feature_selection:
        if args.model == "lasso":
            args.feature_importance = load_feature_importance_lasso(run)
        elif args.model in ["rf", "xgboost"]:
            args.feature_importance = load_feature_importance(run)
        else:
            raise ValueError(
                f"Feature importance is not supported for model {args.model}"
            )


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    """
	Available datasets
	- cll
	- smk
	- toxicity
	- lung
	- metabric-dr__200
	- metabric-pam50__200
	- tcga-2ysurvival__200
	- tcga-tumor-grade__200
	- prostate
	"""

    ####### Dataset
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--dataset_size", type=int, help="100, 200, 330, 400, 800, 1600"
    )
    parser.add_argument(
        "--dataset_feature_set",
        type=str,
        choices=["hallmark", "8000", "16000"],
        default="hallmark",
        help="Note: implemented for Metabric only \
							hallmark = 4160 common genes \
							8000 = the 4160 common genes + 3840 random genes \
							16000 = the 8000 genes above + 8000 random genes",
    )
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)

    ####### Model
    parser.add_argument(
        "--model",
        type=str,
        choices=["lasso", "rf", "cae", "fwal", "xgboost", "supervised_cae", "SEFS"],
        default="fwal",
    )
    parser.add_argument(
        "--num_CAE_neurons", type=int, help="number of features to select for CAE"
    )
    parser.add_argument(
        "--CAE_neurons_ratio",
        type=float,
        default=1.0,
        help="fraction of features to select with CAE",
    )
    parser.add_argument(
        "--layers_for_hidden_representation",
        type=int,
        default=2,
        help="number of layers after which to output the hidden representation used as input to the decoder \
								(e.g., if the layers are [100, 100, 10] and layers_for_hidden_representation=2, \
								then the hidden representation will be the representation after the two layers [100, 100])",
    )
    parser.add_argument(
        "--as_MLP_baseline",
        action="store_true",
        dest="as_MLP_baseline",
        help="Set to true with --model=fwal if want to train FWAL model as a plain MLP ",
    )

    parser.add_argument(
        "--R_num_hidden",
        type=int,
        default=4,
        help="Number of hidden layers in the Reconstruction Module",
    )
    parser.add_argument(
        "--R_hidden_dim",
        type=int,
        default=50,
        help="Dimension of each hidden layer in the Reconstruction Module",
    )
    parser.add_argument(
        "--P_num_hidden",
        type=int,
        default=4,
        help="Number of hidden layers in the Prediction Module",
    )
    parser.add_argument(
        "--P_hidden_dim",
        type=int,
        default=50,
        help="Dimension of each hidden layer in the Prediction Module",
    )
    # General arguments to override specific module configurations
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=None,
        help="General number of hidden layers to override module-specific settings",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="General dimension of hidden layers to override module-specific settings",
    )
    parser.add_argument(
        "--legacy_architecture",
        action="store_true",
        dest="legacy_architecture",
        help="Set to true to train model without dropout layer (enables compatibility with old architecture). Leave num_hidden and hidden_dim at default for it to work.",
    )

    parser.add_argument(
        "--batchnorm",
        type=int,
        default=1,
        help="if 1, then add batchnorm layers in the main network. If 0, then dont add batchnorm layers",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="dropout rate for the main network",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="The factor multiplied to the reconstruction error. \
								If >0, then create a decoder with a reconstruction loss. \
								If ==0, then dont create a decoder.",
    )
    parser.add_argument(
        "--saved_checkpoint_name",
        type=str,
        help="name of the wandb artifact name (e.g., model-1dmvja9n:v0)",
    )
    parser.add_argument(
        "--load_model_weights",
        action="store_true",
        dest="load_model_weights",
        help="True if loading model weights",
    )
    parser.set_defaults(load_model_weights=False)
    parser.add_argument(
        "--load_trained_model_run_name",
        type=str,
        help="name of the wandb run name (e.g., wi1vu9hz)",
    )

    ####### Scikit-learn parameters
    parser.add_argument(
        "--lasso_C", type=float, default=1e3, help="lasso regularization parameter"
    )
    parser.add_argument(
        "--lasso_l1_ratio", type=float, default=1.0, help="lasso l1 ratio parameter"
    )

    parser.add_argument(
        "--rf_n_estimators",
        type=int,
        default=500,
        help="number of trees in the random forest",
    )
    parser.add_argument(
        "--rf_max_depth", type=int, default=5, help="maximum depth of the tree"
    )
    parser.add_argument(
        "--rf_min_samples_leaf",
        type=int,
        default=3,
        help="minimum number of samples in a leaf",
    )

    parser.add_argument(
        "--xgb_max_depth", type=int, default=6, help="maximum depth of the tree"
    )
    parser.add_argument(
        "--xgb_eta", type=float, default=0.3, help="maximum depth of the tree"
    )

    parser.add_argument("--lgb_learning_rate", type=float, default=0.1)
    parser.add_argument("--lgb_max_depth", type=int, default=1)
    parser.add_argument("--lgb_min_data_in_leaf", type=int, default=2)

    parser.add_argument(
        "--tabnet_lambda_sparse",
        type=float,
        default=1e-3,
        help="higher coefficient the sparser the feature selection",
    )

    parser.add_argument(
        "--lassonet_lambda_start",
        default="auto",
        help="higher coefficient the sparser the feature selection",
    )
    parser.add_argument(
        "--lassonet_gamma",
        type=float,
        default=0,
        help="higher coefficient the sparser the feature selection",
    )
    parser.add_argument("--lassonet_epochs", type=int, default=100)
    parser.add_argument("--lassonet_M", type=float, default=10)

    ####### Sparsity
    parser.add_argument(
        "--sparsity_type",
        type=str,
        default="global",
        choices=["global", "local"],
        help="Use global or local sparsity",
    )
    parser.add_argument(
        "--sparsity_method",
        type=str,
        default="sparsity_network",
        choices=["learnable_vector", "sparsity_network"],
        help="The method to induce sparsity",
    )
    parser.add_argument(
        "--mixing_layer_size",
        type=int,
        help="size of the mixing layer in the sparsity network",
    )
    parser.add_argument(
        "--mixing_layer_dropout", type=float, help="dropout rate for the mixing layer"
    )

    parser.add_argument(
        "--sparsity_gene_embedding_type",
        type=str,
        default="nmf",
        choices=["all_patients", "nmf"],
        help="It`s applied over data preprocessed using `embedding_preprocessing`",
    )
    parser.add_argument("--sparsity_gene_embedding_size", type=int, default=50)
    parser.add_argument(
        "--sparsity_regularizer", type=str, default="L1", choices=["L1", "hoyer"]
    )
    parser.add_argument(
        "--sparsity_regularizer_hyperparam",
        type=float,
        default=1.0,
        help="The weight of the sparsity regularizer (used to compute total_loss)",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="gumbel_softmax",
        choices=["sigmoid", "gumbel_softmax"],
        help="Determines type of mask. If sigmoid then real value between 0 and 1. If gumbel_softmax then discrete values of 0 or 1 sampled from the Gumbel-Softmax distribution",
    )
    parser.add_argument(
        "--normalize_sparsity",
        action="store_true",
        dest="normalize_sparsity",
        default=True,
        help="If true, divide sparsity loss by number of features. Defaults to true.",
    )
    parser.add_argument(
        "--normalize_reconstruction",
        type=str,
        default="num_non_masked_features",
        choices=["None", "num_features", "num_non_masked_features"],
        help='Normalization method for reconstruction loss. Defaults to None. If "None", then no normalization is performed. If num_features, then divide by the number of features. If num_non_masked_features, then divide by the number of non-masked features.',
    )
    # parser.add_argument('--only_reconstruct_masked', action='store_true', dest='only_reconstruct_masked', default=True, help='If true, only reconstruct features that were masked. Reconstructed features that were not masked are not used. Defaults to true.')
    parser.add_argument(
        "--no_only_reconstruct_masked",
        dest="only_reconstruct_masked",
        action="store_false",
        help="If flagged, negates the only_reconstruct_masked flag which says to only reconstruct features that were masked. Reconstructed features that were not masked are not used.",
    )
    parser.set_defaults(only_reconstruct_masked=True)

    # Hierarchical
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        dest="hierarchical",
        help="If true, then use hierarchical sparsity",
    )
    parser.add_argument(
        "--sparsity_regularizer_hyperparam_0",
        type=float,
        default=1.0,
        help="The weight of the sparsity regularizer for the first layer",
    )
    parser.add_argument(
        "--share_mask",
        action="store_true",
        dest="share_mask",
        help="If true, then share the hierarchical mask",
    )
    parser.add_argument(
        "--sigmoid_loss",
        action="store_true",
        dest="sigmoid_loss",
        help="If true, then will minimize the sigmoid activation of the shared mask instead of the soft gumbel activation",
    )
    parser.add_argument(
        "--tti_loss_hyperparam",
        type=float,
        default=1.1,
        help="The weight of the tti component of the cross entropy loss",
    )
    parser.add_argument(
        "--selection_threshold",
        type=float,
        default=0.0,
        help="The selection threshold for the sigmoidal feature selection mask. (layer 0 mask). Float between 0 and 1. Set to 0 for no thresholding.",
    )

    ####### DKL
    parser.add_argument(
        "--grid_bound",
        type=float,
        default=5.0,
        help="The grid bound on the inducing points for the GP.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=64,
        help="Dimension of the grid of inducing points",
    )

    ####### Weight predictor network
    parser.add_argument(
        "--wpn_embedding_type",
        type=str,
        default="histogram",
        choices=["histogram", "all_patients", "nmf", "svd"],
        help="histogram = histogram x means (like FsNet)\
								all_patients = randomly pick patients and use their gene expressions as the embedding\
								It`s applied over data preprocessed using `embedding_preprocessing`",
    )
    parser.add_argument(
        "--wpn_embedding_size", type=int, default=50, help="Size of the gene embedding"
    )
    parser.add_argument(
        "--residual_embedding",
        type=str,
        default=None,
        choices=["resnet"],
        help="Implement residual embeddings as e^* = e_{static} + f(e). This hyperparameter defines the type of function f",
    )

    parser.add_argument(
        "--diet_network_dims",
        type=int,
        nargs="+",
        default=[100, 100, 100, 100],
        help="None if you don't want a VirtualLayer. If you want a virtual layer, \
								then provide a list of integers for the sized of the tiny network.",
    )
    parser.add_argument(
        "--nonlinearity_weight_predictor",
        type=str,
        choices=["tanh", "leakyrelu"],
        default="leakyrelu",
    )
    parser.add_argument(
        "--softmax_diet_network",
        type=int,
        default=0,
        dest="softmax_diet_network",
        help="If True, then perform softmax on the output of the tiny network.",
    )

    ####### Training
    parser.add_argument(
        "--use_best_hyperparams",
        action="store_true",
        dest="use_best_hyperparams",
        help="True if you don't want to use the best hyperparams for a custom dataset",
    )
    parser.set_defaults(use_best_hyperparams=False)

    parser.add_argument(
        "--concrete_anneal_iterations",
        type=int,
        default=1000,
        help="number of iterations for annealing the Concrete radnom variables (in CAE and FsNet)",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Specify the max number of steps to train.",
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--patient_preprocessing",
        type=str,
        default="standard",
        choices=["raw", "standard", "minmax"],
        help="Preprocessing applied on each COLUMN of the N x D matrix, where a row contains all gene expressions of a patient.",
    )
    parser.add_argument(
        "--embedding_preprocessing",
        type=str,
        default="minmax",
        choices=["raw", "standard", "minmax"],
        help="Preprocessing applied on each ROW of the D x N matrix, where a row contains all patient expressions for one gene.",
    )

    ####### Self-supervised pre-training
    parser.add_argument(
        "--pretrain",
        action="store_true",
        dest="pretrain",
        help="Boolean, whether to perform the SEFS self-supervised pretraining",
    )
    parser.add_argument(
        "--num_pretrain_steps",
        type=int,
        default=0,
        help="number of steps to pre-train the model with the SEFS self-supervision task",
    )
    parser.add_argument(
        "--pre_alpha",
        type=float,
        default=1,
        help="alpha hyperparameter of the SEFS self-supervision task. Determines the weights of the mask prediction loss in the self-supervised task",
    )
    parser.add_argument(
        "--pre_pi",
        type=float,
        default=0.5,
        help="alpha hyperparameter of the SEFS self-supervision task. Determines the weights of the mask prediction loss in the self-supervised task",
    )

    ####### Training on the entire train + validation data
    parser.add_argument(
        "--train_on_full_data",
        action="store_true",
        dest="train_on_full_data",
        help="Train on the full data (train + validation), leaving only `--test_split` for testing.",
    )
    parser.set_defaults(train_on_full_data=False)
    parser.add_argument(
        "--path_steps_on_full_data",
        type=str,
        default=None,
        help="Path to the file which holds the number of steps to train.",
    )

    ####### Validation
    parser.add_argument(
        "--metric_model_selection",
        type=str,
        default="total_loss",
        choices=["cross_entropy_loss", "total_loss", "balanced_accuracy"],
    )
    parser.add_argument(
        "--pretrain_metric_model_selection",
        type=str,
        default="pre_total_loss",
        choices=["pre_cross_entropy_loss", "pre_total_loss", "pre_reconstruction_loss"],
    )
    parser.add_argument(
        "--patience_early_stopping",
        type=int,
        default=50,
        help="Set number of checks (set by *val_check_interval*) to do early stopping.\
								It will train for at least   args.val_check_interval * args.patience_early_stopping epochs",
    )
    parser.add_argument(
        "--pretrain_patience_early_stopping",
        type=int,
        default=20,
        help="Set number of checks (set by *val_check_interval*) to do early stopping.\
								It will train for at least   args.val_check_interval * args.pretrain_patience_early_stopping epochs",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.1,
        help="See https://lightning.ai/docs/pytorch/stable/common/trainer.html#val-check-interval",
    )

    # type of data augmentation
    parser.add_argument(
        "--valid_aug_dropout_p",
        type=float,
        nargs="+",
        help="List of dropout data augmentation for the validation data loader.\
								A new validation dataloader is created for each value.\
								E.g., (1, 10) creates a dataloader with valid_aug_dropout_p=1, valid_aug_dropout_p=10\
								in addition to the standard validation",
    )
    parser.add_argument(
        "--valid_aug_times",
        type=int,
        nargs="+",
        help="Number time to perform data augmentation on the validation sample.",
    )
    parser.add_argument(
        "--restrict_features", action="store_true", dest="restrict_features"
    )
    parser.add_argument(
        "--chosen_features_list",
        dest="chosen_features_list",
        type=str,
        required=False,
        help='The list is a comma-separated string. Required if --restrict_features. e.g.: "x1,x3,x4"',
    )

    ####### Testing
    parser.add_argument(
        "--testing_type",
        type=str,
        default="cross-validation",
        choices=["cross-validation", "fixed"],
        help="`cross-validation` performs testing on the testing splits \
								`fixed` performs testing on an external testing set supplied in a dedicated file",
    )

    ####### Cross-validation
    parser.add_argument(
        "--repeat_id",
        type=int,
        default=0,
        help="each repeat_id gives a different random seed for shuffling the dataset",
    )
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV splits")
    parser.add_argument(
        "--test_split",
        type=int,
        default=0,
        help="Index of the test split. It should be smaller than `cv_folds`",
    )
    parser.add_argument(
        "--valid_percentage",
        type=float,
        default=0.25,
        help="Percentage of training data used for validation",
    )

    ####### Evaluation by taking random samples (with user-defined train/valid/test sizes) from the dataset
    parser.add_argument(
        "--evaluate_with_sampled_datasets",
        action="store_true",
        dest="evaluate_with_sampled_datasets",
    )
    parser.set_defaults(evaluate_with_sampled_datasets=False)
    parser.add_argument("--custom_train_size", type=int, default=None)
    parser.add_argument("--custom_valid_size", type=int, default=None)
    parser.add_argument("--custom_test_size", type=int, default=None)

    ####### Test Time interventions
    parser.add_argument(
        "--num_necessary_features",
        type=int,
        default=None,
        help="Number of necessary features to select for test-time interventions. Used when model mask_type is sigmoid.",
    )

    ####### Custom evaluation
    parser.add_argument(
        "--only_test_time_intervention_eval",
        action="store_true",
        default=False,
        help="Set this flag to enable only test time interventions.",
    )
    parser.add_argument(
        "--test_time_interventions",
        type=str,
        choices=["evaluate_test_time_interventions", "assist_test_time_interventions"],
        default=None,
        help="choose one of [evaluate_test_time_interventions]. Remember to choose a run with --trained_FWAL_model_run_name",
    )
    parser.add_argument(
        "--trained_FWAL_model_run_name",
        type=str,
        default=None,
        help="Run id, for example plby9cg4",
    )
    parser.add_argument(
        "--evaluate_all_masks",
        action="store_true",
        default=False,
        help="Set this flag to enable all mask evaluations.",
    )
    parser.add_argument(
        "--evaluate_feature_selection",
        action="store_true",
        default=False,
        help="Set this flag to enable feature selection evaluation.",
    )
    parser.add_argument(
        "--evaluate_imputation",
        action="store_true",
        default=False,
        help="Set this flag to enable feature selection evaluation.",
    )
    parser.add_argument(
        "--evaluate_MCAR_imputation",
        action="store_true",
        default=False,
        help="Set this flag to enable feature selection evaluation.",
    )
    parser.add_argument(
        "--retrain_feature_selection",
        action="store_true",
        default=False,
        help="Set this flag to enable feature selection evaluation via retraining for the SKLearn models (RF, Lasso, XGB) ",
    )

    ####### Optimization
    parser.add_argument(
        "--optimizer", type=str, choices=["adam", "adamw"], default="adamw"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["plateau", "cosine_warm_restart", "linear", "lambda"],
        default=None,
    )
    parser.add_argument("--cosine_warm_restart_eta_min", type=float, default=1e-6)
    parser.add_argument("--cosine_warm_restart_t_0", type=int, default=35)
    parser.add_argument("--cosine_warm_restart_t_mult", type=float, default=1)

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--class_weight",
        type=str,
        choices=["standard", "balanced"],
        default="balanced",
        help="If `standard`, all classes use a weight of 1.\
								If `balanced`, classes are weighted inverse proportionally to their size (see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)",
    )

    parser.add_argument("--debugging", action="store_true", dest="debugging")
    parser.set_defaults(debugging=False)
    parser.add_argument("--deterministic", action="store_true", dest="deterministic")
    parser.set_defaults(deterministic=False)

    ####### Others
    parser.add_argument(
        "--overfit_batches",
        type=float,
        default=0,
        help="0 --> normal training. <1 --> overfit on % of the training data. >1 overfit on this many batches",
    )

    # SEEDS
    parser.add_argument(
        "--seed_model_init",
        type=int,
        default=42,
        help="Seed for initializing the model (to have the same weights)",
    )
    parser.add_argument(
        "--seed_model_mask",
        type=int,
        default=None,
        help="Seed for initializing the model mask (to have the same weights)",
    )
    parser.add_argument(
        "--mask_init_value",
        type=int,
        default=None,
        help="Value for deterministic initialisation of the model mask. default=None for random initialisation.",
    )
    parser.add_argument(
        "--seed_training",
        type=int,
        default=42,
        help="Seed for training (e.g., batch ordering)",
    )
    parser.add_argument(
        "--mask_init_p_array",
        type=str,
        default=None,
        help='Value for deterministic initialisation of the model mask. default=None for random initialisation. expects string list of probabilities: e.g.: "0.1,0.99,0,1" ',
    )

    parser.add_argument(
        "--seed_kfold",
        type=int,
        help="Seed used for doing the kfold in train/test split",
    )
    parser.add_argument(
        "--seed_validation",
        type=int,
        help="Seed used for selecting the validation split.",
    )

    # Dataset loading
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of workers for loading dataset",
    )
    parser.add_argument(
        "--no_pin_memory",
        dest="pin_memory",
        action="store_false",
        help="dont pin memory for data loaders",
    )
    parser.set_defaults(pin_memory=True)
    parser.add_argument(
        "--no_persistent_workers",
        dest="persistent_workers",
        action="store_false",
        help="Set to not use persistent workers",
    )
    parser.set_defaults(persistent_workers=True)

    # Experiment set up
    parser.add_argument(
        "--hpc_run",
        action="store_true",
        dest="hpc_run",
        help="True for when running on HPC",
    )

    ####### Wandb logging
    parser.add_argument("--group", type=str, help="Group runs in wand")
    parser.add_argument("--job_type", type=str, help="Job type for wand")
    parser.add_argument("--notes", type=str, help="Notes for wandb logging.")
    parser.add_argument(
        "--tags", nargs="+", type=str, default=[], help="Tags for wandb"
    )
    parser.add_argument(
        "--suffix_wand_run_name",
        type=str,
        default="",
        help="Suffix for run name in wand",
    )
    parser.add_argument(
        "--wandb_log_model",
        action="store_true",
        dest="wandb_log_model",
        help="True for storing the model checkpoints in wandb",
    )
    parser.set_defaults(wandb_log_model=True)
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        dest="disable_wandb",
        help="True if you dont want to crete wandb logs.",
    )
    parser.set_defaults(disable_wandb=False)

    args = parser.parse_args(args)

    if args.load_trained_model_run_name is not None:
        validate_args(args)

    if args.seed_model_mask is None:
        args.seed_model_mask = args.seed_model_init

    if args.normalize_reconstruction == "None":
        args.normalize_reconstruction = None

    if args.dataset == "MNIST":
        args.patient_preprocessing = "raw"
    elif args.dataset in ["PBMC", "PBMC_small"]:
        args.tti_loss_hyperparam = 1
        args.num_pretrain_steps = 200
        args.max_steps = 1000

    return args


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=sklearn.exceptions.UndefinedMetricWarning
    )
    # warnings.filterwarnings("ignore", category=pytorch_lightning.utilities.warnings.LightningDeprecationWarning)

    print("Starting...")

    logging.basicConfig(
        filename=os.path.join(os.getcwd(), "logs_exceptions.txt"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    args = parse_arguments()

    if args.train_on_full_data and args.model in ["dnn", "dietdnn"]:
        assert args.path_steps_on_full_data

        # retrieve the number of steps to train
        aux = pd.read_csv(args.path_steps_on_full_data, index_col=0)
        conditions = {
            "dataset": args.dataset,
            "model": args.model,
            "sparsity_regularizer_hyperparam": args.sparsity_regularizer_hyperparam,
        }
        temp = aux.loc[
            (aux[list(conditions)] == pd.Series(conditions)).all(axis=1)
        ].copy()
        assert temp.shape[0] == 1

        args.max_steps = int(temp["median"].values[0])

    # set seeds
    args.seed_kfold = args.repeat_id
    args.seed_validation = args.test_split

    if args.dataset == "prostate" or args.dataset == "cll":
        # `val_check_interval`` must be less than or equal to the number of the training batches
        args.val_check_interval = 4

    """
	#### Parse dataset size
	when args.dataset=="metabric-dr__200" split into
	args.dataset = "metabric-dr"
	args.dataset_size = 200
	- 
	"""
    if "__" in args.dataset:
        args.dataset, args.dataset_size = args.dataset.split("__")
        args.dataset_size = int(args.dataset_size)

    #### Assert that the dataset is supported
    SUPPORTED_DATASETS = [
        "MNIST",
        "mice_protein",
        "COIL20",
        "gisette",
        "Isolet",
        "madelon",
        "USPS",
        "PBMC",
        "PBMC_small",
        "finance",
        "load_from_run_name",
    ]
    if args.dataset not in SUPPORTED_DATASETS:
        raise Exception(
            f"Dataset {args.dataset} not supported. Supported datasets are {SUPPORTED_DATASETS}"
        )

    if args.dataset == "load_from_run_name":
        assert args.load_trained_model_run_name is not None

    #### Assert custom evaluation with repeated dataset sampling
    if (
        args.evaluate_with_sampled_datasets
        or args.custom_train_size
        or args.custom_valid_size
        or args.custom_test_size
    ):
        assert args.evaluate_with_sampled_datasets
        assert args.custom_train_size
        assert args.custom_test_size
        assert args.custom_valid_size

    #### Assert sparsity parameters
    if args.sparsity_type:
        # if one of the sparsity parameters is set, then all of them must be set
        assert args.sparsity_gene_embedding_type
        assert args.sparsity_type
        assert args.sparsity_method
        assert args.sparsity_regularizer
        # assert args.sparsity_regularizer_hyperparam

    # add best performing configuration
    if args.use_best_hyperparams:
        # if the model uses gene embeddings of any type, then use dataset specific embedding sizes.
        if args.model in ["fsnet", "dietdnn"]:
            if args.dataset == "cll":
                args.wpn_embedding_size = 70
            elif args.dataset == "lung":
                args.wpn_embedding_size = 20
            else:
                args.wpn_embedding_size = 50

        if args.sparsity_type in ["global", "local"]:
            if args.dataset == "cll":
                args.sparsity_gene_embedding_size = 70
            elif args.dataset == "lung":
                args.sparsity_gene_embedding_size = 20
            else:
                args.sparsity_gene_embedding_size = 50

        elif args.model == "rf":
            params = {
                "cll": (3, 3),
                "lung": (3, 2),
                "metabric-dr": (7, 2),
                "metabric-pam50": (7, 2),
                "prostate": (5, 2),
                "smk": (5, 2),
                "tcga-2ysurvival": (3, 3),
                "tcga-tumor-grade": (3, 3),
                "toxicity": (5, 3),
            }

            args.rf_max_depth, args.rf_min_samples_leaf = params[args.dataset]

        elif args.model == "lasso":
            params = {
                "cll": 10,
                "lung": 100,
                "metabric-dr": 100,
                "metabric-pam50": 10,
                "prostate": 100,
                "smk": 1000,
                "tcga-2ysurvival": 10,
                "tcga-tumor-grade": 100,
                "toxicity": 100,
            }

            args.lasso_C = params[args.dataset]

        elif args.model == "tabnet":
            params = {
                "cll": (0.03, 0.001),
                "lung": (0.02, 0.1),
                "metabric-dr": (0.03, 0.1),
                "metabric-pam50": (0.02, 0.001),
                "prostate": (0.02, 0.01),
                "smk": (0.03, 0.001),
                "tcga-2ysurvival": (0.02, 0.01),
                "tcga-tumor-grade": (0.02, 0.01),
                "toxicity": (0.03, 0.1),
            }

            args.lr, args.tabnet_lambda_sparse = params[args.dataset]

        elif args.model == "lgb":
            params = {
                "cll": (0.1, 2),
                "lung": (0.1, 1),
                "metabric-dr": (0.1, 1),
                "metabric-pam50": (0.01, 2),
                "prostate": (0.1, 2),
                "smk": (0.1, 2),
                "tcga-2ysurvival": (0.1, 1),
                "tcga-tumor-grade": (0.1, 1),
                "toxicity": (0.1, 2),
            }

            args.lgb_learning_rate, args.lgb_max_depth = params[args.dataset]

    if args.hpc_run:
        torch.set_float32_matmul_precision("high")

    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    if args.dataset == "MNIST":
        args.reconstruction_loss = "bce"  # binary cross entropy
    else:
        args.reconstruction_loss = "mse"  # mean squared error

    args.test_time_interventions_in_progress = False

    if args.share_mask:
        if args.sigmoid_loss:
            # Using the sigmoid activation of the mask to get the sparsity loss
            args.sparsity_regularizer_hyperparam = 0.0
        else:
            # Using the soft gumbel activation of the mask to get the sparsity loss
            args.sparsity_regularizer_hyperparam_0 = 0.0

    if args.num_hidden is not None:
        args.R_num_hidden = args.P_num_hidden = args.num_hidden

    if args.hidden_dim is not None:
        args.R_hidden_dim = args.P_hidden_dim = args.hidden_dim

    run_experiment(args)
