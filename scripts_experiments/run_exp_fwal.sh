python src/run_experiment.py \
	--model 'fwal' \
	--dataset 'simple_synth' \
	--valid_percentage '0.25' \
	--sparsity_regularizer 'L1' \
	--sparsity_regularizer_hyperparam '1.0' \
	--gamma '1.0' \
	--wpn_embedding_type 'histogram' \
	--use_best_hyperparams \
	--notes 'experiments with sparsity and reconstruction loss combination' \
	--tags 'fwal_first' 'sparsity_regularizer_hyperparam' \     # Run trial runs with "bad" tag, so you know the run is junk
	# --disable_wandb \  # Disable wanb completly