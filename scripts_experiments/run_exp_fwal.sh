python src/run_experiment.py \
	--model 'fwal' \
	--dataset 'simple_synth' \
	--valid_percentage '0.25' \
	--wpn_embedding_type 'histogram' \
	--use_best_hyperparams \
	--tags 'fwal_first' \     # Run trial runs with "bad" tag, so you know the run is junk
	# --disable_wandb \  # Disable wanb completly