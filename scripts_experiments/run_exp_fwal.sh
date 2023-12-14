python src/run_experiment.py \
	--model 'fwal' \
	--dataset 'simple_synth' \
	--max_steps 10 \
	--wpn_embedding_type 'histogram' \
	--use_best_hyperparams \
	--tags 'bad' \     # Run trial runs with this tag, so you know this run is junk
	# --disable_wandb \  # Disable wanb completly