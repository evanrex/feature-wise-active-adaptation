# F-Act - Feature-wise Active adaptation

Codebase for the HI-AI@KDD2024 Worshop paper [From Must to May: Enabling Test-Time Feature
Imputation and Interventions](https://human-interpretable-ai.github.io/assets/pdf/9_From_Must_to_May_Enabling_Te.pdf).


## F-Act Architecture

![F-Act](https://github.com/evanrex/feature-wise-active-adaptation/blob/main/assets/F-Act-1.png)


## Installing the project 
You must have **conda** installed locally. All project dependencies all included in the conda file: environment.yml.

This was tested to work on: CUDA Version: 11.2 (NVIDIA-SMI 460.32.03).

```
<!-- Install the codebase -->
cd REPOSITORY
conda create python=3.10.12 --name fact
conda activate fact
pip install -r requirements.txt


Change `BASE_DIR` from `/src/_config.py` to point to the project directory on your machine.

Search `er647`in the codebase and replace the paths with the paths on your machine.

When you install new modules, save the environment:
```
pip freeze > requirements.txt
```

## Codebase basics
- Store training metrics using **wandb**. It makes running experiments much easier.
- Train using **pytorch-lightening**, which is a wrapper on top of pytorch. It has three main benefits: (1) it make training logic much simpler, (2) it allows us to use the same code for training on CPU or GPU, and (3) integrates nicely with wandb.

## Code structure
- src
	- `run_experiment.py`: code for parsing arguments, and starting experiment
		- def parse_arguments - include all command line arguments
		- def run_experiment - create wandb logger and start training model
		- def train_model - training neural networks using pytorch lightning
		- important commandline arguments
			- dataset
			- model
]			- max_steps - maximum training iterations
			- dropout_rate
			- lr, batch_size, patience_early_stopping
			- lr_scheduler - learning rate scheduler (the default scheduler reduces the learning rate until it gets 10x smaller)


	- `dataset.py`: loading the datasets
	- `model.py`: neural network architecture
		- def create_model - logic to create a neural network
		- class TrainingLightningModule - generic class that includes logging losses etc using pytorch-lightning
		- class FWAL_Hierarchical - for F-Act
