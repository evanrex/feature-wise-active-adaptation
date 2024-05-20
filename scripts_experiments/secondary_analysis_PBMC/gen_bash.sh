#!/bin/bash

# Define the path where the .yaml files are located
YAML_PATH="scripts_experiments/secondary_analysis_PBMC"

# Define the path where the .sh files will be saved
OUTPUT_PATH="slurm_scripts/secondary_analysis_PBMC"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Iterate over all .yaml files in the specified directory
for yaml_file in "$YAML_PATH"/*.yaml; do
    # Extract the filename without path and extension
    base_name=$(basename "$yaml_file" .yaml)

    # Run the wandb sweep command and capture the output
        output=$(wandb sweep "$yaml_file" 2>&1) # Ensure stderr is also captured in case the ID is sent there

    # Debug: Output the result to check manually
    echo "Output from wandb sweep:"
    echo "$output"
    
    # Extract the sweep ID from the output
    # The sweep ID appears after 'Creating sweep with ID:'
    sweep_id=$(echo "$output" | grep 'Creating sweep with ID:' | awk '{print $NF}')

    # Check if the sweep_id was captured
    if [ -z "$sweep_id" ]; then
        echo "Failed to extract sweep ID for $yaml_file"
        continue # Skip this file and go to the next
    fi
    
    # Create the .sh file with the necessary sweep ID
    cat > "$OUTPUT_PATH/$base_name.sh" <<- EOF
		#!/bin/bash
		#SBATCH -J $base_name
		#SBATCH --output=logs/out_%A.out
		#SBATCH --error=logs/err_%A.err
		#SBATCH -A COMPUTERLAB-SL2-CPU
		#SBATCH --time=10:00:00
		#SBATCH -p icelake
		#SBATCH --nodes=1
		#SBATCH --ntasks-per-node=1

		. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
		module purge                               # Removes all modules still loaded
		module load rhel8/default-amp              # REQUIRED - loads the basic environment

		wandb agent evangeorgerex/fwal/$sweep_id
	EOF

    # Output information about what was done
    echo "Generated script for sweep ID $sweep_id at $OUTPUT_PATH/$base_name.sh"
done

