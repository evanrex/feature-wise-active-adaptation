#!/bin/bash

# Define the path where the .yaml files are located
YAML_PATH="scripts_experiments/secondary_analysis_PBMC"

# Define the path where the .wilkes3 files will be saved
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
    
    # Create the .wilkes3 file with the necessary sweep ID using the new template
    cat > "$OUTPUT_PATH/$base_name.wilkes3" <<- EOF
		#!/bin/bash
		#SBATCH -J $base_name
		#SBATCH -A COMPUTERLAB-SL2-GPU
		#SBATCH --nodes=1
		#SBATCH --ntasks-per-node=1
		#SBATCH --gres=gpu:1
		#SBATCH --time=10:00:00
		#SBATCH --mail-type=NONE
		#SBATCH -o ./logs/slurm-%j.out
		#SBATCH -e ./logs/slurm-%j.err
		#SBATCH -p ampere

		. /etc/profile.d/modules.sh
		module purge
		module load rhel8/default-amp

		export OMP_NUM_THREADS=1

		application="wandb agent"
		options="evangeorgerex/fwal/$sweep_id"
		workdir="\$SLURM_SUBMIT_DIR"
		numnodes=\$SLURM_JOB_NUM_NODES
		numtasks=\$SLURM_NTASKS
		mpi_tasks_per_node=\$(echo "\$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

		CMD="\$application \$options"

		cd \$workdir
		echo -e "Changed directory to \`pwd\`.\n"

		JOBID=\$SLURM_JOB_ID

		echo -e "JobID: \$JOBID\n======"
		echo "Time: \`date\`"
		echo "Running on master node: \`hostname\`"
		echo "Current directory: \`pwd\`"

		if [ "\$SLURM_JOB_NODELIST" ]; then
			export NODEFILE=\`generate_pbs_nodefile\`
			cat \$NODEFILE | uniq > machine.file.\$JOBID
			echo -e "\nNodes allocated:\n================"
			echo \`cat machine.file.\$JOBID | sed -e 's/\..*$//g'\`
		fi

		echo -e "\nnumtasks=\$numtasks, numnodes=\$numnodes, mpi_tasks_per_node=\$mpi_tasks_per_node (OMP_NUM_THREADS=\$OMP_NUM_THREADS)"

		echo -e "\nExecuting command:\n==================\n\$CMD\n"

		eval \$CMD 
	EOF

    # Output information about what was done
    echo "Generated script for sweep ID $sweep_id at $OUTPUT_PATH/$base_name.wilkes3"
done
