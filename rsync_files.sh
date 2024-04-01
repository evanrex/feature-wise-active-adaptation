#!/bin/bash

# Define source and destination directories

SOURCE_DIR="/home/er647/projects/feature-wise-active-learning/"
DEST_DIR="/rds/user/er647/hpc-work/FWAL/data/"

# Array of directories to sync
DIRS=("wandb" "fwal")

# Loop through each directory and perform rsync
for dir in "${DIRS[@]}"; do
    echo "Syncing $dir..."
    # Syncing the directory from source to destination
    # --remove-source-files removes the files from the source after they're transferred
    rsync -av --remove-source-files "$SOURCE_DIR/$dir/" "$DEST_DIR/$dir/"
done

echo "Sync complete."
