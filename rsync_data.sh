#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/home/er647/projects/feature-wise-active-learning/data/"
DEST_DIR="/rds/user/er647/hpc-work/FWAL/data/"

# Perform rsync from source to destination
rsync -avz $SOURCE_DIR $DEST_DIR

echo "Synchronization completed."
