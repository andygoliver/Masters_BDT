#!/bin/sh
#SBATCH --job-name=deepsets_net_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=0-01:30
#SBATCH --output=./logs/deepsets_gpu_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/deodagiu/ki_data/intnet_input

# Default parameters for running the deepsets training.
norm=nonorm
train_events=-1
lr=0.001
batch=128
epochs=70
valid_split=0.3
optimiser=adam
loss=softmax_with_crossentropy
metrics=categorical_accuracy
outdir=test
type=vanilla
jet_seed=123
seed=127

# Gather parameters given by user.
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Set up conda environment and cuda.
source /work/deodagiu/miniconda/bin/activate ki_intnets

# Run the script with print flushing instantaneous.
export PYTHONUNBUFFERED=TRUE
./deepsets_train --data_folder $DATA_FOLDER --norm ${norm} --train_events ${train_events} --lr ${lr} --batch ${batch} --epochs ${epochs} --valid_split ${valid_split} --optimiser ${optimiser} --loss ${loss} --metrics ${metrics} --outdir ${outdir} --seed ${seed} --deepsets_type ${type} --jet_seed ${jet_seed}
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
mv ./logs/deepsets_gpu_${SLURM_JOBID}.out ./trained_deepsets/${outdir}