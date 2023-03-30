#!/bin/sh
#SBATCH --job-name=BDT_train
#SBATCH --account=t3
#SBATCH --partition=standard
#SBATCH --mem=20000
#SBATCH --time=0-12:00
#SBATCH --output=./training_logs/BDT_data_cpu_%j.out

# Folder where the data is located for the training and testing of the BDT.
# Change so it suits your configuration.
data_path_train="/work/aoliver/Data/jet_images_c8_pt2.0_andre_train.csv"
data_path_test="/work/aoliver/Data/jet_images_c8_pt2.0_andre_test.csv"

# Default parameters
model_output_dir=/work/aoliver/BDT/Models/
model_name=BDT_from_csv

# Gather parameters given by user.
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Set up conda environment and cuda.
source /work/aoliver/miniconda3/bin/activate T3_masters

# Run the script with print flushing instantaneous.
export PYTHONUNBUFFERED=TRUE
python ./train_BDT_from_csv.py --data_path_train $data_path_train --data_path_test $data_path_test --model_output_dir $model_output_dir --model_name $model_name
export PYTHONUNBUFFERED=FALSE
