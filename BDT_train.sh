#!/bin/sh
#SBATCH --job-name=naggs
#SBATCH --account=t3
#SBATCH --partition=standard
#SBATCH --mem=16000
#SBATCH --time=0-12:00
#SBATCH --output=./training_logs/BDT_data_cpu_%j.out

# Folder where the data is located for the training and testing of the BDT.
# Change so it suits your configuration.
data_path_train="/work/aoliver/Data/binary_class/jet_images_c0_pct_agg_c1-2-3-5-10-20-50-150_train.csv"
# data_path_test="/work/aoliver/Data/jet_images_c8_pt2.0_andre_test.csv"

# Default parameters
# nb_jets=None
model_output_dir=/work/aoliver/BDT/Models/Binary_class/
model_name=BDT_c0_pct_agg_c1-2-3-5-10-20-50-150

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
python ./train_BDT_from_csv.py --data_path_train $data_path_train --model_output_dir $model_output_dir --model_name $model_name
export PYTHONUNBUFFERED=FALSE
