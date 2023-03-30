#!/bin/sh
#SBATCH --job-name=BDT_prepare_data
#SBATCH --account=t3
#SBATCH --partition=standard
#SBATCH --mem=8000
#SBATCH --time=0-01:00
#SBATCH --output=./preparation_logs/BDT_data_cpu_%j.out

data_file_dir=/work/aoliver/Data/train/
output_dir=/work/aoliver/Data/

# Default parameters for preparing data

min_pt=0
max_constituents=8
type='jedinet'
flag='train'
sorted_feature='pT'
sort_ascending=False
positive_class=None


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
# python test.py
python prepare_data_BDT.py $data_file_dir $output_dir --min_pt $min_pt --max_constituents $max_constituents --type $type --flag $flag --sorted_feature $sorted_feature --sort_ascending $sort_ascending --positive_class $positive_class
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
# mv ./logs/deepsets_gpu_${SLURM_JOBID}.out ./preparation_logs/