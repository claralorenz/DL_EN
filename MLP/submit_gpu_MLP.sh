#!/usr//bin/zsh

#SBATCH --account=rwth1434

#SBATCH --output=output.%J.txt

#SBATCH --time=0-120:00:00

#SBATCH --mail-type=END

#SBATCH --mail-user=anna.morelli@rwth-aachen.de

#SBATCH --mem-per-cpu=3900M


#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=48

#SBATCH --exclusive

module load Python/3.10.4
module load sklearn
module load tensorflow
module load pandas
module load numpy
module load random
module load matplotlib

#Run your Python script
python mlp_gs.py

