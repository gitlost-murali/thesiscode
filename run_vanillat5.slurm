#!/bin/bash
#SBATCH --time=5:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH -J contrain-predict
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=m.m.c.kondragunta@student.rug.nl

cd /home/$USER/thesiscode/t5-scripts/ && srun python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 16 --langmodel_name t5-large --ckpt_folder /data/$USER/vanillat5-train --seed 1234

cd /home/$USER/thesiscode/t5-scripts/ && srun python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 16 --langmodel_name Salesforce/codet5-large --ckpt_folder /data/$USER/codet5-train --seed 1234 