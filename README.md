# thesiscode
MWP and code

1. `cd t5-scripts/` and run `python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 4 --langmodel_name t5-large --ckpt_folder ./vanillat5-train --seed 1234 --device cpu --equation_order suffix --dataset_name svamp`

2. `python predict.py --test_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv --best_modelname vanillat5-train_t5-base_svamp_suffix --output_predfile preds.txt`

2. To use code-T5, use the flag `--langmodel_name`
    
    a. `cd t5-scripts/` 

    b.`python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 4 --langmodel_name Salesforce/codet5-large --ckpt_folder ./vanillat5-train --seed 1234 --device cpu --equation_order suffix --dataset_name svamp`