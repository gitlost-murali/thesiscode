# thesiscode
MWP and code

1. `cd t5-scripts/t5-wo-explanation/` and run `python train.py --train_file ../../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../../data/SVAMP/cv_asdiv-a/fold0/dev.csv --batch_size 4 --device cpu`

2. To use code-T5, use the flag `--langmodel_name`
    
    a. `cd t5-scripts/t5-wo-explanation/` 

    b.`python train.py --train_file ../../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../../data/SVAMP/cv_asdiv-a/fold0/dev.csv --batch_size 4 --device cpu --langmodel_name Salesforce/codet5-large`