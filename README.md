# thesiscode
MWP and code

## Pre-requisites
* You can add a post-merge hook in the .git/hooks/ directory in your repository that runs the command "cd projutils/ && pip install ." after a successful pull.

1. First, navigate to the .git/hooks/ directory in your repository. Create a new file called "post-merge" (without any file extension). Open the file in a text editor and add the following line:

```bash
#!/bin/sh
cd projutils/ && pip install .
```

2. Make the file executable by running chmod +x post-merge
3. Now every time you pull from the repository, the post-merge hook will automatically run the command "cd projutils/ && pip install ." after the pull is complete.

## Baseline scripts

1. `cd t5-scripts/` and run `python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 4 --langmodel_name t5-large --ckpt_folder ./vanillat5-train --seed 1234 --device cpu --equation_order suffix --dataset_name svamp`

2. `python predict.py --test_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv --best_modelname vanillat5-train_t5-base_svamp_suffix --output_predfile preds.txt`

2. To use code-T5, use the flag `--langmodel_name`
    
    a. `cd t5-scripts/` 

    b.`python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 4 --langmodel_name Salesforce/codet5-large --ckpt_folder ./vanillat5-train --seed 1234 --device cpu --equation_order suffix --dataset_name svamp`