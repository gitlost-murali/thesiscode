# T5 w/o explanation

## Running 

1. __Selecting a template:__ This folder contains multiple templates in `templatefile.py`. Decide which template you want to try out. Currently, we use the template __TBD__ that gave us __TBD__ results (Look at report for more details). If you want to try other templates, update `specific_utils.py` with the filename you want while importing TemplateHandler1/2 from `templatefile.py`. For example, change the following in `specific_utils.py` for template-2.

    From 

    `from templatefile import TemplateHandler1 as TemplateHander`

    to

    `from templatefile import TemplateHandler2 as TemplateHander`


2. Run the training script

```
python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 4 --langmodel_name Salesforce/codet5-large --ckpt_folder ./codet5-train --seed 1234 --device cpu
```

```
python train.py --train_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --dev_file ../data/SVAMP/cv_asdiv-a/fold0/dev.csv  --learning_rate 1e-4 --batch_size 4 --langmodel_name t5-large --ckpt_folder ./vanillat5-train --seed 1234 --device cpu

3. The best model will be stored in t5explain-files/best-model.ckpt

4. Evaluating the model on the test file

```
python evaluate.py --test_file ../data/SVAMP/cv_asdiv-a/fold0/train.csv --best_modelname ./vanillat5-train_t5-base_svamp_suffix/ --batch_size 16 --device cpu
```

5. Getting the predictions into a file

```
python predict.py --test_file ../data/test.tsv --best_modelname ./vanillat5-train_t5-base_svamp_suffix/  --batch_size 16 --device cpu --output_predfile preds.txt
```
