"""## INFERENCE from checkpoint"""
import json
from specific_utils import LitModel
import pytorch_lightning as pl
from specific_utils import Inference_LitOffData
import argparse
import torch
from projutils import write_preds_tofile, debug_wo_template

def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_file", type=str, default='../../data/dev.tsv', required= True,
                        help="If added, use trained model to predict on test set")

    parser.add_argument("--output_predfile", type=str, default='preds.txt',
                        help="File to store the predictions. Each prediction in a line")

    parser.add_argument("--debug_file", type=str, default='debug.csv',
                        help="Shows failed instances and all their predictions")

    parser.add_argument("--best_modelname", default="models/bestmodel.ckpt", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--device", default="gpu", type=str,
                        help="Type of device to use. gpu/cpu strict naming convention")

    args = parser.parse_args()
    return args


def main():
    '''Main function to test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)

    # load args.best_modelname/args.json
    with open(args.best_modelname + "/args.json", "r") as f:
        model_args = json.load(f)

    model = LitModel(modelname = model_args["langmodel_name"],
                    learning_rate = model_args["learning_rate"],)

    model.load_from_checkpoint(model_args["checkpoint_path"])

    model.eval()
    testdm = Inference_LitOffData(test_file = args.test_file,
                                batch_size = args.batch_size,
                                modelname = model_args["langmodel_name"],
                                datasetname=model_args["dataset_name"],
                                equation_order=model_args["equation_order"],
                                max_label_len=model_args["max_label_len"],
                                max_seq_len=model_args["max_seq_len"])

    device_to_train = args.device if torch.cuda.is_available() else "cpu"
    print("Device to use ", device_to_train)
    trainer = pl.Trainer(accelerator=device_to_train, devices=1)
    trainer.test(model, testdm.test_dataloader())

if __name__ == '__main__':
    main()
