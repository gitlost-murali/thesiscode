import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from sklearn.metrics import f1_score, accuracy_score

from projutils import read_corpus, calculate_confusion_matrix, plot_confusion_matrix

# from templatefile import TemplateHandler1 as TemplateHandler

class T5DatasetClass(Dataset):
    def __init__(self, questions, numbers, 
                       equations, answers, 
                       tokenizer, max_length,
                       label_maxlen = 30):
        super(T5DatasetClass, self).__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.questions=questions
        self.numbers=numbers
        self.equations=equations
        self.answers=answers
        # self.labels = [ self.labelmapper[lb] for lb in self.orglabels]
        self.labels = [lb for lb in self.equations]
        self.label_maxlen = label_maxlen

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        text1 = self.questions[index]
        inputs = self.tokenizer(
            text = text1 ,
            text_pair = None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        target_encoding = self.tokenizer(
            self.labels[index],
            max_length=self.label_maxlen,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        label = target_encoding.input_ids
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        label[label == self.tokenizer.pad_token_id] = -100

        output =  {
            'ids': ids[0],
            'mask': mask[0],
            'labels': label[0],
            'sent_idx': torch.tensor(index, dtype=torch.long),
            'actual_label': self.labels[index]
            }

        return output


class LitOffData(pl.LightningDataModule):
    def __init__(self,
                 train_file: str = 'data/train.tsv',
                 dev_file: str = 'data/dev.tsv',
                 batch_size = 4,
                 max_seq_len = 150,
                 modelname = 't5-base',
                 datasetname = 'svamp',
                 equation_order = 'suffix',
                 max_label_len = 30,
                ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file, self.dev_file = train_file, dev_file
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.datasetname = datasetname
        self.equation_order = equation_order
        self.label_maxlen = max_label_len
        self.read_data()
        
    def read_data(self):
        # Read in the data
        self.question_train, self.numbers_train,\
        self.equation_train, self.answer_train = read_corpus(filename = self.train_file,
                                                             dataname = self.datasetname,
                                                             order = self.equation_order)
        self.question_dev, self.numbers_dev,\
        self.equation_dev, self.answer_dev = read_corpus(filename = self.dev_file,
                                                         dataname = self.datasetname,
                                                         order = self.equation_order)

    def setup(self, stage = None):
        self.train_dataset= T5DatasetClass(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                           questions = self.question_train, 
                                           numbers = self.numbers_train, 
                                           equations = self.equation_train,
                                           answers = self.answer_train,
                                           label_maxlen=self.label_maxlen,
                                        )
        self.val_dataset= T5DatasetClass(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                         questions = self.question_dev, 
                                         numbers = self.numbers_dev, 
                                         equations = self.equation_dev,
                                         answers = self.answer_dev,
                                         label_maxlen=self.label_maxlen,
                                        )

    def train_dataloader(self):
        dataloader=DataLoader(dataset=self.train_dataset,batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        dataloader=DataLoader(dataset=self.val_dataset,batch_size=self.batch_size)
        return dataloader

# define the LightningModule

class LitModel(pl.LightningModule):
    def __init__(self, modelname = "t5-base",
                 dropout = 0.2, learning_rate = 1e-5, batch_size = 4,
                 save_cm_plot = True):
        super().__init__()
        self.base = T5ForConditionalGeneration.from_pretrained(modelname)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_cm_plot = save_cm_plot
        self.log("batch_size", self.batch_size)

    def forward(self, ids, mask, labels, **kwargs):
        out = self.base( input_ids = ids,attention_mask=mask, labels=labels, return_dict=True)
        # https://stackoverflow.com/questions/73314467/output-logits-from-t5-model-for-text-generation-purposes
        # For generating stuff greedily. For debugging purpose only. Can be ignored for now.
        loss = out.loss
        return loss
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self(**batch)
        loss = torch.mean(loss)

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(**batch)
        loss = torch.mean(loss)

        self.log('val_loss', loss,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        outputs = self.base.generate(batch["ids"])
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [self.templatehandler.decode_preds(prd) for prd in preds]
        gts = [self.templatehandler.decode_preds(snt) for snt in batch["actual_label"]]
        return {"preds": preds, "gts": gts}

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.base.generate(batch["ids"])
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [self.templatehandler.decode_preds(prd) for prd in preds]
        gts = [self.templatehandler.decode_preds(snt) for snt in batch["actual_label"]]
        input_sentences = self.tokenizer.batch_decode(batch['ids'], skip_special_tokens = True)
        return {"input_sentences": input_sentences, "preds": preds, "gts": gts}

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        losses = torch.stack([ item["loss"]  for item in outs])
        loss = losses.mean()
        self.log("val_loss_epoch", loss)
        print("val_loss_epoch", loss)

    def test_epoch_end(self,outs):
        # outs is a list of whatever you returned in `validation_step`

        preds = []
        gts = []        
        for item in outs:
            preds.extend(item["preds"])
            gts.extend(item["gts"])

        acc = accuracy_score(preds, gts)
        f1 = f1_score(preds, gts, average='macro')

        if self.save_cm_plot:
            # get the classnames from encoder
            matrix = calculate_confusion_matrix(gts, preds, list(set(gts+preds)) )
            plot_confusion_matrix(matrix)

        self.log("test_epoch_acc", acc)
        self.log("test_epoch_f1", f1)
        print("test_acc_epoch", acc)
        print("test_F1_epoch", f1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {"scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                        "monitor": "train_loss_step", "mode": "min", "interval": "step", "frequency": 5}
        return [optimizer], [lr_scheduler]


class Inference_LitOffData(pl.LightningDataModule):
    def __init__(self, 
                 test_file: str = 'data/test.tsv',
                 batch_size = 4,
                 max_seq_len = 100,
                 modelname = 't5-base',
                 datasetname = 'svamp',
                 equation_order = 'suffix',
                 max_label_len = 30,
                ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.test_file = test_file
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.datasetname = datasetname
        self.equation_order = equation_order
        self.label_maxlen = max_label_len
        self.read_data()
        self.setup()

    def read_data(self):
        # Read in the data
        self.question_test, self.numbers_test,\
        self.equation_test, self.answer_test = read_corpus(filename = self.test_file,
                                               dataname = self.datasetname,
                                               order = self.equation_order)

    def setup(self, stage = None):
        self.test_dataset= T5DatasetClass(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                          questions = self.question_test, 
                                          numbers = self.numbers_test, 
                                          equations = self.equation_test,
                                          answers = self.answer_test,
                                          label_maxlen=self.label_maxlen,)

    def test_dataloader(self):
        dataloader=DataLoader(dataset=self.test_dataset,batch_size=self.batch_size)    
        return dataloader
