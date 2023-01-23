
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from projutils.asthandler import ASTHandler

def calculate_confusion_matrix(Y_test, y_pred, labels,):
    matrix = confusion_matrix(Y_test, y_pred)
    # Convert to pandas dataframe confusion matrix.
    print(matrix)
    matrix = (pd.DataFrame(matrix, index=labels, columns=labels))
    return matrix

def plot_confusion_matrix(matrix):
    fig, _ = plt.subplots(figsize=(9, 8))
    sn.heatmap(matrix, annot=True, cmap=plt.cm.Blues, fmt='g')
    # show the picture
    # plt.show()
    fig.savefig("heatmap.png")
    return

def write_to_textfile(input_string, filename):
    with open(filename, "w", encoding="utf8") as fh:
        fh.write(input_string)

def write_preds_tofile(outs, predkeyname, filename):
    total_preds = []
    for item in outs:
        batch_preds = item[predkeyname]
        total_preds.extend(batch_preds)
    write_to_textfile("\n".join(total_preds), filename)

def flatten_batchdict(outs):
    flatten_items = dict()
    keys_names = list(outs[0].keys())
    
    for eachkey in keys_names:
        flatten_items[eachkey] = []

    for item in outs:
        for eachkey in keys_names:
            flatten_items[eachkey].extend(item[eachkey])
    return flatten_items

def debug_wo_template(outs, filename):
    incorrect_items = []
    flatten_items = flatten_batchdict(outs)
    for snt, prd, gt in zip(flatten_items['input_sentences'],flatten_items['preds'], flatten_items['gts']):
        if prd != gt:
            incorrect_items.append([snt,prd,gt])
    df = pd.DataFrame(incorrect_items, columns=list(outs[0].keys()))
    df.to_csv(filename)

def debug_w_template(outs, filename):
    incorrect_items = []
    flatten_items = flatten_batchdict(outs)
    for snt, e_prd, e_gt, prd, gt in zip(flatten_items['input_sentences'],
                            flatten_items['preds'], flatten_items['gts'],
                            flatten_items['onlylabel_preds'], flatten_items['onlylabel_gts']):
        if prd != gt:
            incorrect_items.append([snt,prd,gt, e_prd, e_gt])
    df = pd.DataFrame(incorrect_items, columns = list(outs[0].keys()))
    df.to_csv(filename)


def read_corpus(filename = "data/train.tsv", delimiter = ",",
                dataname = "svamp", order = "suffix"):
    
    asthandler = ASTHandler()

    if dataname == "svamp":
        df = pd.read_csv(filename, sep=delimiter)
        question = df['Question'].values.tolist()
        numbers = df['Numbers'].values.tolist()
        df['Equation'] = df['Equation'].str.split()
        if order=="suffix":
            # Convert equation from prefix to suffix
            equation = df['Equation'].apply(asthandler.prefix2suffix).values.tolist()
        elif order=="infix":
            # Convert equation from prefix to infix
            equation = df['Equation'].apply(asthandler.prefix2infix).values.tolist()
        answer = df['Answer'].values.tolist()
        return question, numbers, equation, answer
    else:
        raise NotImplementedError("Only svamp dataset is implemented")

def replace_nums(pattern, operands):
    """
    pattern = "+ - number2 number1 number0"
    operands = [5,3,2]
    """
    for i in range(len(operands)):
        pattern = pattern.replace("number"+str(i),str(operands[i]))
    return pattern