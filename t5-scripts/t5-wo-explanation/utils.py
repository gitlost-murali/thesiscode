import pandas as pd
import sys
sys.path.append("../")

def read_corpus(filename = "data/train.tsv", delimiter = ","):
    df = pd.read_csv(filename, sep=delimiter)
    question = df['Question'].values.tolist()
    numbers = df['Numbers'].values.tolist()
    equation = df['Equation'].values.tolist()
    answer = df['Answer'].values.tolist()
    return question, numbers, equation, answer

def replace_nums(pattern, operands):
    """
    pattern = "+ - number2 number1 number0"
    operands = [5,3,2]
    """
    for i in range(len(operands)):
        pattern = pattern.replace("number"+str(i),str(operands[i]))
    return pattern