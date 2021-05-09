import numpy as np 
import pandas as pd 
import torch 
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from .data import  TestDataset
import os
import random
from tqdm import auto
from collections import defaultdict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        
def get_predictions(dataset, model, tokenizer):
    predictions_probs = defaultdict(lambda:np.zeros(20))
    for i in range(len(dataset)):
        id_ = dataset.ids[i]
        inputs = dataset[i]
        keys = list(inputs.keys())
        for key in keys:
            inputs[key] = inputs[key].unsqueeze(0).to(device)
        # output = model(**inputs)
        probs = torch.softmax(model(**inputs).logits, dim=1)
        predictions_probs[id_] += probs.squeeze(0).detach().cpu().numpy()
    predictions = {}
    for key, val in predictions_probs.items():
        predictions[key] = val.argmax()
    return predictions
    
        
def main():
    dataset_path = "dataset"
    max_length = 256
    test_df = pd.read_csv(os.path.join(dataset_path, "Test.csv"))
    train_df = pd.read_csv(os.path.join(dataset_path, "Train.csv"))
    model_dir = os.path.join(dataset_path, "classifier-model", "smaller-model-256")
    tokenizer_path = os.path.join(dataset_path, "masked-lang", "working-dir",  "tokenizer")
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=max_length)

    testset = TestDataset(test_df, tokenizer, max_length=max_length)

    predictions = get_predictions(testset, model, tokenizer)

    classes = list(train_df.Label.unique())
    classes.sort()
    index2class = {i:classes[i] for i in range(len(classes))}

    pred_probs = []
    for index, row in test_df.iterrows():
        pred_probs.append(index2class[predictions[row["ID"]]])
    submission = pd.DataFrame(dict(ID=test_df.ID, Label=pred_probs))
    submission.to_csv("finetuned-best.csv", index=False)

if __name__ == '__main__':
    main()
    