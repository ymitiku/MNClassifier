import numpy as np 
import pandas as pd 
import torch 
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from .data import  get_classification_train_validation_dataset
from transformers import pipeline
import os
import random


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def seed_all(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    # if you are suing GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
def main():
    seed_val = 42
    
    EPOCHS = 10
    
    seed_all(seed_val)
    
    dataset_path = "dataset"
    
    mlm_working_dir = os.path.join(dataset_path, "masked-lang", "working-dir")
    working_dir = os.path.join(dataset_path, "classifier-model")
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
 
    fillmask = pipeline('fill-mask', model=os.path.join(mlm_working_dir, "smaller-model"), tokenizer=os.path.join(mlm_working_dir, "tokenizer"))

    trainset, validset = get_classification_train_validation_dataset(dataset_path, os.path.join(mlm_working_dir, "tokenizer"), fillmask=fillmask)
  
     

    model = RobertaForSequenceClassification.from_pretrained(os.path.join(mlm_working_dir, "smaller-model"), num_labels=len(trainset.class2index))
    model = model.to(device)
    

    training_args = TrainingArguments(
        output_dir= os.path.join(working_dir, "logs"),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=128,
        save_steps=1000,
        save_total_limit=2,
        logging_steps = 100,
        evaluation_strategy = "steps",
        eval_steps = 100,
        load_best_model_at_end= True,
        warmup_steps=500,  
        weight_decay=0.01,  
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset = validset
    )
    
    trainer.train()
    trainer.save_model(os.path.join(working_dir, "model"))


    
if __name__ == '__main__':
    main()
    