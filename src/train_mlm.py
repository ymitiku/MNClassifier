import numpy as np 
import pandas as pd 
import torch 
from transformers import RobertaForMaskedLM,  RobertaConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from .data_mlm import MaskedDatasetPreparer, get_train_validation_dataset
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
    
    working_dir = os.path.join(dataset_path, "masked-lang", "working-dir")
    
    dataset_preparer = MaskedDatasetPreparer("dataset", max_word_frequency=200)
    dataset_preparer.prepare_dataset()
    
    trainset, validset = get_train_validation_dataset(working_dir=working_dir)
    
        
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=trainset.dataset.tokenizer, mlm=True, mlm_probability=0.15
    )
     

    config = RobertaConfig(
        vocab_size=dataset_preparer.vocabulary_size,
        max_position_embeddings=64 + 2,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config=config)
    model.num_parameters()
    model = model.to(device)
    
    

    training_args = TrainingArguments(
        output_dir= os.path.join(working_dir, "logs"),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=128,
        save_steps=1000,
        save_total_limit=2,
        logging_steps = 500,
        evaluation_strategy = "steps",
        eval_steps = 500,
        load_best_model_at_end= True,
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=trainset,
        eval_dataset = validset
    )
    
    trainer.train()
    trainer.save_model(os.path.join(working_dir, "model"))


    
if __name__ == '__main__':
    main()
    