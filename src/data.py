from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os 
import numpy as np 
from tqdm.auto import tqdm
import re
from collections import Counter
import string
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from tokenizers.processors import BertProcessing
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch


def split_to_chunks(sent, max_length):
    sents = re.split("[\.?!]", sent)
    sent_lengths = []
    for sent in sents:
        sent_lengths.append({"sent":sent, "length":len(sent.split())})
        
    current = None
    current_length = None
    output = []
    for i in range(len(sent_lengths)):
        if current is None:
            current = sent_lengths[i]["sent"]
            current_length = sent_lengths[i]["length"]
        elif current_length + sent_lengths[i]["length"] > max_length:
            output.append(current)
            current = sent_lengths[i]["sent"]
            current_length = sent_lengths[i]["length"]
        else:
            current += ". " +sent_lengths[i]["sent"]
            current_length += sent_lengths[i]["length"]
    if current is not None:
        output.append(current)
    return output
    
class OriginalDataset(Dataset):
    def __init__(self, dataset_path, is_train = True, return_ids = False):
        super().__init__()
        self.df = pd.read_csv(dataset_path)
        self.is_train = is_train
        self.return_ids = return_ids
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        if not (self.is_train or self.return_ids):
            return row["Text"]
        elif self.is_train:
            return row["Text"], row["Label"]
        elif self.return_ids:
            return {"id":row["ID"], "text":row["Text"]}

        else:
            return {"id":row["ID"], "text":row["Text"]}, row["Label"]
class SanitizedDatasetMLM(Dataset):
    def __init__(self, dataset_path, is_train, return_ids):
        super().__init__()
        self.is_train = is_train
        self.return_ids = return_ids
        self.build_dataset(dataset_path)
    def build_dataset(self, dataset_path):
        self.word_counts = Counter()
        
        print("Builing dataset")
        original_dataset = OriginalDataset(dataset_path, self.is_train, self.return_ids)
        self.texts = []
        if self.return_ids:
            self.ids = []
        if self.is_train:
            self.labels = []
        for (index, batch) in (tqdm(enumerate(original_dataset))):
            if self.return_ids:
                if self.is_train:
                    inputs, label = batch
                else:
                    inputs = batch
                text = inputs["text"]
                id_ = inputs["id"]
            else:
                if self.is_train:
                    text, label = batch
                else:
                    text = batch
            
            self.texts.append(text)
            if self.return_ids:
                self.ids.append(id_)
            if self.is_train:
                self.labels.append(label)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        
        if self.return_ids:
            if self.is_train:
                return {"id":self.ids[index], "text":self.texts[index]}, self.labels[index]
            else:
                return {"id":self.ids[index], "text":self.texts[index]}
        else:
            if self.is_train:
                return self.texts[index], self.labels[index]
            else:
                return self.texts[index]
        
class MaskedDatasetPreparer():
    def __init__(self, dataset_path,  working_folder = None, vocabulary_size = 10_000, max_length = 256):
        super().__init__()
        if not os.path.exists(os.path.join(dataset_path, "masked-lang")):
            os.mkdir(os.path.join(dataset_path, "masked-lang"))
        self.working_dir = os.path.join(dataset_path, "masked-lang", "working-dir") if working_folder is None else working_folder
        
        self.train_path = os.path.join(dataset_path, "Train.csv")
        self.test_path = os.path.join(dataset_path, "Test.csv")
        self.vocabulary_size = vocabulary_size
        self.max_length = max_length
    
        
        
    def prepare_dataset(self):
        sentences = []
        trainset = SanitizedDatasetMLM(self.train_path, is_train=False, return_ids=False)
        testset = SanitizedDatasetMLM(self.test_path, is_train=False, return_ids=False)
        
        for i in range(len(trainset)):
            sent = trainset[i]
            sents = split_to_chunks(sent, self.max_length)
            sentences.extend(sents)
        for i in range(len(testset)):
            sent = testset[i]
            sents = split_to_chunks(sent, self.max_length)
            sentences.extend(sents)
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)
        if not os.path.exists(os.path.join(self.working_dir, "tokenizer")):
            os.mkdir(os.path.join(self.working_dir, "tokenizer"))
    
        
        with open(os.path.join(self.working_dir, "sentences.txt"), "w") as outfile:
            for sent in sentences:
                outfile.write("{}\n".format(sent))
        paths = [os.path.join(self.working_dir, "sentences.txt")]

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.enable_padding(pad_token="<pad>", length=self.max_length)
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.train(files=paths, vocab_size=self.vocabulary_size, min_frequency=1, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
                
        tokenizer_save_path = os.path.join(self.working_dir, "tokenizer") 
        tokenizer.save_model(tokenizer_save_path)
    
        
class MaskedDataset(Dataset):
    def __init__(self, working_dir, tokenizer):
        self.working_dir = working_dir
        self.tokenizer = tokenizer
        src_file = Path(os.path.join(self.working_dir, "sentences.txt"))
        lines = src_file.read_text(encoding="utf-8").splitlines()
       
        self.inputs = [self.tokenizer.encode(x, padding="max_length", truncation=True ) for x in tqdm(lines)]
        
        np.random.shuffle(self.inputs)
        
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        return self.inputs[index]
class MaskedDatasetSplit(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
def get_mlm_train_validation_dataset(working_dir, train_size = 0.8, max_length = 256):
    tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(working_dir, "tokenizer"), max_len=max_length)
        
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.convert_tokens_to_ids("</s>")),
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.convert_tokens_to_ids("</s>")),
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
    )
    
    all_dataset = MaskedDataset(working_dir, tokenizer)
    
    all_indices = np.arange(len(all_dataset))
    
    np.random.shuffle(all_indices)
    
    train_size = int(train_size * len(all_indices))
    train_indices = all_indices[:train_size]
    valid_indices = all_indices[train_size:]

    trainset = MaskedDatasetSplit(all_dataset, train_indices)
    validset = MaskedDatasetSplit(all_dataset, valid_indices)
    
    return trainset, validset

def get_classification_train_validation_dataset(dataset_path, tokenizer_path, fillmask, train_size = 0.8, max_length=256, augment_size = 5):
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=max_length)
        
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.convert_tokens_to_ids("</s>")),
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.convert_tokens_to_ids("</s>")),
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
    )
    
    df = pd.read_csv(os.path.join(dataset_path, "Train.csv"))
    train_df, valid_df = train_test_split(df, train_size = train_size)
    trainset = ClassificationDataset(train_df, tokenizer, fill_mask=fillmask, augument_size=augment_size)
    validset = ClassificationDataset(valid_df, tokenizer, fill_mask=fillmask, is_train=False)
    
    return trainset, validset


    
class ClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, is_train = True, fill_mask = None, augument_size = 5, max_length = 256):
        super().__init__()
        self.fill_mask = fill_mask
        self.encodings = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_size = augument_size
        all_texts = []
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            text = row["Text"]
            texts = split_to_chunks(text, self.max_length)
            for text in texts:
                if is_train:
                    encoded = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length)
                    txt = self.tokenizer.decode(encoded["input_ids"])
                    words = txt.split()
                    m = min(len(words), self.max_length)
                    K = np.random.randint(1, m-1)
                    masked_sentence = " ".join(words[:K]  + [self.fill_mask.tokenizer.mask_token] + words[K+1:])
                    predictions = self.fill_mask(masked_sentence, top_k = augument_size)
                    augmented_sentences = [predictions[i]['sequence'] for i in range(augument_size)]
            
                    all_texts.extend(augmented_sentences)
                    self.labels.extend([row["Label"]]*len(augmented_sentences))
                else:
                    all_texts.append(text)
                    self.labels.append(row["Label"])
        self.encodings = self.tokenizer(all_texts, truncation=True, padding="max_length", max_length=self.max_length)
        classes = list(set(self.labels))
        classes.sort()
        self.class2index = {classes[i]:i for i in range(len(classes))}
        self.index2class = {i:classes[i] for i in range(len(classes))}
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        

                    
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.class2index[self.labels[index]])
        return item

 
def main():
    dataset_path = "dataset"
    
    dataset_preparer = MaskedDatasetPreparer("dataset")
    dataset_preparer.prepare_dataset()
    
    trainset, validset = get_mlm_train_validation_dataset(os.path.join(dataset_path, "masked-lang", "working-dir"))
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=trainset.dataset.tokenizer, mlm=True, mlm_probability=0.15
    )
    
    
    

if __name__ == '__main__':
    main()
    
    