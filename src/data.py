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
class SanitizedDataset(Dataset):
    def __init__(self, dataset_path, is_train, return_ids, split_document, 
                 min_word_frequency = 5, max_word_frequency = -1, 
                 merge_short_sentence = True, min_sentence_length = 10, vocabulary = None):
        super().__init__()
        self.is_train = is_train
        self.return_ids = return_ids
        self.split_document = split_document
        self.min_word_frequency = min_word_frequency
        self.max_word_frequency = max_word_frequency
        self.merge_short_sentence = merge_short_sentence
        self.min_sentence_length = min_sentence_length
        
        
        
        self.build_dataset(dataset_path, vocabulary)
    def build_dataset(self, dataset_path, vocabulary):
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
            if self.split_document:
                texts = self.split_text(text)
            else:
                texts = [self.santize_sentence(text)]
            self.texts.extend(texts)
            if self.return_ids:
                self.ids.extend([id_] * len(texts))
            if self.is_train:
                self.labels.extend([label] * len(texts))
        if vocabulary is None:
            self.vocabulary = set()
            for word, count in self.word_counts.items():
                if count >= self.min_word_frequency:
                    if self.max_word_frequency == -1 or count <= self.max_word_frequency:
                        self.vocabulary.add(word)
        else:
            self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.max_sentence_length = 0
        for i in range(len(self.texts)):
            words = [word for word in self.texts[i].split() if word in self.vocabulary]
            self.max_sentence_length = max(self.max_sentence_length, len(words))
            self.texts[i] = " ".join(words)
        
    def santize_sentence(self, sentence):
        output = []
        for word in re.split("[^A-Za-z]", sentence):
            word = word.lower()
            self.word_counts[word]+=1
            output.append(word)
        return " ".join(output)
    def split_text(self, text):        
        
        output = []
        current_sentence = None
    
        for sent in re.split("[\.?!]", text):
            sent = self.santize_sentence(sent)
            if current_sentence is None:
                current_sentence = sent 
            elif len(current_sentence.split()) >= self.min_sentence_length:
                output.append(current_sentence)
                current_sentence = sent 
            else:
                current_sentence += " "+ sent
        if current_sentence is not None:
            if len(current_sentence.split()) >= self.min_sentence_length:
                output.append(current_sentence)
            else:
                last_sent = output[-1]
                output[-1] = last_sent +" " + current_sentence
        return output
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
    def __init__(self, dataset_path, split_document = True, 
                 min_word_frequency = 5, max_word_frequency = -1, 
                 merge_short_sentence = True, min_sentence_length = 10, working_folder = None):
        super().__init__()
        self.split_document = split_document
        self.min_word_frequency = min_word_frequency
        self.max_word_frequency = max_word_frequency
        self.merge_short_sentence = merge_short_sentence
        self.min_sentence_length = min_sentence_length
        if not os.path.exists(os.path.join(dataset_path, "masked-lang")):
            os.mkdir(os.path.join(dataset_path, "masked-lang"))
        self.working_dir = os.path.join(dataset_path, "masked-lang", "working-dir") if working_folder is None else working_folder
        
        self.train_path = os.path.join(dataset_path, "Train.csv")
        self.test_path = os.path.join(dataset_path, "Test.csv")
        
    def prepare_dataset(self):
        sentences = []
        trainset = SanitizedDataset(self.train_path, is_train=False, return_ids=False, 
                                    split_document=self.split_document, min_word_frequency= self.min_word_frequency,
                                    max_word_frequency= self.max_word_frequency, merge_short_sentence= self.merge_short_sentence, 
                                    min_sentence_length= self.min_sentence_length)
        testset = SanitizedDataset(self.test_path, is_train=False, return_ids=False, 
                                    split_document=self.split_document, min_word_frequency= self.min_word_frequency,
                                    max_word_frequency= self.max_word_frequency, merge_short_sentence= self.merge_short_sentence,
                                    min_sentence_length= self.min_sentence_length,  vocabulary=trainset.vocabulary)
        
        for i in range(len(trainset)):
            sentences.append(trainset[i])
        
        for i in range(len(testset)):
            sentences.append(testset[0])
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)
        if not os.path.exists(os.path.join(self.working_dir, "tokenizer")):
            os.mkdir(os.path.join(self.working_dir, "tokenizer"))
        
        with open(os.path.join(self.working_dir, "sentences.txt"), "w") as outfile:
            for sent in sentences:
                outfile.write("{}\n".format(sent))
        paths = [os.path.join(self.working_dir, "sentences.txt")]

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.enable_padding(pad_token="<pad>", length=64)
        tokenizer.enable_truncation(max_length=64)
        tokenizer.train(files=paths, vocab_size=trainset.vocabulary_size, min_frequency=5, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
                
        tokenizer_save_path = os.path.join(self.working_dir, "tokenizer") 
        tokenizer.save_model(tokenizer_save_path)
        self.vocabulary_size = trainset.vocabulary_size
        
class MaskedDataset(Dataset):
    def __init__(self, working_dir, tokenizer):
        self.working_dir = working_dir
        self.tokenizer = tokenizer
        src_file = Path(os.path.join(self.working_dir, "sentences.txt"))
        lines = src_file.read_text(encoding="utf-8").splitlines()
        self.inputs = [self.tokenizer.encode(x, padding="max_length", truncation=True ) for x in lines]
        
        
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
def get_train_validation_dataset(working_dir, train_size = 0.8):
    tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(working_dir, "tokenizer"), max_len=64)
        
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

        
    
        
        
def main():
    dataset_path = "dataset"
    
    dataset_preparer = MaskedDatasetPreparer("dataset", max_word_frequency=200)
    dataset_preparer.prepare_dataset()
    
    trainset, validset = get_train_validation_dataset(os.path.join(dataset_path, "masked-lang", "working-dir"))
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=trainset.dataset.tokenizer, mlm=True, mlm_probability=0.15
    )
    
    
    

if __name__ == '__main__':
    main()
    
    