'''
data modules
'''
import torch
from torch.utils.data import Dataset
from .bert import bert_encoder


class CustomDataset(Dataset):
    '''
    Class to construct torch Dataset from dataframe
    '''
    def __init__(self, dataframe, data_field, label_field, tokenizer, max_len):
        self.max_len = max_len
        self.data = dataframe
        self.tokenizer = tokenizer
        self.content = self.data[data_field]
        self.label = self.data[label_field]

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        content = str(self.content[index])
        content = " ".join(content.split())
        encoded_content = bert_encoder(content, self.tokenizer, self.max_len)

        return {
            'input_ids': encoded_content['input_ids'],
            'attention_mask': encoded_content['attention_mask'],
            'token_type_ids': encoded_content['token_type_ids'],
            'label': torch.tensor(self.label[index], dtype=torch.long),
            # 'multi_label': torch.tensor(self.label[index], dtype=torch.float)
        }


def create_label_dict(dataframe, label_col):
    labels = dataframe.groupby(label_col).size().sort_values(ascending=False).index.tolist()
    label_dict = dict([(d, i) for i, d in enumerate(labels)])
    return label_dict


def label2id(dataframe, label_col, label_dict, multi_class=False):
    if multi_class:
        dataframe['label'] = dataframe[label_col].apply(lambda c: [int(c==l) for l in label_dict.keys()])
    else:
        dataframe['label'] = dataframe[label_col].apply(lambda c: label_dict[c])
    return dataframe
