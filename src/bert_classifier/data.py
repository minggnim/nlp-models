'''
data modules
'''
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    '''
    Class to construct torch Dataset from dataframe
    '''
    def __init__(self,
                 dataframe,
                 data_field,
                 label_field,
                 tokenizer,
                 max_len
                 ):
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
        encoded_content = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        features = encoded_content
        labels = torch.tensor(self.label[index], dtype=torch.float)
        # for int ids: labels = torch.tensor(self.label[index], dtype=torch.long)

        return features, labels


def create_label_dict(dataframe, label_col):
    labels = dataframe.groupby(label_col).size().sort_values(ascending=False).index.tolist()
    label_dict = dict([(d, i) for i, d in enumerate(labels)])
    return label_dict


def label_to_id_list(dataframe, label_col, label_dict):
    dataframe[label_col] = dataframe[label_col].apply(lambda c: [int(k in c) for k in label_dict.keys()])
    return dataframe


def label_to_id_int(dataframe, label_col, label_dict):
    dataframe[label_col] = dataframe[label_col].apply(lambda c: label_dict[c])
    return dataframe
