import torch
from torch.utils.data import Dataset
from .bert import bert_encoder


class CustomDataset(Dataset):
    '''
    Class to construct torch Dataset from dataframe
    '''
    def __init__(self, dataframe, DATA_FIELD, LABEL_FIELD, tokenizer, max_len):
        self.max_len = max_len
        self.data = dataframe
        self.tokenizer = tokenizer
        self.content = self.data[DATA_FIELD]
        self.label = self.data[LABEL_FIELD]

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        content = str(self.content[index])
        content = " ".join(content.split())
        ids, mask, token_type_ids = bert_encoder(content, self.tokenizer, self.max_len)

        return {
            'ids': ids,
            'mask': mask,
            'type_ids': token_type_ids,
            'label': torch.tensor(self.label[index], dtype=torch.long),
            # 'multi_label': torch.tensor(self.label[index], dtype=torch.float)
        }
