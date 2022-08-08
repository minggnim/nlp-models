import torch
from transformers import BertTokenizer, BertModel


class BertClass(torch.nn.Module):
    def __init__(self, pretrained_model, num_label):
        super().__init__()
        self.l1 = pretrained_model
        self.l2 = torch.nn.Dropout(self.l1.config.hidden_dropout_prob)
        self.l3 = torch.nn.Linear(self.l1.config.hidden_size, num_label)
        
    def forward(self, ids, mask, type_ids):
        # import pdb; pdb.set_trace();
        last_hidden_state = self.l1(ids, mask, type_ids).last_hidden_state
        cls_representation = last_hidden_state[:, 0, :]
        dropout_output = self.l2(cls_representation)
        pooled_output = self.l3(dropout_output)
        return pooled_output


def bert_encoder(content, tokenizer, max_len):
    inputs = self.tokenizer.encode_plus(
        content,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=self.max_len,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    return ids, mask, token_type_ids


def get_pretrained_tokenizer(PRETRAINED_TOKENIZER, MODEL_NAME):
    if PRETRAINED_TOKENIZER.exists():
        pretrained_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    else:
        pretrained_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        pretrained_tokenizer.save_pretrained(PRETRAINED_TOKENIZER)
    return pretrained_tokenizer


def get_pretrained_model(PRETRAINED_MODEL, MODEL_NAME):
    if PRETRAINED_MODEL.exists():
        pretrained_model = BertModel.from_pretrained(PRETRAINED_MODEL)
    else:
        pretrained_model = BertModel.from_pretrained(MODEL_NAME)
        pretrained_model.save_pretrained(PRETRAINED_MODEL)
    return pretrained_model