'''
bert inheritance class and functions
'''
import torch
from .io import get_pretrained_model


class BertClass(torch.nn.Module):
    '''
    Custom class for bert classification model
    '''
    def __init__(self, num_label, base_model=None):
        super().__init__()
        self.l1 = base_model if base_model else get_pretrained_model()
        self.l2 = torch.nn.Dropout(self.l1.config.hidden_dropout_prob)
        self.l3 = torch.nn.Linear(self.l1.config.hidden_size, num_label)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # import pdb; pdb.set_trace();
        last_hidden_state = self.l1(input_ids, attention_mask, token_type_ids).last_hidden_state
        cls_representation = last_hidden_state[:, 0, :]
        dropout_output = self.l2(cls_representation)
        pooled_output = self.l3(dropout_output)
        return pooled_output


def bert_encoder(content, tokenizer, max_len):
    '''
    encoder function with preset params
    '''
    inputs = tokenizer.encode_plus(
        content,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    return inputs
