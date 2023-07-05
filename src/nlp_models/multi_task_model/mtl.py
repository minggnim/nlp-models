import torch
import transformers
import numpy as np
from ..base.layers import Dropout, Dense
from ..base.utils import get_embedding_group


class AutoModelForMTL(torch.nn.Module):
    '''
    Formation of MTL model, sharing a same base model
    '''
    def __init__(self, base_model_name, num_labels, no_normalize=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model = transformers.AutoModel.from_pretrained(base_model_name)
        self.dropout_layer = Dropout()
        self.dense_layer = Dense(self.base_model.config.hidden_size, num_labels, activation_function=torch.nn.Sigmoid())
        self.normalize = not no_normalize

    def forward(self, **kwargs):
        model_output = self.base_model(**kwargs)
        sentence_embedding = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
        cls_embedding = self.cls_pooling(model_output)
        dropout_output = self.dropout_layer(cls_embedding)
        pooled_output = self.dense_layer(dropout_output)
        return pooled_output, sentence_embedding

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, model_output):
        return model_output[0][:, 0]

    @staticmethod
    def save_model(model, model_dir):
        torch.save(model, model_dir)

    @staticmethod
    def load_model(model_dir, device=torch.device('cpu')):
        return torch.load(model_dir, device)


class MTLInference:
    '''
    class for MTL model infering
    '''
    def __init__(self, tokenizer_card, model_card, num_labels=None, pretrained_model=True, device=torch.device('cpu')) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_card)
        if pretrained_model:
            self.mtl_model = AutoModelForMTL(model_card, num_labels)
        else:
            self.mtl_model = AutoModelForMTL.load_model(model_card, device)
        self.mtl_model.eval()

    def encode(self, inputs):
        return self.tokenizer(inputs, return_tensors='pt', padding=True)

    def predict(self, query):
        with torch.no_grad():
            encoded = self.encode(query)
            outputs = self.mtl_model(**encoded)
        pred_label = outputs[0]
        query_embedding = outputs[1]
        return pred_label, query_embedding

    def pred_raw(self, query):
        return self.predict(query)[0].squeeze().tolist()

    @staticmethod
    def pred_index(raw_pred):
        pred = np.flatnonzero(np.array(raw_pred).__gt__(0.5)).tolist()
        if isinstance(pred, int):
            return [pred]
        return pred

    @staticmethod
    def get_label(pred, label_dict):
        return [label_dict[p] for p in pred]

    def predict_label(self, query):
        p_raw = self.pred_raw(query)
        return self.pred_index(p_raw)

    def optimal_answer(self, query, cat_lookup, corpus, top_k=1):
        pred_label, query_embedding = self.predict(query)
        corpus_embeddings, question_corpus, answer_corpus = get_embedding_group(pred_label, cat_lookup, corpus)
        sim_scores = torch.mm(query_embedding, corpus_embeddings.transpose(0, 1).cpu().tolist())
        top_idx = np.argpartition(sim_scores, range(-top_k, 0))[-top_k:][::-1]
        score = list(sim_scores[top_idx])
        question_matched = [question_corpus[i] for i in top_idx]
        answer_matched = [answer_corpus[i] for i in top_idx]
        return dict(
            question_matched=question_matched,
            answer_matched=answer_matched,
            score=score
        )
