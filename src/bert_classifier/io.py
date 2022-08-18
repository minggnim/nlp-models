import json
from pathlib import Path
import torch
from transformers import BertTokenizer, BertModel


MODEL_NAME = 'bert-base-uncased'
MODEL_DIR = Path('../models/bert')
PRETRAINED_TOKENIZER = MODEL_DIR / 'pretrained/tokenizer-uncased'
PRETRAINED_MODEL = MODEL_DIR / 'pretrained/bert-base-uncased'
FINETUNED_MODEL = MODEL_DIR / 'fine-tuned/fine-tuned-uncased'
FINETUNED_MODEL_STATE = MODEL_DIR / 'fine-tuned/model-state-dict'
FINETUNED_OPT_STATE = MODEL_DIR / 'fine-tuned/opt-state-dict'
CHECKPOINT_DIR = MODEL_DIR / 'checkpoint'


def get_pretrained_tokenizer(MODEL_NAME=MODEL_NAME, PRETRAINED_TOKENIZER=PRETRAINED_TOKENIZER):
    if PRETRAINED_TOKENIZER.exists():
        pretrained_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    else:
        pretrained_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        pretrained_tokenizer.save_pretrained(PRETRAINED_TOKENIZER)
    return pretrained_tokenizer


def get_pretrained_model(MODEL_NAME=MODEL_NAME, PRETRAINED_MODEL=PRETRAINED_MODEL):
    if PRETRAINED_MODEL.exists():
        pretrained_model = BertModel.from_pretrained(PRETRAINED_MODEL)
    else:
        pretrained_model = BertModel.from_pretrained(MODEL_NAME)
        pretrained_model.save_pretrained(PRETRAINED_MODEL)
    return pretrained_model


def save_checkpoint(model, optimizer, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    PATH = CHECKPOINT_DIR / ('chkpt' + epoch + '.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_accuracy,
        'val_acc': val_accuracy,
    }, PATH)


def load_checkpoint(model, optimizer, chkpoint_dir):
    checkpoint = torch.load(chkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    train_acc = checkpoint['train_acc']
    val_acc = checkpoint['val_acc']
    return model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc


def save_model(model, optimizer, pretrained_model):
    torch.save(model.state_dict(), FINETUNED_MODEL_STATE)
    torch.save(optimizer.state_dict(), FINETUNED_OPT_STATE)
    # loaded_model = BertClass(pretrained_model, model.l3.out_features)
    # loaded_model.load_state_dict(torch.load(FINETUNED_MODEL_STATE))
    torch.save(model, FINETUNED_MODEL)


def load_model(model_dir=FINETUNED_MODEL, device='cpu'):
    return torch.load(model_dir, map_location=torch.device(device))


# def load_model_safe(model_dir, num_label):
#     pretrained_model = get_pretrained_model(PRETRAINED_MODEL, MODEL_NAME)
#     model = BertClass(pretrained_model, num_label)
#     model.load_state_dict(torch.load(FINETUNED_MODEL_STATE))
#     return model


def save_label_dict(label_dict, dir='../models/bert/fine-tuned/labels-dict.json'):
    with open(dir, 'w') as f:
        json.dump(label_dict, f)


def load_label_dict(label_dir):
    with open(dir, 'r') as f:
        return json.load(f)
