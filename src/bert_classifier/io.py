'''
IO utils
'''
import json
from pathlib import Path
import torch
from transformers import BertTokenizer, BertModel


MODEL_NAME = 'bert-base-uncased'
MODEL_DIR = Path('../models/bert')
PRETRAINED_TOKENIZER = MODEL_DIR / 'pretrained/tokenizer-uncased'
PRETRAINED_MODEL = MODEL_DIR / 'pretrained/bert-base-uncased'
FINETUNED_DIR = MODEL_DIR / 'fine-tuned'
FINETUNED_MODEL = FINETUNED_DIR / 'fine-tuned-uncased'
FINETUNED_MODEL_STATE = FINETUNED_DIR / 'model-state-dict'
FINETUNED_OPT_STATE = FINETUNED_DIR / 'opt-state-dict'
CHECKPOINT_DIR = MODEL_DIR / 'checkpoint'
LABEL_DICT = MODEL_DIR / 'fine-tuned/labels-dict.json'


def get_pretrained_tokenizer(model_name=MODEL_NAME, pretrained_tokenizer_dir=PRETRAINED_TOKENIZER):
    '''
    get pretrained tokenizer
    '''
    if pretrained_tokenizer_dir.exists():
        pretrained_tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_dir)
    else:
        pretrained_tokenizer = BertTokenizer.from_pretrained(model_name)
        pretrained_tokenizer.save_pretrained(pretrained_tokenizer_dir)
    return pretrained_tokenizer


def get_pretrained_model(model_name=MODEL_NAME, pretrained_model_dir=PRETRAINED_MODEL):
    '''
    get pretrained model
    '''
    if pretrained_model_dir.exists():
        pretrained_model = BertModel.from_pretrained(pretrained_model_dir)
    else:
        pretrained_model = BertModel.from_pretrained(model_name)
        pretrained_model.save_pretrained(pretrained_model_dir)
    return pretrained_model


def save_checkpoint(model, optimizer, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    '''
    save checkpoint
    '''
    if not CHECKPOINT_DIR.exists():
        CHECKPOINT_DIR.mkdir(parents=True)
    chkpt_path = CHECKPOINT_DIR / ('chkpt' + str(epoch) + '.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_accuracy,
        'val_acc': val_accuracy,
    }, chkpt_path)


def load_checkpoint(model, optimizer, chkpoint_dir):
    '''
    load checkpoint
    '''
    checkpoint = torch.load(chkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    train_acc = checkpoint['train_acc']
    val_acc = checkpoint['val_acc']
    return model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc


def save_model(model, optimizer):
    '''
    save model para
    '''
    if not FINETUNED_DIR.exists():
        FINETUNED_DIR.mkdir(parents=True) 
    torch.save(model.state_dict(), FINETUNED_MODEL_STATE)
    torch.save(optimizer.state_dict(), FINETUNED_OPT_STATE)
    # loaded_model = BertClass(pretrained_model, model.l3.out_features)
    # loaded_model.load_state_dict(torch.load(FINETUNED_MODEL_STATE))
    torch.save(model, FINETUNED_MODEL)


def load_model(model_dir=FINETUNED_MODEL, device='cpu'):
    '''
    load the whole model
    '''
    return torch.load(model_dir, map_location=torch.device(device))


# def load_model_safe(model_dir, num_label):
#     pretrained_model = get_pretrained_model(PRETRAINED_MODEL, MODEL_NAME)
#     model = BertClass(pretrained_model, num_label)
#     model.load_state_dict(torch.load(FINETUNED_MODEL_STATE))
#     return model


def save_label_dict(label_dict, dict_file=LABEL_DICT):
    '''save label dictionary'''
    with open(dict_file, 'w') as file:
        json.dump(label_dict, file)


def load_label_dict(label_file=LABEL_DICT):
    '''load label dictionary'''
    with open(label_file, 'r') as file:
        return json.load(file)
