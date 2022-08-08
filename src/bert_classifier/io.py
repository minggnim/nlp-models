import json
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    PATH = Path('../checkpoints') / ('chkpt' + epoch + '.pt')
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


def save_model(model):
    torch.save(model.state_dict(), FINETUNED_MODEL_STATE)
    torch.save(optimizer.state_dict(), FINETUNED_OPT_STATE)
    loaded_model = BertClass(pretrained_model, model.l3.out_features)
    loaded_model.load_state_dict(torch.load(FINETUNED_MODEL_STATE))
    torch.save(loaded_model, FINETUNED_MODEL)


def load_model(model_dir):
    return torch.load_model(model_dir)


def save_label_dict(label_dict, dir='../models/bert/fine-tuned/labels-dict.json'):
    with open(dir, 'w') as f:
        json.dump(label_dict, f)