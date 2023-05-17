import time
import logging
import torch
import transformers
from typing import Optional, Literal
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
from bert_classifier.io import save_checkpoint
from .metrics import accuracy
from .loss import cross_entropy_loss_fn
from .utils import batch_to_device

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def get_optimizer(param_optimizer,
                  optimizer_class = torch.optim.AdamW,
                  optimizer_params: dict = {'lr': 2e-5},
                  weight_decay: float = 0.01
                  ):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_params = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimizer_class(optimizer_grouped_params, **optimizer_params)
    return optimizer


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    '''
    Returns the correct learning rate scheduler.
    Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    '''
    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError(f'Unkown scheduler {scheduler}')


def validate(model: torch.nn.Module,
             test_dataloader: DataLoader,
             device: torch.device = torch.device('cpu'),
             multi_label: bool = False,
             num_labels: Optional[int] = None,
             metrics = accuracy,
             average: Literal['micro', 'macro', 'weighted', 'none'] = 'macro'
             ):
    '''
    Validate module
    '''
    model.eval()
    val_targets = torch.tensor(()).to(device)
    val_outputs = torch.tensor(()).to(device)
    val_loss = torch.tensor(()).to(device)
    with torch.no_grad():
        for features, labels in test_dataloader:
            labels = labels.to(device)
            features = batch_to_device(features, device)
            outputs = model(**features)
            output = outputs[0]
            val_outputs = torch.cat((val_outputs, output), 0)
            val_targets = torch.cat((val_targets, labels), 0)
            val_loss = torch.cat((val_loss, cross_entropy_loss_fn(output, labels, multi_label).reshape(1)), 0)
    avg_val_loss = val_loss.mean()
    avg_val_acc = metrics(val_outputs, val_targets, True, num_labels, device, average)
    return avg_val_loss, avg_val_acc


def custom_trainer(model: torch.nn.Module,
                   train_dataloader: DataLoader,
                   test_dataloader: DataLoader,
                   epochs: int = 1,
                   optimizer_class = torch.optim.AdamW,
                   optimizer_params: dict = {'lr': 2e-5},
                   weight_decay: float = 0.01,
                   scheduler: str = 'WarmupLinear',
                   warmup_steps: int = 10000,
                   multi_label: bool = False,
                   num_labels: Optional[int] = None,
                   tune_base_model: bool = True,
                   metrics = accuracy,
                   device: torch.device = torch.device('cpu')
                   ):
    '''
    Custom training module
    '''
    if not tune_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False
    
    param_optimizer = list(model.named_parameters())
    optimizer_obj = get_optimizer(param_optimizer, optimizer_class, optimizer_params, weight_decay)
    scheduler_obj = get_scheduler(optimizer_obj, scheduler, warmup_steps, len(train_dataloader)*epochs)

    model.train()
    for epoch in trange(epochs):
        model.zero_grad(set_to_none=True)
        logger.info(f'====== Epoch {epoch+1} / {epochs} ======')
        logger.info(f'Total steps: {len(train_dataloader)} || Training in progress...')
        t0 = time.time()
        total_train_loss = total_train_acc = 0
        for _, (features, labels) in enumerate(train_dataloader):
            labels = labels.to(device)
            features = batch_to_device(features, device)
            outputs = model(**features)
            output = outputs[0]
            loss = cross_entropy_loss_fn(output, labels, multi_label)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_obj.step()
            scheduler_obj.step()
            optimizer_obj.zero_grad()

            total_train_loss += loss.item()
            total_train_acc += metrics(output, labels, multi_label, num_labels, device)

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)
        logger.info(f'''
            Training Info: 
                Total Time: {time.time() - t0} sec || 
                Average Loss: {avg_train_loss} ||
                Average Accuracy: {avg_train_acc}
            ''')
        logger.info('Evaluation in progress...')
        avg_val_loss, avg_val_acc = validate(model, test_dataloader, device, multi_label, num_labels, average=None)
        logger.info(f'''
            Validation Info:
                Time: {time.time() - t0} ||
                Loss: {avg_val_loss} ||
                Accuracy: {avg_val_acc}
        ''')
        save_checkpoint(model, optimizer_obj, epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc)
        
