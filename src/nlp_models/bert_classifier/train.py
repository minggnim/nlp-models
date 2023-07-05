'''
Custom training module
'''
import time
import torch
import numpy as np
from tqdm import tqdm
from ..base.io import save_checkpoint
from ..base.utils import batch_to_device
from ..base.metrics import accuracy_metrics


def loss_fn_multiple_labels(outputs, targets):
    '''
    binary cross entropy loss for multi-labels
    '''
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def loss_fn(outputs, targets):
    '''
    cross entropy loss function
    '''
    return torch.nn.CrossEntropyLoss()(outputs, targets)


def optimizer_obj(model, learning_rate):
    opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate
    )
    return opt


def custom_trainer(model, optimizer, train_dataloader, test_dataloader, epochs=5, device='cpu'):
    '''
    custom training module
    '''
    for epoch in range(epochs):
        model.train()
        print("")
        print(f'======== Epoch {epoch+1} / {epochs} ========')
        print(f'Total steps: {len(train_dataloader)} || Training in progress...')
        t0 = time.time()
        total_train_loss, total_train_accuracy = 0, 0
        for _, (features, label) in tqdm(enumerate(train_dataloader)):
            # import pdb; pdb.set_trace();
            model.zero_grad(set_to_none=True)
            features = batch_to_device(features, device)
            # ids = batch['input_ids'].squeeze(1).to(device)
            # mask = batch['attention_mask'].squeeze(1).to(device)
            # type_ids = batch['token_type_ids'].squeeze(1).to(device)
            label = label.to(device)
            output = model(**features)

            loss = loss_fn(output, label)
            total_train_loss += loss.item()

            total_train_accuracy += accuracy_metrics(output.detach().cpu().numpy(), label.detach().cpu().numpy())

            # if step % 5000 == 0:
            #     print(f'Epoch: {epoch}, Loss: {loss.item()}')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print(f'''
            Training time: {time.time() - t0} seconds ||
            Training loss: {avg_train_loss} ||
            Training accuracy: {avg_train_accuracy}
            ''')
        print('Evaluation in progress...')
        val_outputs, val_targets, val_loss = validate(model, test_dataloader, device)
        avg_val_loss = np.array(val_loss).mean()
        avg_val_accuracy = accuracy_metrics(val_outputs, val_targets)
        print(f'''
            Validation time: {time.time() - t0} seconds
            Validation loss: {avg_val_loss} ||
            Validation accuracy: {avg_val_accuracy}
            ''')
        save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy)


def validate(model, test_dataloader, device='cpu'):
    '''
    produce validation results
    '''
    model.eval()
    val_targets, val_outputs, val_loss = [], [], []
    with torch.no_grad():
        for _, (features, label) in enumerate(test_dataloader):
            features = batch_to_device(features, device)
            label = label.to(device)
            outputs = model(**features)
            val_targets.extend(label.detach().cpu().numpy())
            val_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            val_loss.extend([loss_fn(outputs, label).detach().cpu().numpy()])
    return val_outputs, val_targets, val_loss
