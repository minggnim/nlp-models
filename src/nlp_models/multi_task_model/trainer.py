import gc
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
from tqdm.notebook import tqdm
import wandb
import torch
import transformers
from torch.utils.data import DataLoader
from ..base.metrics import accuracy
from ..base.utils import batch_to_device
from ..base.loss import cross_entropy_loss_fn


@dataclass
class Configs:
    epochs: int = 1
    optimizer_class = torch.optim.AdamW
    optimizer_params: Dict[str, float] = field(default_factory=lambda: ({"lr": 2e-5}))
    weight_decay: float = 0.01
    scheduler: str = 'WarmupLinear'
    warmup_steps: int = 10000
    num_labels: Optional[int] = None
    tune_base_model: bool = True


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 configs: Configs,
                 metrics=accuracy,
                 device: torch.device = torch.device('cpu'),
                 chkpt_dir=Path('../chkpt')
                 ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.configs = configs
        self.metrics = metrics
        self.device = device
        self.chkpt_dir = chkpt_dir
        self.set_params()
        self.train_logs = []
        self.val_logs = []
    
    def set_params(self):
        '''
        set parameters to optimize and schedule
        '''
        if not self.configs.tune_base_model:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        self.optimizer = self.get_optimizer(
            list(self.model.named_parameters()),
            self.configs.optimizer_class,
            self.configs.optimizer_params,
            self.configs.weight_decay)
        
        self.scheduler = self.get_scheduler(
            self.optimizer,
            self.configs.scheduler,
            self.configs.warmup_steps,
            len(self.train_dataloader) * self.configs.epochs)
    
    def train(self):
        epochs = tqdm(range(1, self.configs.epochs + 1), leave=True, desc="Training...")
        for epoch in epochs:
            self.model.train()
            epochs.set_description(f"EPOCH {epoch} / {self.configs.epochs} | training...")
            self.train_one_epoch(epoch)
            self.clear()

            self.model.eval()
            epochs.set_description(f"EPOCH {epoch} / {self.configs.epochs} | validating...")
            self.validate_one_epoch(epoch)
            self.clear()

            self.print_per_epoch(epoch)
            self.save_checkpoint(epoch)

    def continue_training(self, chkpt_file):
        '''
        continue training from checkpoint
        '''
        self.model, self.optimizer, _ = self.load_checkpoint(chkpt_file, self.model, self.optimizer)
        self.schedule_cold_start()
        self.train()

    def train_one_epoch(self, epoch):
        batches = tqdm(self.train_dataloader, total=len(self.train_dataloader))
        total_train_loss = total_train_acc = 0
        for features, labels in batches:
            labels = labels.to(self.device)
            features = batch_to_device(features, self.device)
            outputs = self.model(**features)
            loss = cross_entropy_loss_fn(outputs[0], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_train_loss += loss.item()
            total_train_acc += self.metrics(
                outputs[0],
                labels,
                self.configs.num_labels,
                self.device
            ).item()
            
            batches.set_description(f"Train Loss Step: {loss.item():.2f}")
        
        self.logger(
            epoch,
            total_train_acc / len(self.train_dataloader),
            total_train_loss / len(self.train_dataloader),
            'train')
        wandb.log({
            'train loss': total_train_acc / len(self.train_dataloader),
            'train acc': total_train_loss / len(self.train_dataloader),
        })

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        '''
        Validate module
        '''
        self.model.eval()
        val_targets = torch.tensor(()).to(self.device)
        val_outputs = torch.tensor(()).to(self.device)
        val_loss = torch.tensor(()).to(self.device)

        batches = tqdm(self.test_dataloader, total=len(self.test_dataloader))
        for features, labels in batches:
            labels = labels.to(self.device)
            features = batch_to_device(features, self.device)
            outputs = self.model(**features)
            val_outputs = torch.cat((val_outputs, outputs[0]), 0)
            val_targets = torch.cat((val_targets, labels), 0)
            loss = cross_entropy_loss_fn(outputs[0], labels).reshape(1)
            val_loss = torch.cat((val_loss, loss), 0)
            batches.set_description(f"Validation Loss Step: {loss.item():.2f}")
        
        avg_val_loss = val_loss.mean().item()
        avg_val_acc = self.metrics(
            val_outputs,
            val_targets,
            self.configs.num_labels,
            self.device
        ).item()
        
        self.logger(epoch, avg_val_acc, avg_val_loss, 'test')
        wandb.log({
            'val loss': avg_val_loss,
            'val acc': avg_val_acc
        })

    @staticmethod
    def get_optimizer(param_optimizer,
                      optimizer_class=torch.optim.AdamW,
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

    @staticmethod
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

    @staticmethod
    def load_checkpoint(chkpt_dir, model, optimizer=None, device=torch.device('cpu')):
        chkpt = torch.load(chkpt_dir, device)
        model.load_state_dict(chkpt['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        return model, optimizer, chkpt
    
    def save_checkpoint(self, epoch):
        if not self.chkpt_dir.exists():
            self.chkpt_dir.mkdir(parents=True)
        epoch -= 1
        chkpt_path = self.chkpt_dir / ('chkpt' + str(epoch) + '.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_logs[epoch][f'epoch_{epoch}']['loss'],
            'val_loss': self.val_logs[epoch][f'epoch_{epoch}']['loss'],
            'train_acc': self.train_logs[epoch][f'epoch_{epoch}']['accuracy'],
            'val_acc': self.val_logs[epoch][f'epoch_{epoch}']['accuracy'],
        }, chkpt_path)
        
    def schedule_cold_start(self):
        self.scheduler = self.get_scheduler(
            self.optimizer,
            self.configs.scheduler,
            0,
            len(self.train_dataloader) * self.configs.epochs)

    def print_per_epoch(self, epoch):
        print(f"\n\n{'-'*30}EPOCH {epoch}/{self.configs.epochs}{'-'*30}")
        epoch -= 1
        train_loss = self.train_logs[epoch][f'epoch_{epoch}']['loss']
        train_acc = self.train_logs[epoch][f'epoch_{epoch}']['accuracy']
        val_loss = self.val_logs[epoch][f'epoch_{epoch}']['loss']
        val_acc = self.val_logs[epoch][f'epoch_{epoch}']['accuracy']
        print(f"Train -> LOSS: {train_loss} | ACCURACY: {train_acc}")
        print(f"Validation -> LOSS: {val_loss} | ACCURACY: {val_acc}\n\n\n")

    def logger(self, epoch, metrics, loss, mode):
        log = {
            f'epoch_{epoch-1}': {
                'loss': loss,
                'accuracy': metrics
            }
        }
        if mode == 'train':
            self.train_logs.append(log)
        else:
            self.val_logs.append(log)

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()
