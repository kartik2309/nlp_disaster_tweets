import os
import shutil

import torch
from torch.utils.data import DataLoader


class Train:
    def __init__(
            self,
            is_checkpoint=False,
            is_pretrained=False,
            save_path=None,
            model_name=None,
            model: torch.nn.Module = None,
            optimizer=None,
            scheduler=None,
            loss_fn=None,
            lr=None,
            **kwargs
    ):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.is_checkpoint = is_checkpoint
        self.model_name = model_name
        self.save_path = save_path
        self.model = model.to(device=device)
        self.optimizer = optimizer(self.model.parameters(), float(lr))
        self.is_pretrained = is_pretrained
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **kwargs)
        else:
            self.scheduler = None
        self.loss_fn = loss_fn()
        self.previous_accuracy = 0
        if is_checkpoint:
            if save_path is None:
                raise ValueError("Argument save_path cannot of None type when a checkpoint is to be loaded.")
            self.completed_epochs, last_lost, self.previous_accuracy = self.load_checkpoint()

            print("Loaded checkpoint!")
            print("Completed epochs:", self.completed_epochs, "Previous Loss:",
                  last_lost, "Previous Accuracy:", self.previous_accuracy)

        if is_pretrained:
            _, _, _ = self.load_checkpoint()
            print("Loaded Pretrained Model")

        return

    def train(self, epochs, train_loader: DataLoader, val_loader: DataLoader = None):
        losses = []
        accuracies = []

        if self.is_checkpoint:
            epochs = epochs - self.completed_epochs

        for epoch in range(epochs):

            losses.clear()
            accuracies.clear()

            if not self.model.training:
                self.model.train()

            if self.is_checkpoint:
                print("Epoch:", self.completed_epochs + epoch + 1)
            else:
                print("Epoch:", epoch + 1)
            for batch in train_loader:
                input_ids = batch['input_ids']
                labels = batch['labels']
                if 'attention_mask' in batch.keys():
                    attention_mask = batch['attention_mask']
                    pred = self.model(input_ids, attention_mask=attention_mask)
                else:
                    pred = self.model(input_ids)

                loss = self.loss_fn(pred, labels)
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accuracies.append(self.accuracy(pred, labels).data)

            if self.scheduler is not None:
                self.scheduler.step()
                print("Adjusted last lr was:", self.scheduler.get_last_lr())

            avg_loss = self.compute_avg(losses)
            avg_accuracy = self.compute_avg(accuracies).item()

            print("Average training loss:", avg_loss,
                  ", Average training accuracy:", avg_accuracy)

            losses.clear()
            accuracies.clear()

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids']
                        labels = batch['labels']
                        if 'attention_mask' in batch.keys():
                            attention_mask = batch['attention_mask']
                            pred = self.model(input_ids, attention_mask=attention_mask)
                        else:
                            pred = self.model(input_ids)
                        loss = self.loss_fn(pred, labels)

                        losses.append(loss.item())
                        accuracies.append(self.accuracy(pred, labels).data)

                avg_loss = self.compute_avg(losses)
                avg_accuracy = self.compute_avg(accuracies).item()

                print("Average val loss:", avg_loss,
                      ", Average val accuracy:", avg_accuracy)

            if avg_accuracy > self.previous_accuracy:
                print("Saving checkpoint!")
                if self.save_path is not None:
                    if self.is_checkpoint:
                        self.save_checkpoint(self.completed_epochs + epoch + 1, avg_loss, avg_accuracy)
                    else:
                        self.save_checkpoint(epoch + 1, avg_loss, avg_accuracy)

                self.previous_accuracy = avg_accuracy

    def compute_avg(self, quantity: list):
        return sum(quantity) / len(quantity)

    def accuracy(self, pred, labels):
        with torch.no_grad():
            classes = torch.argmax(pred, dim=-1)
            return torch.mean((classes == labels).float())

    def save_checkpoint(self, epoch, loss, accuracy):

        model_path = os.path.join(self.save_path, self.model_name)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if self.scheduler is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy,
                'scheduler': self.scheduler.state_dict()
            },
                model_path + '/model.pth')

            shutil.copyfile('config/model_config.yaml', model_path + '/model_config.yaml')

            return

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        },
            model_path + '/model.pth')

        shutil.copyfile('config/model_config.yaml', model_path + '/model_config.yaml')

        return

    def load_checkpoint(self):
        model_path = os.path.join(self.save_path, self.model_name)
        checkpoint = torch.load(model_path + '/model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        return epoch, loss, accuracy
