import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import argparse

from Network.LSTM import LSTM_MultiTask
from dataset import HDF5Dataset

# =========================================
# 👉 Command Line Arguments
# =========================================
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--num_classes1', default=19, type=int)
parser.add_argument('--num_classes2', default=6, type=int)
parser.add_argument('--patience', default=5, type=int, help='Early stopping patience')
parser.add_argument('--log_dir', default='logs_multitask_cls', type=str)
parser.add_argument('--save_dir', default='checkpoints_multitask_cls', type=str)
parser.add_argument('--seed', default=42, type=int)
opt = parser.parse_args()


# =========================================
# 👉 Setup
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

writer = SummaryWriter(opt.log_dir)


# =========================================
# 👉 AverageMeter (tracking loss)
# =========================================
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =========================================
# 👉 EarlyStopping Utility
# =========================================
class EarlyStopping:
    """
    Stops training if validation loss doesn't improve for <patience> epochs.
    """
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_epoch = 0
        self.verbose = verbose

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# =========================================
# 👉 Train & Eval Functions
# =========================================
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    losses = AverageMeter()

    for batch_idx, (x, y1, y2) in enumerate(loader):
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

        # pred1, pred2 = model(x)
        # loss1 = F.cross_entropy(pred1, y1)
        # loss2 = F.cross_entropy(pred2, y2)
        # loss = loss1 + loss2  # combine losses

        pred = model(x)
        loss = F.cross_entropy(pred, y2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.size(0))

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(loader)}] Loss: {losses.avg:.4f}")
            writer.add_scalar('Train/Loss', losses.avg, epoch * len(loader) + batch_idx)

    return losses.avg


@torch.no_grad()
def eval_epoch(model, loader, epoch):
    model.eval()
    losses = AverageMeter()
    # correct1, correct2 = 0, 0
    correct = 0
    total = 0

    for x, y1, y2 in loader:
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

        # pred1, pred2 = model(x)
        # loss1 = F.cross_entropy(pred1, y1)
        # loss2 = F.cross_entropy(pred2, y2)
        # loss = loss1 + loss2
        # losses.update(loss.item(), x.size(0))

        # _, p1 = pred1.max(1)
        # _, p2 = pred2.max(1)
        # total += y1.size(0)
        # correct1 += p1.eq(y1).sum().item()
        # correct2 += p2.eq(y2).sum().item()

        pred = model(x)
        loss = F.cross_entropy(pred, y2)
        losses.update(loss.item(), x.size(0))

        _, p1 = pred.max(1)
        total += y2.size(0)
        correct += p1.eq(y2).sum().item()

    # acc1 = correct1 / total
    # acc2 = correct2 / total

    # print(f"Validation -> Loss: {losses.avg:.4f}, Acc1: {acc1:.4f}, Acc2: {acc2:.4f}")

    acc = correct / total

    print(f"Validation -> Loss: {losses.avg:.4f}, Acc1: {acc:.4f}")

    # writer.add_scalar('Val/Loss', losses.avg, epoch)
    # writer.add_scalar('Val/Acc1', acc1, epoch)
    # writer.add_scalar('Val/Acc2', acc2, epoch)

    writer.add_scalar('Val/Loss', losses.avg, epoch)
    writer.add_scalar('Val/Acc', acc, epoch)

    # return losses.avg, acc1, acc2
    return losses.avg, acc


# =========================================
# 👉 Main Training Function (with Early Stop)
# =========================================
def train_model():
    print("Preparing data...")
    train_set = HDF5Dataset('train_data.h5')
    val_set = HDF5Dataset('test_data.h5')

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    model = LSTM_MultiTask(
        input_size=13,
        hidden_size=16,
        num_layers=2,
        num_classes1=opt.num_classes1,
        num_classes2=opt.num_classes2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    early_stopping = EarlyStopping(patience=opt.patience)

    best_loss = float('inf')

    for epoch in range(opt.epochs):
        print(f"\n=== Epoch [{epoch+1}/{opt.epochs}] ===")
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        val_loss, acc = eval_epoch(model, val_loader, epoch)
        scheduler.step()

        # Save model each epoch
        torch.save(model.state_dict(), os.path.join(opt.save_dir, f"model_epoch{epoch+1}.pth"))

        # Track best model + Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(opt.save_dir, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best model: {best_path}")

        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping at epoch {epoch+1}. Best epoch: {early_stopping.best_epoch+1}")
            break

    print(f"Training complete. Best val loss at epoch {early_stopping.best_epoch+1}")
    writer.close()


if __name__ == "__main__":
    train_model()