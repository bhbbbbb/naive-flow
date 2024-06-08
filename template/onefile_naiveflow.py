# pylint: disable=all
from torch.utils.data import DataLoader, RandomSampler, random_split

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import naive_flow as nf


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        IN_DIM = 28 * 28
        OUT_DIM = 10

        def build_block(in_dim: int, out_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
            )

        self.seq = nn.Sequential(
            build_block(IN_DIM, 64),
            build_block(64, 32),
            build_block(32, OUT_DIM),
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.flatten(x, 1)
        return torch.softmax(self.seq(x), 1)


DEVICE = 'cuda:0'


def main():
    ROOT = './log'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))]
    )
    train_set = torchvision.datasets.MNIST(
        root=ROOT, train=True, download=True, transform=transform
    )
    train_set, val_set = random_split(
        train_set, [0.85, 0.15], generator=torch.Generator().manual_seed(1)
    )
    test_set = torchvision.datasets.MNIST(
        root=ROOT, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        sampler=RandomSampler(train_set, num_samples=128),
        # shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=4, shuffle=False, num_workers=0,
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_set, batch_size=4, shuffle=False, num_workers=0,
        persistent_workers=False
    )

    model = MLP()
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=20,
        pct_start=0.1,
        steps_per_epoch=len(train_loader),
    )

    tracker_config = nf.tracker.TrackerConfig(
        epochs_per_checkpoint=0,
        enable_logging=True,
        save_n_best=0,
        early_stopping_rounds=5,
        save_end=False,
        comment="_NMIST",
    )

    tracker = nf.tracker.SimpleTracker(
        model,
        optimizer,
        **dict(tracker_config),
    )

    writer = SummaryWriter(
        log_dir=tracker.log_dir, purge_step=tracker.start_epoch
    )
    tracker.register_summary_writer(writer)
    writer.add_text('tracker_config', nf.strfconfig(tracker_config))
    tracker.register_scalar(
        'loss/val', scalar_type='loss', for_early_stopping=True
    )
    tracker.register_scalar('loss/*', scalar_type='loss')
    tracker.register_scalar('accuracy/*', scalar_type='accuracy')

    for epoch in tracker.range(20):

        train_loss, train_acc = train_epoch(
            model, optimizer, scheduler, train_loader
        )
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)

        val_loss, val_acc = eval_epoch(model, val_loader)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('accuracy/val', val_acc, epoch)

        test_loss, test_acc = eval_epoch(model, test_loader)
        writer.add_scalar('loss/eval', test_loss, epoch)
        writer.add_scalar('accuracy/eval', test_acc, epoch)

    print('results: ', tracker.get_best_scalars())


def train_epoch(
    model: MLP, optimizer: Adam, scheduler: OneCycleLR, trainloader: DataLoader
):

    model.train()
    train_loss = 0.
    correct = 0
    total_labels = 0
    step = 0
    for imgs, labels in trainloader:
        step += 1
        imgs: torch.Tensor
        labels: torch.Tensor
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        predictions = model.forward(imgs)
        loss = F.cross_entropy(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        total_labels += len(labels)
        correct += (predictions.argmax(1) == labels).sum().item()

    return (train_loss / step, correct / total_labels)


def eval_epoch(model: MLP, eval_loader: DataLoader):

    model.eval()
    eval_loss = 0.
    correct = 0
    total_labels = 0
    step = 0
    for imgs, labels in eval_loader:
        step += 1
        imgs: torch.Tensor
        labels: torch.Tensor
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        predictions = model.forward(imgs)

        loss = F.cross_entropy(predictions, labels)

        eval_loss += loss.item()
        correct += (predictions.argmax(1) == labels).sum().item()
        total_labels += len(labels)

    return (eval_loss / step, correct / total_labels)


if __name__ == '__main__':
    main()
