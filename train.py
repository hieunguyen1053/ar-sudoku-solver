import pkbar
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import Net
from digit_dataset import DigitDataset


def train(model, optimizer, device, train_loader, probar):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        probar.update(batch_idx, values=[('loss', loss), ])
    return epoch_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target)

    return test_loss / len(test_loader)


def main():
    writer = SummaryWriter('runs/sudoku')
    device = torch.device('cpu')
    model = Net().to(device)

    params = torch.load('data/data.pt')

    dataset1 = DigitDataset(params['train']['images'], params['train']['labels'])
    dataset2 = DigitDataset(params['test']['images'], params['test']['labels'])

    train_loader = DataLoader(dataset1, batch_size=32)
    test_loader = DataLoader(dataset2, batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    batch_per_epoch = len(train_loader)
    for epoch in range(1, epochs+1):
        probar = pkbar.Kbar(target=batch_per_epoch, epoch=epoch-1,
                            num_epochs=epochs, width=30, always_stateful=False)
        train_loss= train(model, optimizer, device, train_loader, probar)
        val_loss = evaluate(model, test_loader, device)

        probar.add(1, values=[('train_loss', train_loss), ('val_loss', val_loss),])
        writer.add_scalar('training loss',
                            train_loss,
                            epoch * len(train_loader) + batch_per_epoch)
        writer.add_scalar('validation loss',
                            val_loss,
                            epoch * len(test_loader) + batch_per_epoch)
    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
