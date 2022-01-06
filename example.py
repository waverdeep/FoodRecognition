import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':
    x = torch.randn(1, 20)
    y = torch.tensor([[1.0, 0., 1.0, 0., 0.]]) # get classA and classC as active

    model = nn.Linear(20, 5)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    for epoch in range(20):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print('loss: {:.3f}'.format(loss.item()))