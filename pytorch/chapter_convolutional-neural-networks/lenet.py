import torch
from d2l import torch as d2l
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10),
)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train(net, train_iter, test_iter, num_epochs, lr):
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # CrossEntropyLoss会自动计算softmax和负对数似然损失
    loss = nn.CrossEntropyLoss()

    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        train_iter_loss = []
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_iter_loss.append(l.item())
        train_loss.append(sum(train_iter_loss) / len(train_iter_loss))

        test_iter_loss = []
        for X, y in test_iter:
            with torch.no_grad():
                y_hat = net(X)
                l = loss(y_hat, y)
                test_iter_loss.append(l.item())
        test_loss.append(sum(test_iter_loss) / len(test_iter_loss))

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}"
        )

    return train_loss, test_loss


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_epochs, lr = 10, 0.1
    train_loss, test_loss = train(net, train_iter, test_iter, num_epochs, lr)
