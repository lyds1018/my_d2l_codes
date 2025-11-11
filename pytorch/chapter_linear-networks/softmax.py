import torch
from torch import nn
from d2l import torch as d2l


# 定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# 模型参数初始化函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
# apply 函数会递归地将 init_weights 函数应用到 net 的每一层
net.apply(init_weights)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')    # reduction='none'，不做平均

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
if __name__ == '__main__':
    num_epochs = 10
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()   # 先取平均再反向
            trainer.zero_grad()
            l.backward()
            trainer.step()
        print(f'epoch {epoch+1}', f'test acc {d2l.evaluate_accuracy(net, test_iter):.3f}')

