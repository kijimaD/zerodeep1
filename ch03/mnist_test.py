import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# (訓練画像、訓練ラベル), (テスト画像、テストラベル)
(x_train, t_train), (x_train, t_test) = \
    load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_train.shape)
print(t_train.shape)
