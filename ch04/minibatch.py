import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# (訓練画像、訓練ラベル), (テスト画像、テストラベル)
(x_train, t_train), (x_train, t_test) = \
    load_mnist(flatten=True, normalize=False)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = t_train[batch_mask]
t_batch = t_train[batch_mask]
