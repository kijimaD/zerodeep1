import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
  # 重みの初期化
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * \
    np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * \
    np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  # 推論
  # yは、ニューラルネットワークの出力層の結果、すなわち入力データ x に対する予測結果。この予測結果は、各クラスに対する確率分布を表す。入力データがそれぞれのクラスに属する確率のベクトルともいえる。
  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y

  # 損失関数
  # x:入力データ, t:教師データ
  # 予測結果と教師データから誤差を計算する
  def loss(self, x, t):
    y = self.predict(x)
    return cross_entropy_error(y, t)

  # 認識精度
  # 全体から求める
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  # x:入力データ, t:教師データ
  # 重み・バイアスパラメータに対する勾配を求める
  # 数値微分によって、各パラメータの損失関数に対する勾配を求める
  def numerical_gradient(self, x, t):
      loss_W = lambda W: self.loss(x, t)

      grads = {} # 勾配
      grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
      grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
      grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
      grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

      return grads
