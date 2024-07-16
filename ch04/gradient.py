import numpy as np

# 微分
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x) # xと同じ形状の配列を生成する

#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x)

#         x[idx] = tmp_val - h
#         fxh2 = f(x)

#         grad[idx] = (fxh1 - fxh2) / (2*h)
#         x[idx] = tmp_val # 値を元に戻す

#     return grad

# 勾配降下
# lr -> learning rate
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x

  for i in range(step_num):
    # 関数の勾配を計算する
    grad = numerical_gradient(f, x)
    # 現在の点 x を勾配 grad の方向に学習率 lr だけ移動させる。この操作により、関数 f の値が減少する方向に移動する
    x -= lr * grad

  return x


# 関数fの勾配を計算する。引数に損失関数が渡された場合、損失関数の勾配を計算することになる
# なぜこれを求めるか: 勾配がわかれば、降下していけるから
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # 次元イテレーション
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext() # すべての要素を処理し終わるとit.finishedがtrueになって、whileを抜ける

    return grad
