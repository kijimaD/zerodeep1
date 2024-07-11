def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# def cross_entropy_error(y, t):
#     delta = 1e-7 # 微細な値を追加してマイナス無限大を発生させないようにする
#     return -np.sum(t * np.log(y + delta))

# バッチ対応版
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 教師データがラベルで与えられた場合
# one-hot表現でtが0の要素は、交差エントロピー誤差も0であるから、その計算は無視してよい
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
