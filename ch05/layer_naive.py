# 乗算レイヤ
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 順伝播
    # 掛け算
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # 逆伝播 (dout -> 微分)
    # ひっくり返して乗算
    def backward(self, dout):
        # xとyをひっくり返す
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

# 加算レイヤ
class AddLayer:
    def __init__(self):
        pass

    # 順伝播
    # 加算
    def forward(self, x, y):
        out = x + y
        return out

    # 逆伝播
    # 加算レイヤでは、上流から伝わってきた微分をそのまま下流に流すだけ
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
