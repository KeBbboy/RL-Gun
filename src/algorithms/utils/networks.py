# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ======================
# = 1. 网络结构定义      =
# ======================

class ActorModel(tf.keras.Model):
    """
    Actor 网络：输入观测，输出各离散动作的 logits。
    论文中 Actor 隐藏层：2 层 [512, 256]，可选 Tanh/ReLU 激活
    act_space_list: [Discrete(N), Discrete(N), Discrete(N), Discrete(2), Discrete(2)]
    """
    # def __init__(self, obs_dim, act_dim, hidden_units=(512, 256), activation='relu'):
    #     super().__init__()
    #     self.layers_ = []
    #     for u in hidden_units:
    #         self.layers_.append(tf.keras.layers.Dense(u, activation=activation))
    #     # 输出层：logits
    #     self.logits = tf.keras.layers.Dense(act_dim)
    #
    # def call(self, x):
    #     for layer in self.layers_:
    #         x = layer(x)
    #     return self.logits(x)
    def __init__(self, obs_dim, act_space_list, hidden_units, activation="relu"):
        """
        act_space_list: [Discrete(N), Discrete(N), Discrete(N), Discrete(2), Discrete(2)]
        """
        super().__init__()
        self.hidden_layers = []
        for u in hidden_units:
            self.hidden_layers.append(layers.Dense(u, activation=activation))
        # 为每个离散子空间建立一个 head
        self.heads = []
        for sp in act_space_list:
            self.heads.append(layers.Dense(sp.n))  # 输出 logits, 不带激活

    def call(self, obs):
        x = obs
        for layer in self.hidden_layers:
            x = layer(x)
        outputs = [head(x) for head in self.heads]

        # DEBUG: 检查每个 head 的输出 shape 是否正确
        for i, (logits, sp) in enumerate(zip(outputs, self.heads)):
            # print(f"[Debug] Head {i}: logits shape = {logits.shape}")
            assert logits.shape[1] == self.heads[i].units, \
                f"Head {i} logits shape mismatch: {logits.shape[1]} vs expected {self.heads[i].units}"

        return outputs


class CriticModel(tf.keras.Model):
    """
    Critic 网络：输入所有 agent 的 obs+act 拼接，输出单个 Q 值。
    论文中 Critic 隐藏层：3 层 [1024, 512, 256]，可选 Tanh/ReLU 激活
    """
    def __init__(self, input_dim, hidden_units=(1024, 512, 256), activation='relu'):
        super().__init__()
        self.layers_ = []
        for u in hidden_units:
            self.layers_.append(tf.keras.layers.Dense(u, activation=activation))
        # 输出层：单一 Q 值
        self.q = tf.keras.layers.Dense(1)

    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
        return self.q(x)
    






