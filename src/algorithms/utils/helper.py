import tensorflow as tf


# # 折扣回报
# def discount_with_dones(rewards, dones, gamma):
#     """
#     用于多步TD！！
#     计算折扣回报序列：若 done 为 1，则后续回报不再累积。
#
#     参数:
#       rewards: list or np.array, 每步的即时回报
#       dones:   list or np.array, 结束标志（0 或 1）
#       gamma:   float, 折扣因子
#
#     返回:
#       List[float], 与 rewards 等长的折扣回报序列
#     """
#     discounted = []
#     r = 0.0
#     # 从最后一步开始反向累积
#     for reward, done in zip(rewards[::-1], dones[::-1]):
#         r = reward + gamma * r
#         # 如果 done==1，就把累计值清零
#         r *= (1.0 - done)
#         discounted.append(r)
#     # 再反转回来，保证与原序列顺序一致
#     return discounted[::-1]


def update_target_weights(main_weights, target_weights, tau):
    """
    使用 Polyak averaging (soft update) 将主网络参数更新到目标网络。

    参数:
      main_weights:   list of tf.Variable, 主网络的可训练参数
      target_weights: list of tf.Variable, 目标网络的可训练参数
      tau:            float in [0,1], 更新系数（越小更新越慢）
    """
    for w_main, w_target in zip(main_weights, target_weights):
        # w_target = (1 - tau) * w_target + tau * w_main
        w_target.assign((1.0 - tau) * w_target + tau * w_main)


def minimize_and_clip(optimizer, loss_fn, variables, clip_val=None):
    """
    Compute gradients of loss_fn() w.r.t. variables, optionally clip them,
    and apply via optimizer.

    Args:
        optimizer:  A tf.keras.optimizers.Optimizer instance.
        loss_fn:    A zero-argument function that computes and returns the loss tensor.
        variables:  Iterable of tf.Variable to optimize.
        clip_val:   Float or None. If set, gradients are clipped by global norm.

    Returns:
        The computed loss tensor (before clipping).
    """
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, variables)
    if clip_val is not None:
        grads = [tf.clip_by_norm(g, clip_val) if g is not None else None for g in grads]
    optimizer.apply_gradients(zip(grads, variables))
    return loss


def huber_loss(x, delta=1.0):
    """
    常用于 Critic 网络的回归损失
    Compute Huber loss elementwise:
      0.5 * x^2                  if |x| <= delta
      delta * (|x| - 0.5*delta)  if |x| >  delta

    Args:
        x:     tf.Tensor of arbitrary shape.
        delta: Float threshold where loss transitions from quadratic to linear.

    Returns:
        tf.Tensor same shape as x, the Huber loss per element.
    """
    return tf.where(
        tf.abs(x) <= delta,
        0.5 * tf.square(x),
        delta * (tf.abs(x) - 0.5 * delta)
    )
