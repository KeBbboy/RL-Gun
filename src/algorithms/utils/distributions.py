# distributions.py
"""
A TensorFlow 2.x compatible probability distributions module for RL policies.
Supports Discrete (Categorical), Continuous (Gaussian), Bernoulli, MultiDiscrete spaces.
Built on TensorFlow Probability.
"""
import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Discrete, Box, MultiDiscrete, MultiBinary

# Alias for convenience
tfd = tfp.distributions


class Pd:
    """Abstract base class for a probability distribution."""
    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.dist = tfd.Categorical(logits=logits)

    def sample(self):
        return self.dist.sample()

    def logp(self, x):
        return self.dist.log_prob(x)

    def entropy(self):
        return self.dist.entropy()

    def mode(self):
        return tf.argmax(self.dist.logits_parameter(), axis=-1)


class DiagGaussianPd(Pd):
    def __init__(self, loc, scale):
        # loc: [batch_size, dim], scale: same shape
        self.dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def sample(self):
        return self.dist.sample()

    def logp(self, x):
        return self.dist.log_prob(x)

    def entropy(self):
        return self.dist.entropy()

    def mode(self):
        return self.dist.mean()


class BernoulliPd(Pd):
    def __init__(self, logits):
        self.dist = tfd.Bernoulli(logits=logits)

    def sample(self):
        return self.dist.sample()

    def logp(self, x):
        return self.dist.log_prob(x)

    def entropy(self):
        return self.dist.entropy()

    def mode(self):
        return tf.cast(self.dist.probs_parameter() > 0.5, tf.int32)


# Factory to create distribution given Gym action space and network output

def make_pdtype(ac_space):
    """
    Given a Gym action space, returns a function that maps network output(s) to a Pd instance.
    Usage:
      pd_constructor = make_pdtype(env.action_space)
      pd = pd_constructor(*network_outputs)
    """
    if isinstance(ac_space, Discrete):
        def pd_fn(flat_logits):
            return CategoricalPd(flat_logits)
        return pd_fn

    if isinstance(ac_space, Box):
        assert len(ac_space.shape) == 1, "Only 1D Box spaces supported"
        def pd_fn(flat_params):
            # assume flat_params shape [..., 2*dim] split into loc and log_std
            dim = ac_space.shape[0]
            loc, log_std = tf.split(flat_params, num_or_size_splits=2, axis=-1)
            scale = tf.exp(log_std)
            return DiagGaussianPd(loc, scale)
        return pd_fn

    if isinstance(ac_space, MultiDiscrete):
        def pd_fn(flat_logits):
            # flat_logits: concatenated logits for each sub-space
            splits = ac_space.nvec
            logits_split = tf.split(flat_logits, splits, axis=-1)
            samples = []
            for lg in logits_split:
                samples.append(CategoricalPd(lg))
            return samples  # list of Pd instances
        return pd_fn

    if isinstance(ac_space, MultiBinary):
        def pd_fn(flat_logits):
            return BernoulliPd(flat_logits)
        return pd_fn

    raise NotImplementedError(f"Unsupported action space: {ac_space}")
