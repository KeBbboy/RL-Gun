# maddpg/agent_trainer.py

class AgentTrainer:
    """
    AgentTrainer 基类，定义了所有 Trainer 必须实现的接口。
    """
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args):
        """
        name: str, agent 名称
        obs_shape_n: List of observation shapes for each agent
        act_space_n: List of Gym action_spaces for each agent
        agent_index: int, 当前 agent 的索引
        args: argparse.Namespace，包含超参和 flag
        """
        self.name = name
        self.obs_shape_n = obs_shape_n
        self.act_space_n = act_space_n
        self.agent_index = agent_index
        self.args = args

    def action(self, obs, mask=None):
        """
        给定 obs（ndarray 或 tf.Tensor），返回一个动作（int 或 ndarray）。
        mask: 可选，invalid action masking 时使用。
        """
        raise NotImplementedError

    def experience(self, obs, act, rew, next_obs, done):
        """
        把一步交互 (obs, act, rew, next_obs, done) 存到 buffer。
        """
        raise NotImplementedError

    def preupdate(self):
        """
        在每次环境 step 前调用，用于清理或重置某些状态（可选）。
        """
        pass

    def update(self, agents, t, env=None):
        """
        在训练循环中被外部调用，用于执行一次网络参数更新。
        agents: List[AgentTrainer], 包含所有 agent
        t: int, 当前全局训练步数
        env: 可选，环境对象（用于 IAM 时获取 mask）
        """
        raise NotImplementedError
