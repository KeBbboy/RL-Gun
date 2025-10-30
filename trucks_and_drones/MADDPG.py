import numpy as np
import random
import copy
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

# ================= Replay Buffer with Prioritized Experience Replay =================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)

    def push(self, transition, td_error):
        max_prio = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            # replace oldest
            self.buffer.pop(0)
            self.buffer.append(transition)
            self.priorities.popleft()
            self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, td_errors):
        for idx, td in zip(batch_indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self):
        return len(self.buffer)

# ================= Actor and Critic Networks =================
class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(MLPActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class MLPCritic(nn.Module):
    def __init__(self, full_obs_dim, full_act_dim, hidden_dim=128):
        super(MLPCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(full_obs_dim + full_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        return self.net(x)

# ================= IAM-PER-MADDPG Algorithm =================
class IAM_PER_MADDPG:
    def __init__(self, env, n_agents, obs_dims, act_dims,
                 buffer_size=int(1e6), batch_size=1024,
                 gamma=0.95, tau=0.01,
                 actor_lr=1e-3, critic_lr=1e-3,
                 alpha=0.6, beta_start=0.4, beta_frames=100000):

        self.env = env
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        # Networks
        self.actors = [MLPActor(obs_dims[i], act_dims[i]).to(device) for i in range(n_agents)]
        self.critics = [MLPCritic(sum(obs_dims), sum(act_dims)).to(device) for _ in range(n_agents)]
        self.target_actors = copy.deepcopy(self.actors)
        self.target_critics = copy.deepcopy(self.critics)

        # Optimizers
        self.actor_opts = [optim.Adam(self.actors[i].parameters(), lr=actor_lr) for i in range(n_agents)]
        self.critic_opts = [optim.Adam(self.critics[i].parameters(), lr=critic_lr) for i in range(n_agents)]

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        self.frame = 1

    def select_action(self, obs_all_agents, invalid_masks):
        actions = []
        for i, actor in enumerate(self.actors):
            logits = actor(torch.FloatTensor(obs_all_agents[i]).to(device))
            # mask invalid
            logits[invalid_masks[i]] = -1e8
            probs = torch.softmax(logits, dim=-1)
            # epsilon-greedy
            if random.random() < 0.1:
                action = random.randrange(self.act_dims[i])
            else:
                action = probs.argmax().item()
            actions.append(action)
        return actions

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        transitions, indices, weights = self.buffer.sample(self.batch_size, beta)
        batch = Transition(*zip(*transitions))

        # convert batch to tensors
        obs_batch = [torch.FloatTensor(np.vstack(batch.obs[:, i])).to(device) for i in range(self.n_agents)]
        act_batch = [torch.LongTensor(batch.acts[:, i]).to(device) for i in range(self.n_agents)]
        rew_batch = [torch.FloatTensor(batch.rews[:, i]).to(device) for i in range(self.n_agents)]
        next_obs_batch = [torch.FloatTensor(np.vstack(batch.next_obs[:, i])).to(device) for i in range(self.n_agents)]
        done_batch = [torch.FloatTensor(batch.dones[:, i]).to(device) for _ in range(self.n_agents)]
        weights = torch.FloatTensor(weights).to(device)

        # update each agent
        td_errors = []
        for i in range(self.n_agents):
            # Critic update
            full_obs = torch.cat(obs_batch, dim=-1)
            full_act = torch.cat([nn.functional.one_hot(act_batch[j], self.act_dims[j]).float() for j in range(self.n_agents)], dim=-1)
            q_values = self.critics[i](full_obs, full_act).squeeze()

            # target
            next_full_obs = torch.cat(next_obs_batch, dim=-1)
            next_full_act = torch.cat([nn.functional.one_hot(self.target_actors[j](next_obs_batch[j]).argmax(dim=-1), self.act_dims[j]).float() for j in range(self.n_agents)], dim=-1)
            q_next = self.target_critics[i](next_full_obs, next_full_act).squeeze()
            q_target = rew_batch[i] + self.gamma * q_next * (1 - done_batch[i])

            td_error = q_values - q_target.detach()
            critic_loss = (weights * td_error.pow(2)).mean()

            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()

            td_errors.append(td_error.abs().cpu().detach().numpy())

            # Actor update
            curr_full_obs = torch.cat(obs_batch, dim=-1)
            curr_acts = []
            for j in range(self.n_agents):
                if j == i:
                    logits = self.actors[j](obs_batch[j])
                    curr_acts.append(nn.functional.one_hot(logits.argmax(dim=-1), self.act_dims[j]).float())
                else:
                    curr_acts.append(nn.functional.one_hot(act_batch[j], self.act_dims[j]).float())
            curr_full_act = torch.cat(curr_acts, dim=-1)
            actor_loss = -self.critics[i](curr_full_obs, curr_full_act).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

        # update priorities
        self.buffer.update_priorities(indices, np.mean(td_errors, axis=1))

        # update target networks
        for i in range(self.n_agents):
            for param, target_param in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.frame += 1

    def store(self, obs, acts, rews, next_obs, dones, invalid_masks):
        # compute initial td_error to push with max priority
        self.buffer.push((obs, acts, rews, next_obs, dones), td_error=1.0)

# Transition tuple
Transition = namedtuple('Transition', ('obs', 'acts', 'rews', 'next_obs', 'dones'))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Usage Example =================
if __name__ == '__main__':
    from custom_env import CustomEnv
    env = CustomEnv(...)
    n_agents = env.n_agents
    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
    act_dims = [env.action_space[i].n for i in range(n_agents)]

    algo = IAM_PER_MADDPG(env, n_agents, obs_dims, act_dims)
    episodes = 5000
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            invalid_masks = env.get_invalid_action_masks()
            acts = algo.select_action(obs, invalid_masks)
            next_obs, rew, done, info = env.step(acts)
            algo.store(obs, acts, rew, next_obs, done, invalid_masks)
            algo.update()
            obs = next_obs
