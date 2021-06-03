import torch
import torch.nn as nn
from torch.distributions import Categorical

from .model import load_actor_model, load_critic_model
from .utils import wrap_into_batch, fancy, tensor_to_list


# set device to cpu or cuda
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    print("(PPO.py) Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("(PPO.py) Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = load_actor_model()
        self.actor.to(device)
        self.actor.config.pad_token_id = -1

        # critic
        self.critic = load_critic_model()
        self.critic.to(device)
        self.critic.config.pad_token_id = -1

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # determine an action
        action_probs = self.actor(state)[0][-1,:]
        action_probs = torch.softmax(action_probs, dim=0)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        # return an action and expected value of a current state.
        action_probs = self.actor(state)[0][:, -1, :]  # evaluate as batch -> [:,-1,:] # evaluate as unit -> [-1,:]
        action_probs = torch.softmax(action_probs, dim=1)  # evaluate as batch -> dim=1 # evaluate as unit -> dim=0
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)[0]

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.batch_size = (8, 4, 3, 2, 1)  # (size<=200, 200<size<=300, 300<size<=500, 500<size<=800, 800<size)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.LongTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        assert len(self.buffer.rewards) == len(self.buffer.actions) \
               == len(self.buffer.logprobs) == len(self.buffer.is_terminals)

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            if is_terminal:
                discounted_reward = 0
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for indexes in wrap_into_batch(self.buffer.states, self.batch_size)[1]:
                old_states = torch.tensor(fancy(self.buffer.states, indexes, fn=tensor_to_list)).to(device)
                old_actions = torch.tensor(fancy(self.buffer.actions, indexes, fn=tensor_to_list)).to(device)
                old_logprobs = torch.tensor(fancy(self.buffer.logprobs, indexes, fn=tensor_to_list)).to(device)
                batch_rewards = rewards[indexes]

                # old_states.shape == (batch, state_dim)  # ex cartpole -> (batch, 4)
                # old_actions.shape == (batch,)
                # old_logprobs.shape == (batch,)

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values, dim=0)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = batch_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # state_values.shape == (batch,)
                # batch_rewards.shape == (batch,)

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values.squeeze(), batch_rewards.squeeze()) - 0.01*dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            print('update loss', loss.mean())
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
