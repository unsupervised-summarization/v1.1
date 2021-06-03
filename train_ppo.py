import os

from ppo.env import Env
from ppo.PPO import PPO


env_name = "Unsupervised Summarization"

update_timestep = 1000        # update policy every n timesteps
K_epochs = 1                  # update policy for K epochs in one PPO update
J_epochs = 1                  # update reward model for J epochs in one PPO update

eps_clip = 0.2                # clip parameter for PPO
gamma = 0.99                  # discount factor

lr_actor = 1e-6               # learning rate for actor network
lr_critic = 5e-6              # learning rate for critic network
lr_reconstructor = 1e-6       # learning rate for reconstructor (reward model)

reconstructor_batch_size = 2  # batch size for reconstructor (reward model)

random_seed = 0               # set random seed if required (0 = no random seed)

env = Env()

run_num_pretrained = 0  # change this to prevent overwriting weights in same env_name folder

directory = "ppo/PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
env_checkpoint_path = directory + "ENV_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

print("--------------------------------------------------------------------------------------------")
print("PPO update frequency : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO J epochs : ", J_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("--------------------------------------------------------------------------------------------")
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)
print("optimizer learning rate reconstructor : ", lr_reconstructor)
print("============================================================================================")

# initialize a PPO agent
ppo_agent = PPO(lr_actor, lr_critic, gamma, K_epochs, eps_clip)

# recover checkpoints
try:
    ppo_agent.load(checkpoint_path)
    env.load(env_checkpoint_path)
    print('policy and env checkpoints were restored')
except FileNotFoundError:
    print('failed to restore policy and env checkpoints')
env.reward_logger.save(env_checkpoint_path + '-logger')  # logs for total rewards
env.reward_class.reward_logger.save(env_checkpoint_path + '-logger2')  # logs for each rewards

# training loop
just_time_step = 0
time_step = 0
while True:
    state = env.reset()
    # episode
    while True:
        # select action with policy
        action = ppo_agent.select_action(state)
        state, done = env.step(action)

        # saving is_terminals
        ppo_agent.buffer.is_terminals.append(done)

        time_step += 1
        just_time_step += 1

        # break; if the episode is over
        if done:
            try:
                reward = env.get_reward(log=True)
            except RuntimeError as e:
                if 'index' in str(e):
                    # if text is too long that the model can't take in them,
                    # skip this episode.
                    print(e)
                    print('-> skip this episode')
                    length = len(ppo_agent.buffer.rewards)
                    ppo_agent.buffer.actions = ppo_agent.buffer.actions[:length]
                    ppo_agent.buffer.states = ppo_agent.buffer.states[:length]
                    ppo_agent.buffer.logprobs = ppo_agent.buffer.logprobs[:length]
                    ppo_agent.buffer.rewards = ppo_agent.buffer.rewards[:length]
                    ppo_agent.buffer.is_terminals = ppo_agent.buffer.is_terminals[:length]
                    break
                else:
                    raise RuntimeError(e)
            if reward >= -0.1:
                print('document:', env.document.replace('\n', '\\n'))
                print('summary:', env.summary.replace('\n', '\\n'))
            print('reward:', reward, 'time_step:', just_time_step)
            for _ in range(len(ppo_agent.buffer.actions)-len(ppo_agent.buffer.rewards)):
                ppo_agent.buffer.rewards.append(reward)

            # train reward model
            env.update_reward_model(lr_reconstructor, J_epochs, reconstructor_batch_size)

            # update policy
            ppo_agent.update()

            prt_document = env.document.replace("\n", "\\n")
            prt_summary = env.summary.replace("\n", "\\n")
            with open('log.txt', 'a+') as f:
                f.write(f'\nreward:{reward}\ndocument:{prt_document}\nsummary:{prt_summary}\n')

            break

    if just_time_step >= update_timestep:
        # save models
        just_time_step = 0

        env.append_log()
        env.save(env_checkpoint_path)
        env.reward_logger.save(env_checkpoint_path + '-logger')  # logs for total rewards
        env.reward_class.reward_logger.save(env_checkpoint_path + '-logger2')  # logs for each rewards
        print("env saved")

        ppo_agent.save(checkpoint_path)
        print("policy saved")
