from DQN_agent import DQNAgent
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

# Globals
# setting device on GPU if available, else CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU available.")
else:
    device = torch.device('cpu')
    print("GPU not available.")

# 0: LQ
# 1: Fixed Hinf
# 2: Dynamic Hinf
CONTROLLER = 1


def main():
    # Setup the environment
    env = gym.make('Acrobot-v1')

    torch.random.manual_seed(1)  # for successful exploration
    np.random.seed(0)

    state_num = 6
    action_num = 3
    hidden_size = 500
    discount_factor = 1
    epsilon = 1
    epsilon_decay_dqn = 0.0001
    learning_rate_dqn = 0.00001
    num_episodes = 200

    episode_gains = []
    mean_score = []
    e = 0

    # Create the agent
    agent = DQNAgent(state_num, action_num, hidden_size, learning_rate_dqn, epsilon, epsilon_decay_dqn,
                     discount_factor, 'Acrobot')

    while e < num_episodes:
        state = env.reset()
        done = False
        cumulated_reward = 0

        while not done:

            action = agent.select_action(state)
            state_new, reward, done, _ = env.step(action)

            if CONTROLLER == 0:
                agent.lq_controlled_learning(state, action, reward, state_new, done)
            elif CONTROLLER == 1:
                agent.robust_hinf_learning(state, action, reward, state_new, done)
                # Overfit on the final step ('steady state')
                if done and cumulated_reward > -480:
                    for trajectory_replay in range(500):
                        agent.robust_hinf_learning(state, action, reward, state_new, done)
            elif CONTROLLER == 2:
                agent.robust_dynamic_hinf_learning(state, action, reward, state_new, done)
            else:  # UNCONTROLLED
                agent.train_network(state, action, reward, state_new)


            state = state_new
            cumulated_reward += reward


        episode_gains.append(cumulated_reward)
        print("Episode ", e, " gain: ", cumulated_reward)

        # Check if won
        if e > 100:
            mean_score.append(np.mean(episode_gains[-100:]))
        else:
            mean_score.append(np.mean(episode_gains))
        e = e + 1

    plot_data(episode_gains, mean_score)
    env.close()


def plot_data(data, average):
    ax = plt.axes()
    x = torch.linspace(0, len(data), len(data))
    ax.plot(x, data, label="Episode rewards")
    ax.plot(x, average, label="Average reward")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
