from DQN_agent import DQNAgent
import gym
import torch
import matplotlib.pyplot as plt
import pickle
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
    # Set up the environment and the agent
    env = gym.make('CartPole-v0')

    state_num = 2  # Easier to learn if only the angle and angular velocity are considered
    action_num = 2
    hidden_size = 500
    discount_factor = 1
    epsilon = 1
    epsilon_decay_dqn = 0.0001
    learning_rate_dqn = 0.00005
    num_episodes = 500
    n_win_ticks = 195
    e = 0

    # Logging
    episode_gains = []
    mean_score = []
    Theta1_arr = []
    Theta2_arr = []
    Q1_arr = []
    Q2_arr = []
    reward_arr = []

    # Create the agent
    agent = DQNAgent(state_num, action_num, hidden_size, learning_rate_dqn, epsilon, epsilon_decay_dqn,
                     discount_factor, 'CartPole')

    while e < num_episodes:
        state = env.reset()
        done = False
        cumulated_reward = 0

        state_mem = []
        state_n_mem = []
        action_mem = []
        reward_mem = []
        done_mem = []

        while not done:
            action = agent.select_action(state[2:4])
            state_new, reward, done, _ = env.step(action)

            if CONTROLLER == 0:
                agent.lq_controlled_learning(state[2:4], action, reward, state_new[2:4], done)
            elif CONTROLLER == 1:
                agent.robust_hinf_learning(state[2:4], action, reward, state_new[2:4], done)
            elif CONTROLLER == 2:
                agent.robust_dynamic_hinf_learning(state[2:4], action, reward, state_new[2:4], done)
            else:  # UNCONTROLLED
                agent.train_network(state[2:4], action, reward, state_new[2:4])
            state = state_new
            cumulated_reward += reward

            state_mem.append(state[2:4])
            state_n_mem.append(state_new[2:4])
            action_mem.append(action)
            reward_mem.append(reward)
            done_mem.append(done)

            reward_arr.append(reward)

            # Compute the NTK for logging
            Theta1, Theta2, tmp1, tmp2 = agent.compute_temporal_difference_state_space(state[2:4], action,
                                                                                       reward, state_new[2:4])

            Theta1_arr.append(Theta1.item())
            Theta2_arr.append(Theta2.item())

            Q1_arr.append(agent.forward(state[2:4])[action].item())
            Q2_arr.append(torch.max(agent.forward(state_new[2:4])).item())

        episode_gains.append(cumulated_reward)
        print("Episode ", e, " gain: ", cumulated_reward)

        # Check if won
        if e > 100:
            mean_score.append(np.mean(episode_gains[-100:]))
            if mean_score[e] >= n_win_ticks:
                print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
        else:
            mean_score.append(np.mean(episode_gains))
        e = e + 1

    loss_array = agent.loss_arr
    control_input_array = agent.control_input_arr

    plot_data(episode_gains, mean_score, n_win_ticks)
    env.close()

    pickle.dump([episode_gains, mean_score, Q1_arr, Q2_arr, Theta1_arr, Theta2_arr,
                 control_input_array, loss_array], open("cartpole_agent_HinfFix.p", "wb"))


def plot_data(data, average, win):
    ax = plt.axes()
    x = torch.linspace(0, len(data), len(data))
    win = [win] * len(data)
    ax.plot(x, data, label="Episode rewards")
    ax.plot(x, win, label="Pass threshold")
    ax.plot(x, average, label="Average reward")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
