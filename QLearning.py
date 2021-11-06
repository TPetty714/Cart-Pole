import math
import numpy as np
import time
import gym
import torch
import random

env = gym.make("CartPole-v1")
tot_bins = 5
cart_position_state_array = np.linspace(-2.5, +2.5, num=tot_bins-1, endpoint=False)
cart_velocity_state_array = np.linspace(-2.5, +2.5, num=tot_bins-1, endpoint=False)
pole_angle_state_array = np.linspace(-12.0, +12.0, num=tot_bins-1, endpoint=False)
tip_velocity_state_array = np.linspace(-2.5, +2.5, num=tot_bins-1, endpoint=False)
env.reset()

episodes = 100000
exploration_rate = 0.5
min_explore_rate = 0.01
min_learning_rate = 0.1
learning_rate = 0.1
discount_factor = 0.1

def select_action(observation, Q):
    if random.random() < exploration_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[observation])
        if action == 0:
            action = env.action_space.sample()
    return action + 1

def select_explore_rate(x):
    return max(min_explore_rate, min(1, 1.0 - math.log10((x+1)/25)))

def select_learning_rate(x):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((x+1)/25)))

def run():
    global episodes
    global exploration_rate
    global learning_rate
    action_space = env.action_space.n
    Q = np.zeros((tot_bins, tot_bins, tot_bins, tot_bins) + (action_space,))
    for i in range(episodes):
        exploration_rate = select_explore_rate(i)
        learning_rate = select_learning_rate(i)
        finalReward = 0
        observation = env.reset()
        observation = (
            np.digitize(observation[0], cart_position_state_array),
            np.digitize(observation[1], cart_velocity_state_array),
            np.digitize(observation[2], pole_angle_state_array),
            np.digitize(observation[3], tip_velocity_state_array)
        )
        action = env.action_space.sample() + 1
        while True:
            # env.render()
            new_observation, reward, done, info = env.step(action - 1)
            new_observation = (
                np.digitize(new_observation[0], cart_position_state_array),
                np.digitize(new_observation[1], cart_velocity_state_array),
                np.digitize(new_observation[2], pole_angle_state_array),
                np.digitize(new_observation[3], tip_velocity_state_array)
            )
            finalReward += reward
            Q[observation, action] = (1 - learning_rate) * Q[observation, action] + \
                                     learning_rate * (finalReward + discount_factor * np.amax(Q[new_observation]))

            if done:
                print(finalReward)
                break
            observation = new_observation
            action = select_action(observation, Q)
        # print(finalReward)
        env.close()

# import gym
# import numpy as np
# import random
# import math
#
# environment = gym.make('CartPole-v1')
#
# no_buckets = (1, 1, 6, 3)
# no_actions = environment.action_space.n
#
# state_value_bounds = list(zip(environment.observation_space.low,
# environment.observation_space.high))
# state_value_bounds[1] = [-0.5, 0.5]
# state_value_bounds[3] = [-math.radians(50), math.radians(50)]
#
# action_index = len(no_buckets)
# q_value_table = np.zeros(no_buckets + (no_actions,))
#
# min_explore_rate = 0.01
# min_learning_rate = 0.1
#
# max_episodes = 1000
# max_time_steps = 250
# streak_to_end = 120
# solved_time = 199
# discount = 0.99
# no_streaks = 0
#
# def select_action(state_value, explore_rate):
#     if random.random() <explore_rate:
#         action = environment.action_space.sample()
#     else:
#         action = np.argmax(q_value_table[state_value])
#     return action
#
# def select_explore_rate(x):
#     return max(min_explore_rate, min(1, 1.0 - math.log10((x+1)/25)))
#
# def select_learning_rate(x):
#     return max(min_learning_rate, min(0.5, 1.0 - math.log10((x+1)/25)))
#
# def bucketize_state_value(state_value):
#     bucket_indexes = []
#     for i in range(len(state_value)):
#         if state_value[i] <= state_value_bounds[i][0]:
#             bucket_index = 0
#         elif state_value[i] >= state_value_bounds[i][1]:
#             bucket_index = no_buckets[i] - 1
#         else:
#             bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
#             offset = (no_buckets[i]-1)*state_value_bounds[i][0]/bound_width
#             scaling = (no_buckets[i]-1)/bound_width
#             bucket_index = int(round(scaling*state_value[i] - offset))
#         bucket_indexes.append(bucket_index)
#     return tuple(bucket_indexes)
#
# for episode_no in range(max_episodes):
#     explore_rate = select_explore_rate(episode_no)
#     learning_rate = select_learning_rate(episode_no)
#
#     observation = environment.reset()
#
#     start_state_value = bucketize_state_value(observation)
#     previous_state_value = start_state_value
#
#     for time_step in range(max_time_steps):
#         environment.render()
#         selected_action = select_action(previous_state_value, explore_rate)
#         observation, reward_gain, completed, _ = environment.step(selected_action)
#         state_value = bucketize_state_value(observation)
#         best_q_value = np.amax(q_value_table[state_value])
#         q_value_table[previous_state_value + (selected_action,)] += learning_rate * (
#         reward_gain + discount * (best_q_value) - q_value_table[previous_state_value + (selected_action,)])
#
#         # print('Episode number : %d' % episode_no)
#         # print('Time step : %d' % time_step)
#         # print('Selection action : %d' % selected_action)
#         # print('Current state : %s' % str(state_value))
#         # print('Reward obtained : %f' % reward_gain)
#         # print('Best Q value : %f' % best_q_value)
#         # print('Learning rate : %f' % learning_rate)
#         # print('Explore rate : %f' % explore_rate)
#         # print('Streak number : %d' % no_streaks)
#
#         if completed:
#             print('Episode %d finished after %f time steps' % (episode_no, time_step))
#             if time_step>= solved_time:
#                 no_streaks += 1
#             else:
#                 no_streaks = 0
#                 break
#
#     previous_state_value = state_value
#
#     if no_streaks>streak_to_end:
#         break
if __name__ == '__main__':
    run()