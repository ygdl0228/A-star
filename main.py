# @Time    : 2023/4/9 23:01
# @Author  : ygd
# @FileName: main.py
# @Software: PyCharm

# @Time    : 2023/4/10 10:11
# @Author  : ygd
# @FileName: test.py
# @Software: PyCharm

import random
from multiprocessing import Pipe, Process
import numpy as np
import torch
from agent import ReplayBuffer, DQN
from env import route
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

env_config = {'road_length': 1, 'YC_interval': 1, 'QC_interval': 1, 'buffer_length': 2, 'node_nums_x': 16,
              'node_nums_YC': 6, 'node_nums_QC': 3, 'max_speed': 6, 'min_speed': 1, 'max_acceleration': 1,
              'min_acceleration': -1}

agent_config = {'state_dim': 8, 'hidden_dim': 128, 'action_dim': 12, 'learning_rate': 1e-2, 'lr_decay': 0.97,
                'min_lr': 1e-7,
                'gamma': 0.9, 'epsilon': 0.8,
                'epsilon_decay': 0.97, 'min_epsilon': 1e-4, 'target_update': 100,
                'device': torch.device("cuda") if torch.cuda.is_available() else torch.device(
                    "cpu"), 'dqn_type': 'D3QN',
                'output_dir': f'./output/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def simulate(i, eps, agent, pipe):
    print(f'第{i + 1}进程')
    episode_return = 0
    env = route(**env_config)
    state = env.reset()
    done = False
    transition_dict = []
    while not done:
        avail_action_mask = env.get_avail_agent_action()
        action = agent.take_action(state, avail_action_mask, eps)
        next_state, reward, done = env.step(action)
        next_avail_action_mask = env.get_avail_agent_action()
        transition_dict.append((state, action, reward, next_state, next_avail_action_mask.tolist(), done))
        state = next_state
        episode_return += reward
    print(f'第{i + 1}进程采样结束')
    data = [transition_dict, episode_return]
    pipe.send(data)


def main():
    num_episodes = 10000
    buffer_size = 1000000000
    minimal_size = 100
    batch_size = 64
    num_process = 4
    set_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(**agent_config)
    return_list1 = []
    return_list2 = []
    return_list3 = []
    return_list4 = []
    loss_list = []
    for eps in range(num_episodes):
        print(f'第{eps + 1}次训练')
        pipes = [Pipe() for _ in range(num_process)]
        processes = [Process(target=simulate, args=(i, eps, agent, pipes[i][1])) for i in range(num_process)]
        for p in processes:
            p.start()
        episode_return = [0] * num_process
        for i in range(num_process):
            pipe = pipes[i][0]
            data = pipe.recv()
            state, action, reward, next_state, next_avail_action_mask, done = zip(*data[0])
            for j in range(len(data[0])):
                replay_buffer.add(state[j], action[j], reward[j], next_state[j], next_avail_action_mask[j], done[j])
            episode_return[i] = data[1]
        torch.save(agent.q_net, agent.weight_dir + f'{eps}_model.pkl')
        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_naam, b_d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'next_avail_action_mask': b_naam,
                'dones': b_d
            }
            dqn_loss = agent.update(transition_dict, eps)
            loss_list.append(dqn_loss)

        return_list1.append(episode_return[0])
        return_list2.append(episode_return[1])
        return_list3.append(episode_return[2])
        return_list4.append(episode_return[3])
        plt.subplot(231)
        plt.plot(return_list1)
        plt.title("子进程1")
        plt.subplot(232)
        plt.plot(return_list2)
        plt.title("子进程2")
        plt.subplot(233)
        plt.plot(return_list3)
        plt.title("子进程3")
        plt.subplot(234)
        plt.plot(return_list4)
        plt.title("子进程4")
        plt.subplot(235)
        plt.plot(loss_list)
        plt.title("loss")
        plt.tight_layout()
        plt.savefig('return')
        plt.show()
    for p in processes:
        p.terminate()


if __name__ == "__main__":
    main()
