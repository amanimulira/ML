import gym
from DQN import Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Deep Q Learning
DQN = SummaryWriter(f"runs/Deep_Q_Network/data")

if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.00001)
    scores, eps_history = [], []
    n_games = 500
# GAMMA IS DISCOUNT FACTOR
    for i in range(n_games):

        score = 0
        done = False
        observation = env.reset()

        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    env.close()