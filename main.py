import gym
from DQNAgent import DQNAgent
from numpy import reshape
import math

env = gym.make('CartPole-v0')
env._max_episode_steps = 500
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]

load_model = True
train = True
episodes = 5000

agent = DQNAgent(state_size, action_size, load_model, train)


def normalize(obs):
    return [(obs[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]


for e in range(episodes):
        state = env.reset()
        next_state = normalize(state)
        state = reshape(state, [1, state_size])

        for time_t in range(env._max_episode_steps):
            env.render()
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = normalize(next_state)
            next_state = reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, time_t))
                break

        agent.train(32)

        if(e%100==0):
            agent.save_model()

