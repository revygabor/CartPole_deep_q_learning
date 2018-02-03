import gym
from DQNAgent import DQNAgent
from numpy import reshape

env = gym.make('CartPole-v0')
env._max_episode_steps = 100000
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

load_model = True
episodes = 5000

agent = DQNAgent(state_size, action_size, load_model)

for e in range(episodes):
        state = env.reset()
        state = reshape(state, [1, state_size])

        done = False
        time = 0
        while not done:
            env.render()
            action = agent.act(state)

            state, _ , done, _ = env.step(action)
            state = reshape(state, [1, state_size])

            time += 1

            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, time))
                break

