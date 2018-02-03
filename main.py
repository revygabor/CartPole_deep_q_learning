import gym
from DQNAgent import DQNAgent
from numpy import reshape

env = gym.make('CartPole-v0')
env._max_episode_steps = 10000
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

load_model = True
train = True
episodes = 5000

agent = DQNAgent(state_size, action_size, load_model, train)

for e in range(episodes):
        state = env.reset()
        state = reshape(state, [1, state_size])

        for time_t in range(10000):
            env.render()
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, time_t))
                break

        agent.train(32)

        if(e%100==0):
            agent.save_model()
