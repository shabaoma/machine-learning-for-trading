import numpy as np
from random import sample
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def format_time(t):
    """Return a formatted time string 'HH:MM:SS'
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'


class DDQNAgent:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_model(self, trainable=True):
        layers = []
        n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.num_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0
            self.log_progress()

        self.experience.append((s, a, r, s_prime, not_done))

    def log_progress(self):
        if self.episodes % 10 == 0:  # 每10个episode输出一次
            nav_ma_100 = np.mean(self.rewards_history[-100:]) if len(self.rewards_history) >= 100 else np.mean(self.rewards_history)
            nav_ma_10 = np.mean(self.rewards_history[-10:]) if len(self.rewards_history) >= 10 else np.mean(self.rewards_history)
            market_nav_100 = 1.0  # 这里需要根据实际情况修改
            market_nav_10 = 1.0   # 这里需要根据实际情况修改
            win_ratio = 0.2       # 这里需要根据实际情况修改
            total = 0             # 这里需要根据实际情况修改
            
            template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
            template += 'Market: {:>6.1%} ({:>6.1%}) | '
            template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
            print(template.format(self.episodes, format_time(total), 
                                nav_ma_100-1, nav_ma_10-1, 
                                market_nav_100-1, market_nav_10-1, 
                                win_ratio, self.epsilon))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return

        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_network.predict(next_states, verbose=0)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict(next_states, verbose=0)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict(states, verbose=0)
        q_values[np.arange(self.batch_size), actions] = targets.numpy()

        # 使用fit来训练网络，这样可以显示训练进度
        history = self.online_network.fit(states, q_values,
                                        batch_size=self.batch_size,
                                        epochs=1,
                                        verbose=0)
        self.losses.append(history.history['loss'][0])
        self.train_steps += 1

        if self.total_steps % self.tau == 0:
            self.update_target()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
