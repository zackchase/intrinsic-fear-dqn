# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gamma', dest='gamma', type=float, default=.99, help='Discount factor for RL agent')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=.05, help='Exploration parameter for e-greedy')
parser.add_argument('--initial_replay', dest='initial_replay_size', type=int, default=20000, help='replay buffersize')

parser.add_argument('--env', dest='env_name', type=str, default="Breakout-v0", help='Selected enviroment from the openAI gym')
parser.add_argument('--device', dest='device', type=str, default='/cpu:0', help='device to use')

parser.add_argument('--results', dest='results_file', type=str, default='episodes.txt', help='Destination to save results')
parser.add_argument('--verbose', dest='verbose', type=bool, default=False, help='If verbose flag is on, print too much stuff')
parser.add_argument('--debug', dest='debug', type=bool, default=False, help='If debug flag is on, print error message')
parser.add_argument('--render', dest='render', type=bool, default=False, help='Render the gameplay to the monitor')
parser.add_argument('--buffer_size', dest='buffer_size', type=int, default=400000, help='Replay Buffer Size')


parser.add_argument('--fear_on', dest='fear_on', type=bool, default=False, help='Flag to indicate whether to use intrinsic fear')
parser.add_argument('--fear_radius', dest='fear_radius', type=int, default=1, help='How long a sequence before catastrophe to include')
parser.add_argument('--fear_factor', dest='fear_factor', type=float, default=0, help='How much to weight the fear penalty')
parser.add_argument('--fear_linear', dest='fear_linear', type=float, default=1000000, help='For scaling in fear factor, number of turns over which to run the linear gain')
parser.add_argument('--fear_warmup', dest='fear_warmup', type=int, default=20000, help='How many experiences before applying the intrinsic fear')



config = parser.parse_args()

results_file = "results/" + config.results_file
results_full_file = "results/" + "full-" + config.results_file


ENV_NAME = config.env_name  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 20000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = .1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = config.initial_replay_size  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = config.buffer_size  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
FEAR_LEARNING_RATE = .00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = False
TRAIN = True
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time




class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.avg_fear = 0
        # Create replay memory
        self.replay_memory = []


        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights


        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)


        #############################################################
        #   Build fear network
        #   Define its update operation
        #############################################################
        self.fear_s, self.fear_scores, fear_network = self.build_fear_network()

        fear_network_weights = fear_network.trainable_weights

        self.fear_y, self.fear_loss, self.fear_grads_update = self.build_fear_training_op(fear_network_weights)

        tf_config = tf.ConfigProto( allow_soft_placement=True,
                              log_device_placement=False )

        self.sess = tf.InteractiveSession(config=tf_config)
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.train.SummaryWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.initialize_all_variables())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

        self.danger_states = []
        self.safe_states = []



    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model


    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update




    ############################################################################
    #      Fear network
    ############################################################################
    def build_fear_network(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])

        fear_scores = model(s)
        return s, fear_scores, model


    def build_fear_training_op(self, fear_network_weights):
        y = tf.placeholder(tf.float32, [None])

        yhat = tf.minimum(tf.maximum(tf.reduce_sum(self.fear_scores, reduction_indices=1), .01), .99)

        loss =  tf.reduce_mean( - (y * tf.log(yhat) + (1-y) * tf.log(1-yhat)))

        optimizer = tf.train.RMSPropOptimizer(FEAR_LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=fear_network_weights)

        return y, loss, grads_update







    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def get_action(self, state):

        if config.fear_on and (self.t > config.fear_warmup):
            if config.debug:
                print("evaluating fear model")
            fear_score = self.fear_scores.eval(feed_dict={self.fear_s: np.float32(np.array([state]) / 255.0)})[0,0]
            if config.verbose:
                print(fear_score)
            self.avg_fear = self.avg_fear *.99 + fear_score * .01
        else:
            fear_score = None

        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            del self.replay_memory[0]

        if len(self.safe_states) > NUM_REPLAY_MEMORY:
            excess = len(self.safe_states) - NUM_REPLAY_MEMORY
            random.shuffle(self.safe_states)
            del self.safe_states[:excess]

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print("     EPISODE: %s TIMESTEP: %s DURATION: %s EPSILON: %s TOTAL_REWARD: %s AVG_MAX_Q: %s AVG_LOSS: %s MODE: %s FEAR: %s" % (
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode, self.avg_fear))

            with open(results_full_file, "a") as f:
                #f.write("%s, %s, %s, %s, %s, %s, %s, %s" % (self.episode+1, self.t, self.duration, self.epsilon, self.total_reward, self.total_q_max/float(self.duration), self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))
                f.write("" + str(self.total_reward) + "," + str(self.t) + "," + str(self.total_q_max/float(self.duration)) + "\n")
            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})

        if config.fear_on and (self.t > config.fear_warmup):
            adjusted_fear_factor = np.min([config.fear_factor * float(self.t)/config.fear_linear, config.fear_factor])
            fear_penalty = (1 - terminal_batch) * adjusted_fear_factor * np.max(self.fear_scores.eval(feed_dict={self.fear_s: np.float32(np.array(next_state_batch) / 255.0)}), axis=1) + terminal_batch * 10 * adjusted_fear_factor
            print("fear factor", adjusted_fear_factor, "fear penalty:", fear_penalty)
            y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1) - fear_penalty
        else:
            y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

        if config.fear_on and (self.t > config.fear_warmup):
            if config.debug:
                print("training fear model")
            if (len(self.danger_states) >= int(BATCH_SIZE/2)) and (len(self.safe_states) > int(BATCH_SIZE/2)):
                # print("danger and safe states large enough, starting to update")
                fear_minibatch = random.sample(self.danger_states, int(BATCH_SIZE/2)) + random.sample(self.safe_states, int(BATCH_SIZE/2))
                fear_y_batch = [1] * int(BATCH_SIZE / 2)  + [0] * int(BATCH_SIZE / 2)
                fear_loss, _ = self.sess.run([self.fear_loss, self.fear_grads_update], feed_dict={
                    self.fear_s: np.float32(np.array(fear_minibatch) / 255.0),
                    self.fear_y: fear_y_batch
                })
                if config.verbose:
                    print (fear_loss)


    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.scalar_summary(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.t += 1

        return action

    def log_danger(self, new_danger_states):
        self.danger_states += new_danger_states

    def log_safe(self, new_safe_states):
        self.safe_states += new_safe_states


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


# with tf.device(config.device):

env = gym.make(ENV_NAME)
agent = Agent(num_actions=env.action_space.n)

if TRAIN:  # Train mode
    ########################################################################
    #   Outer loop over episodes
    ########################################################################
    iteration = 0
    results_file = "results/" + config.results_file
    for _ in range(NUM_EPISODES):
        total_reward = 0
        episode_observations = []
        terminal = False
        observation = env.reset()
        for _ in range(random.randint(1, NO_OP_STEPS)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = agent.get_initial_state(observation, last_observation)
        while not terminal:
            iteration += 1
            last_observation = observation
            action = agent.get_action(state)
            observation, reward, terminal, _ = env.step(action)
            total_reward += reward
            if config.render:
                env.render()
            processed_observation = preprocess(observation, last_observation)
            state = agent.run(state, action, reward, terminal, processed_observation)
            if not terminal:
                episode_observations.append(state)

        if terminal:
            if config.fear_on:
                agent.log_danger(episode_observations[-config.fear_radius:])
                agent.log_safe(episode_observations[:-config.fear_radius])

                for e in episode_observations:
                    del e
                del episode_observations
                episode_observations = []

        with open(results_file, "a") as f:
            f.write(str(total_reward) + "," + str(iteration) + "\n")

else:  # Test mode
    # env.monitor.start(ENV_NAME + '-test')
    for _ in range(NUM_EPISODES_AT_TEST):
        terminal = False
        observation = env.reset()
        for _ in range(random.randint(1, NO_OP_STEPS)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = agent.get_initial_state(observation, last_observation)
        while not terminal:
            last_observation = observation
            action = agent.get_action_at_test(state)
            observation, _, terminal, _ = env.step(action)
            env.render()
            processed_observation = preprocess(observation, last_observation)
            state = np.append(state[1:, :, :], processed_observation, axis=0)
    # env.monitor.close()
