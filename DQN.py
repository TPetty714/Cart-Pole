'''
Programmed following SentDex youtube tutorials
https://www.youtube.com/watch?v=t3fbETsIBCY
'''


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import tensorflow as tf
import random
import time
import numpy as np
import gym

env = gym.make("CartPole-v1")

MEMORY_SIZE = 50000
MIN_MEMORY_SIZE = 1000
MODEL_NAME = "256x2"
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

class DQN:
    def __init__(self):
        # main model gets trained every step
        self.model = self.create_model()

        # Target model this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.observation_space.shape[0]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), input_shape=env.observation_space.shape[0]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.action_space.n, activation="linear"))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['Accuracy'])
        return model

    def update_memory(self, transition):
        self.memory.append(transition)

    def get_qs(self, state):
        return self.model_predict(np.array(state).reshape(-1. *state.shape))[0]

    def train(self, terminal_state, step):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        minibatch = random.sample(self.memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0




from keras.callbacks import TensorBoard
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
