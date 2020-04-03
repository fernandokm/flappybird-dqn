import datetime
import json
import random
from collections import deque

import numpy as np
from tensorflow.keras.models import load_model


def as_decay(value):
    if isinstance(value, Decay):
        return value
    else:
        return Decay(value)


class Decay:
    def __init__(self, value):
        self.value = value

    def step(self):
        return self.value

    def to_dict(self):
        return {'value': self.value}

    @staticmethod
    def from_dict(d):
        if set(d.keys()) == {'value'}:
            return Decay(d['value'])
        return ExponentialDecay.from_dict(d)


class ExponentialDecay(Decay):
    def __init__(self, initial, decay, limit):
        super().__init__(initial)
        self.initial = initial
        self.decay = decay
        self.limit = limit

    def step(self):
        self.value = max(self.limit, self.value*self.decay)
        return super().step()

    def to_dict(self):
        return {
            'value': self.value,
            'initial': self.initial,
            'decay': self.decay,
            'limit': self.limit
        }

    @staticmethod
    def from_dict(d):
        ed = ExponentialDecay(d['initial'], d['decay'], d['limit'])
        ed.value = d['value']
        return ed


class DQN:
    def __init__(self, model, *, memory_size=16384, gamma=.99, epsilon=.1, action_probs=None):
        self.model = model
        self.gamma = gamma
        self.epsilon = as_decay(epsilon)
        self.memory_size = memory_size

        self.state_size = model.input_shape[1]
        self.action_count = model.output_shape[1]

        self.memory = deque(maxlen=memory_size)

        self.action_probs = action_probs

    def qs(self, states):
        return self.model.predict(states)

    def q(self, state):
        return self.qs(state.reshape(1, *state.shape))[0]

    def policy(self, state, q=None):
        if np.random.random() < self.epsilon.value:
            return np.random.choice(self.action_count, p=self.action_probs)
        if q is None:
            q = self.q(state)
        return np.argmax(q)

    def remember(self, s1, a, s2, r, done):
        self.memory.append((s1, a, s2, r, done))

    def _train_batch(self, batch, epochs):
        x = np.array([mem[0] for mem in batch])
        y = self.model.predict(x)
        q2 = self.model.predict(np.array([mem[2] for mem in batch]))
        for i, (s1, a, s2, r, done) in enumerate(batch):
            if done:
                y[i, a] = r
            else:
                y[i, a] = r + self.gamma * np.max(q2[i])
        self.model.fit(x, y, epochs=epochs, verbose=0)

    def train_memory(self, batch_size, *, batch_count=1, epochs=1):
        for _ in range(batch_count):
            if batch_size >= len(self.memory):
                batch = self.memory
            else:
                batch = random.sample(self.memory, k=batch_size)
            self._train_batch(batch, epochs=epochs)

    def to_dict(self):
        return {
            'gamma': self.gamma,
            'epsilon': self.epsilon.to_dict(),
            'memory_size': self.memory_size,
            'action_probs': self.action_probs,
        }

    def save(self, name):
        name = self._normalize_name(name)
        self.model.save(name + '.h5')
        with open(name + '.json', 'w') as f:
            json.dump(self.to_dict(), f, sort_keys=True)

    @staticmethod
    def load(name):
        name = DQN._normalize_name(name)
        model = load_model(name + '.h5')
        with open(name + '.json', 'r') as f:
            d = json.load(f)
        return DQN(model,
                   memory_size=d['memory_size'],
                   gamma=d['gamma'],
                   epsilon=Decay.from_dict(d['epsilon']),
                   action_probs=d['action_probs'])

    @staticmethod
    def _normalize_name(name):
        if name.endswith('.h5'):
            return name[:-3]
        elif name.endswith('.json'):
            return name[:-5]
        return name
