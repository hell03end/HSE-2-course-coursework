import matplotlib.pyplot as plt
import numpy as np
import random
try:
    from dev.mock import Node
except ImportError as error:
    print(error.args)
    from .mock import Node


# @TODO: add generators for training data
class TrainingSamples:
    def __init__(self, random_state: int=123) -> None:
        self.__seed = int(random_state)
        self.__distances = set()
        self.__params = {}
        self.__samples = []
        random.seed(self.__seed)

    def _reload_state(self, random_state: int) -> None:
        self.__seed = int(random_state)
        random.seed(self.__seed)

    def _update_state(self) -> None:
        random.seed(self.__seed)

    def get_gauss_sample(self, count=2, size=1000, bias=1, noise=0,
                         shuffle=True, classified=False, **kwargs) -> np.array:
        self._update_state()
        self.__distances.clear()
        self.__params = {
            'sigma': kwargs.get("sigma", 0.3)
        }
        self.__samples = []

        for i in range(count):
            pair = ((random.randint(0, i+1))/bias,
                    (random.randint(0, i+1))/bias)
            while pair in self.__distances:
                pair = ((random.randint(0, i+1))/bias,
                        (random.randint(0, i+1))/bias)
            self.__distances.add(pair)
            for _ in range(size if isinstance(size, int) else size[i]):
                if classified:
                    weight = (random.gauss(pair[0], **self.__params),
                              random.gauss(pair[1], **self.__params))
                    self.__samples.append(Node(weight, subclass_id=i))
                else:
                    self.__samples.append(
                        (random.gauss(pair[0], **self.__params),
                         random.gauss(pair[1], **self.__params)))

        if isinstance(size, float) or isinstance(size, int):
            noise_size = size/100*noise
        else:
            noise_size = max(size)/100*noise
        for i in range(int(noise_size)):
            params = {
                'a': 0 - 0.5/self.__params['sigma'],
                'b': max(
                    max(np.array(list(self.__distances))[:, 0]),
                    max(np.array(list(self.__distances))[:, 1])
                ) + 0.5/self.__params['sigma']
            }
            if classified:
                weight = (random.uniform(**params), random.uniform(**params))
                self.__samples.append(Node(weight))
            else:
                self.__samples.append((random.uniform(**params),
                                       random.uniform(**params)))
        if shuffle:
            random.shuffle(self.__samples)
        self.__samples = np.array(self.__samples)
        return self.current_state()

    def get_beta_sample(self, count=2, size=1000, bias=1, noise=0,
                        shuffle=True, scale=1, classified=False,
                        **kwargs) -> np.array:
        self._update_state()
        self.__distances.clear()
        self.__params = {
            'alpha': kwargs.get("alpha", 10),
            'beta': kwargs.get("beta", 10)
        }
        self.__samples = []

        for i in range(count):
            pair = ((random.randint(0, i + 1)) / bias,
                    (random.randint(0, i + 1)) / bias)
            while pair in self.__distances:
                pair = ((random.randint(0, i + 1)) / bias,
                        (random.randint(0, i + 1)) / bias)
            self.__distances.add(pair)
            for _ in range(size if isinstance(size, int) else size[i]):
                if classified:
                    weight = \
                        (random.betavariate(**self.__params)*scale + pair[0],
                         random.betavariate(**self.__params)*scale + pair[1])
                    self.__samples.append(Node(weight, subclass_id=i))
                else:
                    self.__samples.append(
                        (random.betavariate(**self.__params)*scale + pair[0],
                         random.betavariate(**self.__params)*scale + pair[1]))

        if isinstance(size, float) or isinstance(size, int):
            noise_size = size / 100 * noise
        else:
            noise_size = max(size)/100 * noise
        for i in range(int(noise_size)):
            params = {
                'a': 0 - scale,
                'b': max(
                    max(np.array(list(self.__distances))[:, 0]),
                    max(np.array(list(self.__distances))[:, 1])
                ) + scale
            }
            if classified:
                weight = (random.uniform(**params), random.uniform(**params))
                self.__samples.append(Node(weight))
            else:
                self.__samples.append((random.uniform(**params),
                                random.uniform(**params)))
        if shuffle:
            random.shuffle(self.__samples)
        self.__samples = np.array(self.__samples)
        return self.current_state()

    def display_sample(self):
        if not len(self.__samples):
            return
        if isinstance(self.__samples[0], Node):
            x, y, subclass = [], [], []
            for sample in self.__samples:
                x.append(sample.feature_vector[0])
                y.append(sample.feature_vector[1])
                subclass.append(sample.subclass_id)
            plt.scatter(x, y, c=subclass)
        else:
            plt.scatter(self.__samples[:, 0], self.__samples[:, 1])

    def current_state(self) -> dict:
        return {
            'seed': self.__seed,
            'distances': self.__distances,
            'params': self.__params,
            'samples': self.__samples,
        }
