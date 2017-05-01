import numpy as np
import random


# @TODO: add generators for training data
class TrainingSamples:
    def __init__(self, random_state: int=123) -> None:
        self.__seed = int(random_state)
        self.__distances = set()
        self.__params = {}
        random.seed(self.__seed)

    def reload_state(self, random_state: int) -> None:
        self.__seed = int(random_state)
        random.seed(self.__seed)

    def update_state(self) -> None:
        random.seed(self.__seed)

    def get_gauss_sample(self, count=2, size=1000, bias=1, noise=0,
                         **kwargs) -> np.array:
        self.update_state()
        self.__distances.clear()
        self.__params = {
            'sigma': kwargs.get("sigma", 0.3)
        }
        samples = []

        for i in range(count):
            pair = ((random.randint(0, i+1))/bias,
                    (random.randint(0, i+1))/bias)
            while pair in self.__distances:
                pair = ((random.randint(0, i+1))/bias,
                        (random.randint(0, i+1))/bias)
            self.__distances.add(pair)
            for _ in range(size if isinstance(size, int) else size[i]):
                samples.append(
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
            samples.append((random.uniform(**params),
                            random.uniform(**params)))
        return np.array(samples)

    def current_state(self) -> dict:
        return {
            'seed': self.__seed,
            'distances': self.__distances,
            'params': self.__params,
        }
