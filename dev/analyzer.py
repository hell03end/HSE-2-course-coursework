from copy import deepcopy
import numpy as np
import re
try:
    from dev.ESOINN import EnhancedSelfOrganizingIncrementalNN
    from dev.commons import enable_logging
except ImportError as error:
    print(error.args)
    from .ESOINN import EnhancedSelfOrganizingIncrementalNN
    from .commons import enable_logging


class Analyzer:
    def __init__(self, nn: EnhancedSelfOrganizingIncrementalNN,
                 logging_level="debug"):
        if not isinstance(nn, EnhancedSelfOrganizingIncrementalNN):
            raise ValueError(f"Only EnhancedSelfOrganizingIncrementalNN can be"
                             f"analyzed, got: {type(nn)}")
        self._state = nn.current_state(deep=True)
        logger_name = re.sub(r"[' <>]", '', str(self.__class__))
        logger_name = re.sub(r"^class", '', logger_name)
        self._logger = enable_logging(f"{logger_name}", logging_level)

    def update_state(self, nn: EnhancedSelfOrganizingIncrementalNN) -> None:
        self._state = nn.current_state(deep=True)

    def get_mean_classes_feature_vectors(self) -> dict:
        if not self._state['classes']:
            self._logger.error("There are no classes!")
        nodes = self._state['nodes']
        mean_features = {}
        for class_id in self._state['classes']:
            mean_features[class_id] = \
                np.mean([nodes[node].feature_vector
                         for node in self._state['classes'][class_id]])
        return mean_features

    def get_mean_classes_density(self) -> dict:
        if not self._state['classes']:
            self._logger.error("There are no classes!")
        nodes = self._state['nodes']
        mean_density = {}
        for class_id in self._state['classes']:
            mean_density[class_id] = \
                np.mean([nodes[node].density
                         for node in self._state['classes'][class_id]])
        return mean_density

    def get_classes_sizes(self) -> dict:
        if not self._state['classes']:
            self._logger.error("There are no classes!")
        classes = {
            class_id: len(self._state['classes'][class_id])
            for class_id in self._state['classes']
        }
        return classes

    def get_inclass_distances(self):
        pass

    def get_outclass_distances(self):
        pass
