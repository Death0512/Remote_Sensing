from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy as np
from src.core.contracts import Detection


class BaseDetector(ABC):
    @abstractmethod
    def detect(
        self, image_input: Union[str, np.ndarray], **kwargs: Any
    ) -> List[Detection]:
        pass
