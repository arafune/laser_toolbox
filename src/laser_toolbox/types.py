import numpy as np
from numpy.typing import NDArray

Scalar = float | np.floating
ScalarOrArray = Scalar | NDArray[np.floating]
