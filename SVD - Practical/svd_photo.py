import matplotlib.pyplot as pyt
import numpy as np

# arcsin in radians
def arcsine(x):
    return(np.arcsin(x))

# log function
def my_func(x):
    return(((np.sqrt(3) / 5) * np.log(np.abs(1 - x))) + (((2 * np.sqrt(3)) / 5) * np.log(np.abs(1 + x))))

x_values = np.arange(0.011, 4 * np.pi, 0.01)

y_values_arc = arcsine(x_values)
y_values_my_func = my_func(x_values)