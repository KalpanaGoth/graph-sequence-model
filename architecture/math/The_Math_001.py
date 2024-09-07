import matplotlib.pyplot as plt
import numpy as np

# Visualization of ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Generate data points
x_values = np.linspace(-10, 10, 100)
y_values = relu(x_values)

# Plotting ReLU function
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='ReLU(x) = max(0, x)', color='blue')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.show()
