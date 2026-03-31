import numpy as np
import matplotlib.pyplot as plt

# Квадратична функція
def quadratic_function(x):
    return x**2 + 5*x + 6

# Градієнт квадратичної функції
def gradient(x):
    return 2*x + 5

# Алгоритм градієнтного спуску
def gradient_descent(learning_rate, epochs, initial_x):
    x_values = [initial_x]
    for epoch in range(epochs):
        current_x = x_values[-1]
        grad = gradient(current_x)
        new_x = current_x - learning_rate * grad
        x_values.append(new_x)
    return x_values

# Параметри градієнтного спуску
learning_rate = 0.1
epochs = 20
initial_x = 0

# Запуск градієнтного спуску
optimized_x_values = gradient_descent(learning_rate, epochs, initial_x)

# Графік функції та градієнтного спуску
x_values = np.linspace(-5, 2, 100)
y_values = quadratic_function(x_values)

plt.plot(x_values, y_values, label='f(x) = x^2 + 5x + 6')
plt.scatter(optimized_x_values, [quadratic_function(x) for x in optimized_x_values], color='red', label='Градієнтний спуск')
plt.title('Градієнтний спуск для квадратичної функції')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()