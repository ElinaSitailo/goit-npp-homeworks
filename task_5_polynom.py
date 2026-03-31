from timeit import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


try:
    from IPython import get_ipython
except ImportError:

    def get_ipython():
        return None


# 1. Згенеруйте дані
np.random.seed(0)  # Для відтворюваності результатів
a = np.random.rand(100)
b = np.random.rand(100)


def polynomial(a, b):
    return 4 * a**2 + 5 * b**2 - 2 * a * b + 3 * a - 6 * b


y = polynomial(a, b)

# 2. додаткові ознаки для кожного степеня
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(np.column_stack((a, b)))


# 3. Реалізуйте функції для методів градієнтного спуску
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    gradient_norms = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        loss = (1 / m) * np.sum(errors**2)
        gradient = (1 / m) * X.T.dot(errors)

        losses.append(loss)
        gradient_norms.append(np.linalg.norm(gradient))

        theta -= learning_rate * gradient

    return theta, losses, gradient_norms


def SGD(X, y, learning_rate=0.01, iterations=1000, seed=0):

    m, n = X.shape
    rng = np.random.default_rng(seed)
    theta = np.zeros(n)
    losses = []
    gradient_norms = []

    for _ in range(iterations):
        indices = rng.permutation(m)

        for idx in indices:
            X_i = X[idx : idx + 1]
            y_i = y[idx : idx + 1]

            predictions = X_i.dot(theta)
            errors = predictions - y_i
            gradient = X_i.T.dot(errors)

            theta -= learning_rate * gradient.flatten()

        epoch_predictions = X.dot(theta)
        epoch_errors = epoch_predictions - y
        epoch_loss = (1 / m) * np.sum(epoch_errors**2)
        epoch_gradient = (1 / m) * X.T.dot(epoch_errors)

        losses.append(epoch_loss)
        gradient_norms.append(np.linalg.norm(epoch_gradient))

    return theta, losses, gradient_norms


def rmsprop(X, y, learning_rate=0.01, iterations=1000, beta=0.9, epsilon=1e-8):
    m, n = X.shape
    theta = np.zeros(n)
    cache = np.zeros(n)
    losses = []
    gradient_norms = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        loss = (1 / m) * np.sum(errors**2)
        gradient = (1 / m) * X.T.dot(errors)

        losses.append(loss)
        gradient_norms.append(np.linalg.norm(gradient))

        cache = beta * cache + (1 - beta) * gradient**2
        theta -= learning_rate * gradient / (np.sqrt(cache) + epsilon)

    return theta, losses, gradient_norms


def adam(X, y, learning_rate=0.01, iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, n = X.shape
    theta = np.zeros(n)
    m_t = np.zeros(n)
    v_t = np.zeros(n)
    losses = []
    gradient_norms = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        loss = (1 / m) * np.sum(errors**2)
        gradient = (1 / m) * X.T.dot(errors)

        losses.append(loss)
        gradient_norms.append(np.linalg.norm(gradient))

        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient**2
        m_hat = m_t / (1 - beta1 ** (i + 1))
        v_hat = v_t / (1 - beta2 ** (i + 1))
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return theta, losses, gradient_norms


def nadam(X, y, learning_rate=0.01, iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, n = X.shape
    theta = np.zeros(n)
    m_t = np.zeros(n)
    v_t = np.zeros(n)
    losses = []
    gradient_norms = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        loss = (1 / m) * np.sum(errors**2)
        gradient = (1 / m) * X.T.dot(errors)

        losses.append(loss)
        gradient_norms.append(np.linalg.norm(gradient))

        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient**2
        m_hat = m_t / (1 - beta1 ** (i + 1))
        v_hat = v_t / (1 - beta2 ** (i + 1))
        theta -= learning_rate * (beta1 * m_hat + (1 - beta1) * gradient / (1 - beta1 ** (i + 1))) / (np.sqrt(v_hat) + epsilon)

    return theta, losses, gradient_norms


def run_with_timeit(func):
    result_box = {}

    def wrapped():
        result_box["result"] = func()

    ipython = get_ipython()
    if ipython is not None:
        wrapper_name = f"_single_run_timeit_wrapper_{id(wrapped)}"
        ipython.user_ns[wrapper_name] = wrapped
        try:
            timeit_result = ipython.run_line_magic(
                "timeit",
                f"-o -q -n 1 -r 1 {wrapper_name}()",
            )
            elapsed = timeit_result.average
        finally:
            ipython.user_ns.pop(wrapper_name, None)
    else:
        elapsed = timeit(wrapped, number=1)

    return elapsed, result_box["result"]


ITERATIONS = 1500

gd_time, (_, losses_gd, grad_norms_gd) = run_with_timeit(lambda: gradient_descent(X, y, iterations=ITERATIONS))

sgd_time, (_, losses_sgd, grad_norms_sgd) = run_with_timeit(
    lambda: SGD(
        X,
        y,
        iterations=ITERATIONS,
        seed=0,
    )
)

rmsprop_time, (_, losses_rmsprop, grad_norms_rmsprop) = run_with_timeit(lambda: rmsprop(X, y, iterations=ITERATIONS))

adam_time, (_, losses_adam, grad_norms_adam) = run_with_timeit(lambda: adam(X, y, iterations=ITERATIONS))

nadam_time, (_, losses_nadam, grad_norms_nadam) = run_with_timeit(lambda: nadam(X, y, iterations=ITERATIONS))

results = [
    {
        "Iterations": ITERATIONS,
        "GD Time": gd_time,
        "SGD Time": sgd_time,
        "RMSProp Time": rmsprop_time,
        "Adam Time": adam_time,
        "Nadam Time": nadam_time,
        "GD Losses": losses_gd,
        "SGD Losses": losses_sgd,
        "RMSProp Losses": losses_rmsprop,
        "Adam Losses": losses_adam,
        "Nadam Losses": losses_nadam,
        "GD Grad Norms": grad_norms_gd,
        "SGD Grad Norms": grad_norms_sgd,
        "RMSProp Grad Norms": grad_norms_rmsprop,
        "Adam Grad Norms": grad_norms_adam,
        "Nadam Grad Norms": grad_norms_nadam,
    }
]

df_results = pd.DataFrame(results)

print("Execution Times:")
print(df_results[["Iterations", "GD Time", "SGD Time", "RMSProp Time", "Adam Time", "Nadam Time"]])
print("\nLosses and Gradient Norms:")
print(df_results[["Iterations", "GD Losses", "SGD Losses", "RMSProp Losses", "Adam Losses", "Nadam Losses"]])
print("\nGradient Norms:")
print(df_results[["Iterations", "GD Grad Norms", "SGD Grad Norms", "RMSProp Grad Norms", "Adam Grad Norms", "Nadam Grad Norms"]])

# 5. Будуємо графіки для loss та норми градієнта для кожного з методів окремо, щоб визначити оптимальну кількість ітерацій.

for method in ["GD", "SGD", "RMSProp", "Adam", "Nadam"]:
    row = df_results.iloc[0]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(row[f"{method} Losses"], label=f"{ITERATIONS} Iterations")
    plt.title(f"{method} Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(row[f"{method} Grad Norms"], label=f"{ITERATIONS} Iterations")
    plt.title(f"{method} Gradient Norm over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.legend()

    plt.tight_layout()
    plt.show()
