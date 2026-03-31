# завантажити середовище FrozenLake-v1 з бібліотеки Open AI Gym.

# Для виконання завдання необхідно виконати такі кроки:

# 1. Завантажити середовище за допомогою gym.make('FrozenLake-v1').
# 2. Використовуючи опис функції compute_value_function() в конспекті у розділі Ітерації за політиками запрограмувати її повний вигляд.
# 3. Використовуючи опис функції policy_iteration() в конспекті у розділі Ітерації за політиками запрограмувати її повний вигляд.
# 4. Візуалізувати отриману оптимальну політику за допомогою функції show_render() з конспекта.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def compute_value_function(env, policy, gamma=0.99, theta=1e-6):
    transitions = env.unwrapped.P  # Отримуємо інформацію про переходи з середовища
    value_function = np.zeros(env.observation_space.n)  # Ініціалізуємо вектор значень для всіх станів нулями
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for transition_probability, next_state, reward, _ in transitions[s][a]:
                    v += evaluate_action_contribution(gamma, value_function, action_prob, transition_probability, next_state, reward)
            delta = max(delta, abs(v - value_function[s]))
            value_function[s] = v

        if delta < theta:  # Якщо максимальна зміна в значеннях менша за поріг, зупиняємо ітерації
            break

    return value_function


def evaluate_action_contribution(gamma, value_function, action_prob, transition_probability, next_state, reward):
    return action_prob * transition_probability * (reward + gamma * value_function[next_state])


def policy_iteration(env, gamma=0.99, theta=1e-6):
    transitions = env.unwrapped.P  # Отримуємо інформацію про переходи з середовища
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n  # Ініціалізуємо політику рівномірно для всіх станів

    while True:
        value_function = compute_value_function(env, policy, gamma, theta)
        policy_stable = True
        for s in range(env.observation_space.n):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for transition_probability, next_state, reward, _ in transitions[s][a]:
                    action_values[a] += evaluate_action_contribution(gamma, value_function, 1, transition_probability, next_state, reward)
            best_action = np.argmax(action_values)  # Визначаємо найкращу дію для стану s, вибираючи дію з максимальним значенням
            if old_action != best_action:  # Якщо найкраща дія відрізняється від поточної, оновлюємо політику та позначаємо її як нестабільну
                policy_stable = False
            policy[s] = np.eye(env.action_space.n)[best_action]  # Оновлюємо політику для стану s, встановлюючи ймовірність 1 для найкращої дії та 0 для інших
        if policy_stable:
            break
    return policy


def show_render_plt(img):
    if isinstance(img, np.ndarray):
        plt.imshow(img)
        plt.axis("off")  # Optional: hides the axis ticks for render mode "rgb_array"
        plt.show()

    if isinstance(img, str):  # for render_mode="text"
        print(img)


def show_render(env, policy):
    state, _ = env.reset()
    show_render_plt(env.render())

    done = False
    while not done:
        action = int(np.argmax(policy[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        show_render_plt(env.render())

    print(f"Episode finished with reward: {reward}")


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    optimal_policy = policy_iteration(env)
    show_render(env, optimal_policy)
    env.close()
