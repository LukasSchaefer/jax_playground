import time

import matplotlib.pyplot as plt

from value_iteration import generate_random_mdp, value_iteration_numpy, value_iteration_jax, value_iteration_torch


def plot_times(num_statess, np_times, jax_times, torch_cpu_times, torch_gpu_times):
    plt.plot(num_statess, np_times, label="Numpy")
    plt.plot(num_statess, torch_cpu_times, label="Torch (CPU)")
    plt.plot(num_statess, torch_gpu_times, label="Torch (CUDA)")
    plt.plot(num_statess, jax_times[0], label="Jax (CUDA, 0)")
    plt.plot(num_statess, jax_times[1], label="Jax (CUDA, 1)")
    plt.plot(num_statess, jax_times[2], label="Jax (CUDA, 2)")
    plt.xlabel("Number of States")
    plt.ylabel("Time (s)")
    plt.legend()
    # plt.yscale("log")
    plt.savefig("value_iteration_speedtest.pdf")


if __name__ == "__main__":
    num_statess = [10, 50, 100, 200, 400, 800, 1600]
    num_actions = 10

    np_times = []
    jax_times = [[] for _ in range(3)]
    torch_cpu_times = []
    torch_gpu_times = []

    for num_states in num_statess:
        print(f"Generating Random MDP with {num_states} states ...")
        states, actions, transition_function, reward_function = generate_random_mdp(num_states, num_actions)

        print("Numpy Value Iteration ...")
        start = time.time()
        policy, value = value_iteration_numpy(transition_function, reward_function)
        end = time.time()
        np_times.append(end - start)

        print("Torch (CPU) Value Iteration ...")
        start = time.time()
        policy, value = value_iteration_torch(transition_function, reward_function, device="cpu")
        end = time.time()
        torch_cpu_times.append(end - start)

        print("Torch (GPU) Value Iteration ...")
        start = time.time()
        policy, value = value_iteration_torch(transition_function, reward_function, device="cuda")
        end = time.time()
        torch_gpu_times.append(end - start)

        for round in range(3):
            print(f"Jax Value Iteration (round {round}) ...")
            start = time.time()
            policy, value = value_iteration_jax(transition_function, reward_function, gamma=0.9)
            end = time.time()
            jax_times[round].append(end - start)
        print()

    plot_times(num_statess, np_times, jax_times, torch_cpu_times, torch_gpu_times)
