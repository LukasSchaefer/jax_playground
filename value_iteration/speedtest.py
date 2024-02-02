import time

import matplotlib.pyplot as plt

from value_iteration import generate_random_mdp, value_iteration_numpy, value_iteration_jax, value_iteration_torch

ROUNDS = 3


def plot_times(
        num_statess,
        np_times=None,
        jax_times=None,
        torch_cpu_times=None,
        torch_gpu_times=None,
        log_scale=False,
        save_name="value_iteration_speedtest.jpg",
):
    plt.figure()
    if np_times is not None:
        plt.plot(num_statess, np_times, label="Numpy")
    if jax_times is not None:
        for r in range(ROUNDS):
            plt.plot(num_statess, jax_times[r], label=f"Jax (CUDA, {r+1})")
    if torch_cpu_times is not None:
        for r in range(ROUNDS):
            plt.plot(num_statess, torch_cpu_times[r], label=f"Torch (CPU, {r+1})")
    if torch_gpu_times is not None:
        for r in range(ROUNDS):
            plt.plot(num_statess, torch_gpu_times[r], label=f"Torch (CUDA, {r+1})")
    plt.xlabel("Number of States")
    plt.ylabel("Time (s)")
    if log_scale:
        plt.yscale("log")
    plt.legend()
    # plt.yscale("log")
    plt.savefig(save_name)


if __name__ == "__main__":
    num_statess = [10, 50, 100, 200, 400, 800, 1600]
    num_actions = 10

    np_times = []
    jax_times = [[] for _ in range(ROUNDS)]
    torch_cpu_times = [[] for _ in range(ROUNDS)]
    torch_gpu_times = [[] for _ in range(ROUNDS)]

    for num_states in num_statess:
        print(f"Generating Random MDP with {num_states} states ...")
        states, actions, transition_function, reward_function = generate_random_mdp(num_states, num_actions)

        print("Numpy Value Iteration ...")
        start = time.time()
        policy, value = value_iteration_numpy(transition_function, reward_function)
        end = time.time()
        np_times.append(end - start)

        for r in range(ROUNDS):
            print(f"Torch (CPU) Value Iteration (round {r+1}) ...")
            start = time.time()
            policy, value = value_iteration_torch(transition_function, reward_function, device="cpu")
            end = time.time()
            torch_cpu_times[r].append(end - start)

        for r in range(ROUNDS):
            print(f"Torch (GPU) Value Iteration (round {r+1}) ...")
            start = time.time()
            policy, value = value_iteration_torch(transition_function, reward_function, device="cuda")
            end = time.time()
            torch_gpu_times[r].append(end - start)

        for r in range(ROUNDS):
            print(f"Jax Value Iteration (round {r+1}) ...")
            start = time.time()
            policy, value = value_iteration_jax(transition_function, reward_function, gamma=0.9)
            end = time.time()
            jax_times[r].append(end - start)
        print()

    plot_times(num_statess, np_times, jax_times, torch_cpu_times, torch_gpu_times)
    plot_times(num_statess, np_times, jax_times, torch_cpu_times, torch_gpu_times, log_scale=True, save_name="value_iteration_speedtest_log.jpg")
    plot_times(num_statess, None, jax_times, torch_cpu_times, torch_gpu_times, save_name="value_iteration_speedtest_no_numpy.jpg")
    plot_times(num_statess, None, jax_times, None, torch_gpu_times, save_name="value_iteration_speedtest_gpu_only.jpg")
