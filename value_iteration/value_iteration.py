from functools import partial
import time

import numpy as np
import jax
from jax import numpy as jnp
from jax import jit, random, vmap
import torch


GAMMA = 0.9
THETA = 1e-15


def generate_random_mdp(num_states, num_actions):
    """Generates a random MDP with the specified number of states and actions

    :param num_states (int): number of states
    :param num_actions (int): number of actions
    :return: state_names (list of str),
             action_names (list of str),
             P (np.array of shape (num_states, num_actions, num_states)),
             R (np.array of shape (num_states, num_actions, num_states)),
    """
    states = ["s" + str(i) for i in range(num_states)]
    actions = ["a" + str(i) for i in range(num_actions)]

    transitions = np.random.rand(num_states, num_actions, num_states)
    transitions /= transitions.sum(axis=2, keepdims=True)
    rewards = np.random.rand(num_states, num_actions, num_states)
    return states, actions, transitions, rewards


def value_iteration_numpy(transitions, rewards, gamma=GAMMA, theta=THETA):
    """
    Performs value iteration on the given MDP.

    Args:
        transitions: transition function as a numpy array of shape (num_states, num_actions, num_states)
        rewards: reward function as a numpy array of shape (num_states, num_actions)
        gamma: discount factor
        theta: convergence threshold

    Returns:
        (np.ndarray, np.ndarray): (policy, value)
    """
    num_states, num_actions, _ = transitions.shape

    # Initialize value function
    value = np.zeros(num_states)

    delta = np.inf
    while delta >= theta:
        # Compute the value function
        value_new = np.max(np.sum(transitions * (rewards + gamma * value), axis=-1), axis=-1)
        delta = np.max(np.abs(value_new - value))
        value = value_new

    # Compute the policy
    policy = np.argmax(np.sum(transitions * (rewards + gamma * value), axis=-1), axis=-1)

    return policy, value


def value_iteration_torch(transitions, rewards, device, gamma=GAMMA, theta=THETA):
    """
    Performs value iteration on the given MDP.

    Args:
        transitions: transition function as a numpy array of shape (num_states, num_actions, num_states)
        rewards: reward function as a numpy array of shape (num_states, num_actions)
        device: device to use for computation
        gamma: discount factor
        theta: convergence threshold

    Returns:
        (np.ndarray, np.ndarray): (policy, value)
    """
    num_states, num_actions, _ = transitions.shape

    # convert to torch tensors
    transitions = torch.tensor(transitions, device=device)
    rewards = torch.tensor(rewards, device=device)

    # Initialize value function
    value = torch.zeros(num_states, device=device)

    delta = float("inf")
    while delta >= theta:
        # Compute the value function
        value_new = torch.max(torch.sum(transitions * (rewards + gamma * value), dim=-1), dim=-1)[0]
        delta = torch.max((value_new - value).abs())
        value = value_new

    # Compute the policy
    policy = torch.argmax(torch.sum(transitions * (rewards + gamma * value), dim=-1), dim=-1)

    return policy.detach().cpu().numpy(), value.detach().cpu().numpy()


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None, None))
def _compute_value(transitions, rewards, value, gamma):
    return jnp.max(np.sum(transitions * (rewards + gamma * value), axis=-1), axis=-1)


@jax.jit
def _compute_delta(value_new, value):
    return jnp.max(jnp.abs(value_new - value))


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None, None))
def _compute_policy(transitions, rewards, value, gamma):
    return jnp.argmax(jnp.sum(transitions * (rewards + gamma * value), axis=-1), axis=-1)


def value_iteration_jax(transitions, rewards, gamma=GAMMA, theta=THETA):
    """
    Performs value iteration on the given MDP.

    Args:
        transitions: transition function as a numpy array of shape (num_states, num_actions, num_states)
        rewards: reward function as a numpy array of shape (num_states, num_actions)
        gamma: discount factor
        theta: convergence threshold

    Returns:
        (np.ndarray, np.ndarray): (policy, value)
    """
    num_states, num_actions, _ = transitions.shape

    # convert to jax arrays
    transitions = jnp.array(transitions)
    rewards = jnp.array(rewards)

    # Initialize value function
    value = jnp.zeros(num_states)

    delta = np.inf
    while delta >= theta:
        # Compute the value function
        value_new = _compute_value(transitions, rewards, value, gamma)
        delta = _compute_delta(value_new, value)
        value = value_new

    # Compute the policy
    policy = _compute_policy(transitions, rewards, value, gamma)

    return policy, value


if __name__ == "__main__":
    print("Generating Random MDP ...")
    num_states = int(input("Number of states: "))
    num_actions = int(input("Number of actions: "))
    states, actions, transition_function, reward_function = generate_random_mdp(num_states, num_actions)
    print()

    print("Value Iteration (Numpy)")
    start = time.time()
    policy, value = value_iteration_numpy(transition_function, reward_function)
    end = time.time()
    print(f"Time taken: {end - start:.3f}s")
    print(f"Policy: {policy}")
    print(f"Value: {value}")
    print()
    print("----------------------------------------")
    print()

    print("Value Iteration (Torch CPU)")
    start = time.time()
    policy, value = value_iteration_torch(transition_function, reward_function, device="cpu")
    end = time.time()
    print(f"Time taken: {end - start:.3f}s")
    print(f"Policy: {policy}")
    print(f"Value: {value}")
    print()

    print("Value Iteration (Torch CUDA)")
    start = time.time()
    policy, value = value_iteration_torch(transition_function, reward_function, device="cuda")
    end = time.time()
    print(f"Time taken: {end - start:.3f}s")
    print(f"Policy: {policy}")
    print(f"Value: {value}")
    print()
    print("----------------------------------------")
    print()

    print("Value Iteration (Jax)")
    for round in range(3):
        print(f"\tRound {round}:")
        start = time.time()
        policy, value = value_iteration_jax(transition_function, reward_function, gamma=0.9)
        end = time.time()
        print(f"\tTime taken: {end - start:.3f}s")
        print(f"\tPolicy: {policy}")
        print(f"\tValue: {value}")
        print()
