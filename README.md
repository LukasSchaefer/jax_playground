# JAX Playground
This repository is a work-in-progress and serves to test out JAX as a acceleration/ DL framework.

## Hardware for Speedtests
Speedtests are conducted on a Ubuntu system with a Intel® Core™ i7-10750H CPU @ 2.60GHz, 16GB of RAM, and a GTX 1650Ti GPU (4GB VRAM).

## Value Iteration
![Image showing the execution time of value iteration on MDPs of varying number of states with numpy, torch, and jax.](value_iteration/value_iteration_speedtest.jpg "Value Iteration Speedtest")

![Image showing the execution time of value iteration on MDPs of varying number of states with torch and jax on GPU only.](value_iteration/value_iteration_speedtest_gpu_only.jpg "Value Iteration Speedtest (GPU only)")
