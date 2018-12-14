# MSR Capstone Project - Policy Gradient Implementation on Sawyer
Srikanth Kilaru

Northwestern University (Fall 2018)

## Overview
This README covers the software implementation aspects of my capstone project.
Please see my [portfolio page](https://srikanth-kilaru.github.io/projects/2018/final-proj-RL) for more details about the project.
The source code for the PG agent implementation is based upon the skeleton code provided as part of the [UC Berkeley course on RL taught in Fall 2018](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf)

## Software Design
The main parts of the software implementation are -

1.> Agent code (in pg_agent.py), where the policy is initialized, iteratively improved, new trajectories are rolled out, and advantage normalization is done
2.> Environment code (in ros_env.py) where the observations are gathered, rewards are calculated and new actions are implemented
3.> Test code where the learnt policy is executed either unti lthe goal is reached or for a fixed number of times (typically the length of the trajectory used during training)
4.> Initialization parameters (in init.yaml file) where different hyper-parameters such as trajectory length, batch size, gamma, learning rate, size of neural network, and environment settings like joints to be manipulated while training, goal locations, velocity or torque mode of control etc. are defined
5.> During run time the logs and learnt policy is written to a new timestamped directory

The interface between the agent and the Sawyer ROS environment is exactly the same as the OpenAI Gym environment. New actions generated by the policy are executed in the environment by calling the 'step()' function which returns, new observations, rewards and the done flag, and at the beginning of every new trajectory, the 'reset()' function is invoked where Sawyer returns to initial condition and a new goal from the list specified is chosen.

A simple simulator is also implemented in the ros_env.py file which can be used by specifying the --sim flag during training and testing. NOTE: The simulator mode works only for velocity control mode.

During training the init.yaml file is copied to the logging and policy directory so that testing can benefit from the same initialization values.
This code should be easily portable to other robots and the environment code can easily be front ended with other RL algorithms.