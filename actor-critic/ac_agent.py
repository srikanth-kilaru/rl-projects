#!/usr/bin/env python

"""
#########################################
# Srikanth Kilaru
# Fall 2018
# MS Robotics, Northwestern University
# Evanston, IL
# srikanthkilaru2018@u.northwestern.edu
# Adapted from UC Berkeley CS294-112 Fall 2018
##########################################
"""
import numpy as np
import tensorflow as tf
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process
import argparse
import ros_env
import sys
from shutil import copyfile
import yaml

# Utility functions
def normalize(values, mean=0.0, std=1.0):
    std_away = (values - np.mean(values))/(np.std(values) + 1e-8)
    return mean + std * std_away


def build_mlp(input_placeholder, output_size, scope,
              n_layers, size,
              activation=tf.tanh,
              output_activation=None):

    with tf.variable_scope(scope):
        #Input layer
        layer = tf.layers.dense(input_placeholder, size,
                                activation=activation,
                                use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        for n in range(n_layers-1):
            layer = tf.layers.dense(layer, size, activation=activation,
                                    use_bias=True,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        # Output fully connected layer 
        output_placeholder = tf.layers.dense(layer, output_size,
                                             activation=output_activation,
                                             use_bias=True,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        
    return output_placeholder


def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)

#============================================================#
# Agent running Actor Critic Algorithm
#============================================================#

class Agent(object):
    def __init__(self, path):
        super(Agent, self).__init__()

        stream = open(path + "/init.yaml", "r")
        config = yaml.load(stream)
        stream.close()

        self.gamma = config['gamma']
        self.n_iter = config['n_iter']
        self.learning_rate = config['learning_rate']
        self.reward_to_go = config['reward_to_go']
        self.normalize_advantages = config['normalize_advantages']
        self.seed = config['seed']
        self.nn_baseline = config['nn_baseline']
        self.n_layers = config['n_layers']
        self.size = config['size']
        self.ob_dim = config['goal_obs_dim'] + config['jnt_obs_dim']
        self.ac_dim = config['act_dim']
        self.max_path_length = config['max_path_length']
        self.min_timesteps_per_batch = config['min_timesteps_per_batch']
        self.num_target_updates = config['num_target_updates']
        self.num_grad_steps_per_target_update = config['num_grad_steps_per_target_update']
        
        print("gamma = ", self.gamma)
        print("n_iter = ", self.n_iter)
        print("learning_rate = ", self.learning_rate)
        print("reward to go = ", self.reward_to_go)
        print("normalize advantages = ", self.normalize_advantages)
        print("seed = ", self.seed)
        print("nn_baseline = ", self.nn_baseline)
        print("n_layers = ", self.n_layers)
        print("size = ", self.size)        
        print("max_path_length = ", self.max_path_length)
        print("min_timesteps_per_batch = ", self.min_timesteps_per_batch)
        print("num_target_updates = ", self.num_target_updates)
        print("num_grad_steps_per_target_update = ", self.num_grad_steps_per_target_update)
        print("ob_dim = ", self.ob_dim)
        print("ac_dim = ", self.ac_dim)


    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards, next_obs, terminals = [], [], [], [], []
            steps = 0
            while True:
                obs.append(ob)
                ac = self.sess.run(self.sy_sampled_ac,
                                   feed_dict={self.sy_ob_no: ob[None]}) 
                ac = ac[0]
                acs.append(ac)
                ob, rew, done = env.step(ac)
                # add the observation after taking a step to next_obs
                next_obs.append(ob)
                rewards.append(rew)
                steps += 1
                # If the episode ended, the corresponding terminal value is 1
                # otherwise, it is 0
                if done or steps > self.max_path_length:
                    terminals.append(1.0)
                    break
                else:
                    terminals.append(0.0)
            path = {"observation" : np.array(obs, dtype=np.float32), 
                    "reward" : np.array(rewards, dtype=np.float32), 
                    "action" : np.array(acs, dtype=np.float32),
                    "next_observation": np.array(next_obs, dtype=np.float32),
                    "terminal": np.array(terminals, dtype=np.float32)}
        
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch


# Function where the agent initialization, MLP setup and training is done
def train_AC(logdir, path):

    start = time.time()

    # Initialize the ROS/Sim Environment
    env = ros_env.Env(path, train_mode=True)
    
    # initialize the ROS agent
    agent = Agent(path)

    # Set Up Logger
    setup_logger(logdir, locals())
    
    # Set random seeds
    tf.set_random_seed(agent.seed)
    np.random.seed(agent.seed)

    # build computation graph
    agent.sy_ob_no = tf.placeholder(shape=[None, agent.ob_dim], name="ob",
                                    dtype=tf.float32)
    
    agent.sy_ac_na = tf.placeholder(shape=[None, agent.ac_dim], name="ac",
                                    dtype=tf.float32) 
    
    agent.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) 
    
    # The policy takes in an observation and produces a distribution over
    # the action space

    agent.sy_mean = build_mlp(agent.sy_ob_no, agent.ac_dim,
                              "policy_forward_pass",
                              n_layers=agent.n_layers,
                              size=agent.size)
    agent.sy_logstd = tf.Variable(tf.zeros([1, agent.ac_dim], name="logstd"))

    
    # We can sample actions from this action distribution.

    agent.sy_sampled_ac = agent.sy_mean + tf.exp(agent.sy_logstd) * tf.random_normal(tf.shape(agent.sy_mean))
    
    # We can also compute the logprob of the actions that were actually taken by the policy
    # This is used in the loss function.
    temp = (agent.sy_ac_na - agent.sy_mean) / tf.exp(agent.sy_logstd)
    agent.sy_logprob_n = - 0.5 * tf.reduce_sum(tf.square(temp), axis=1)
    
    actor_loss = tf.reduce_sum(-agent.sy_logprob_n * agent.sy_adv_n)
    agent.actor_update_op = tf.train.AdamOptimizer(agent.learning_rate).minimize(actor_loss)
    
    # define the critic
    agent.critic_prediction = tf.squeeze(build_mlp(agent.sy_ob_no,
                                                   1,
                                                   "nn_critic",
                                                   n_layers=agent.n_layers + 2,
                                                   size=agent.size + 32))
    agent.sy_target_n = tf.placeholder(shape=[None],
                                       name="critic_target",
                                       dtype=tf.float32)
    agent.critic_loss = tf.losses.mean_squared_error(agent.sy_target_n,
                                                     agent.critic_prediction)
    agent.critic_update_op = tf.train.AdamOptimizer(agent.learning_rate * 1.25).minimize(agent.critic_loss)
    
    
    # tensorflow: config, session, variable initialization
    
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True # may need if using GPU
    agent.sess = tf.Session(config=tf_config)
    agent.sess.__enter__() # equivalent to `with agent.sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #===============================#
    # Training Loop
    #===============================#

    total_timesteps = 0
    for itr in range(agent.n_iter):
        print("********** Iteration %i ************"%itr)
        itr_mesg = "Iteration started at "
        itr_mesg += time.strftime("%d-%m-%Y_%H-%M-%S")
        print(itr_mesg)
        
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation,
        # action for the policy gradient update by concatenating across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])

        # Call tensorflow operations to:
        # (1) update the critic

        # Use a bootstrapped target values to update the critic
        # Compute the target values r(s, a) + gamma*V(s') by calling the critic to compute V(s')
        # In total, take n=agent.num_grad_steps_per_target_update*agent.num_target_updates gradient update steps
        # Every agent.num_grad_steps_per_target_update steps, recompute the target values
        # by evaluating V(s') on the updated critic
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing the target
        # otherwise the values will grow without bound.
        for _ in range(agent.num_target_updates):
            v_sp = agent.sess.run(agent.critic_prediction, feed_dict={agent.sy_ob_no: next_ob_no})
            v_sp = (1 - terminal_n) * v_sp
            target = re_n + agent.gamma * v_sp
            
            for _ in range(agent.num_grad_steps_per_target_update):
                agent.sess.run(agent.critic_update_op, feed_dict={agent.sy_ob_no: ob_no, agent.sy_target_n: target})
                
        # (2) use the updated critic to compute the advantage
        # First, estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # To get the advantage, subtract the V(s) to get A(s, a) = Q(s, a) - V(s)
        # This requires calling the critic twice --- to obtain V(s') when calculating Q(s, a),
        # and V(s) when subtracting the baseline
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing Q(s, a)
        # otherwise the values will grow without bound.
        
        v_s = agent.sess.run(agent.critic_prediction, feed_dict={agent.sy_ob_no: ob_no})
        v_sp1 = agent.sess.run(agent.critic_prediction, feed_dict={agent.sy_ob_no: next_ob_no})
        v_sp1 = (1 - terminal_n) * v_sp1
        
        adv_n = re_n + agent.gamma * v_sp1 - v_s

        if agent.normalize_advantages:
            adv_n = normalize(adv_n)
            

        # (3) use the estimated advantage values to update the actor

        agent.sess.run(agent.actor_update_op,
                       feed_dict={agent.sy_ob_no: ob_no,
                                  agent.sy_ac_na: ac_na, agent.sy_adv_n: adv_n})
        
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()

        model_file = os.path.join(logdir, "model.ckpt")
        save_path = saver.save(agent.sess, model_file)
        print("Model saved in file: %s" % save_path)
        
    env.close_env_log()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    # path where the python script for agent and env reside
    path = args.path

    dpath = path + '/data'
    if not(os.path.exists(dpath)):
        os.makedirs(dpath)
    logdir = "AC" + '_' + time.strftime("%d-%m-%Y_%H-%M")
    logdir = os.path.join(dpath, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    fname = os.path.basename(sys.argv[0])
    copyfile(path + fname, logdir + "/" + fname)

    train_AC(logdir, path)
    
if __name__ == "__main__":
    main()
