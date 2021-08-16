import tensorflow as tf
import tf_agents
from tf_agents.agents import TFAgent
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import tf_policy
import numpy as np
from tf_agents.replay_buffers import episodic_replay_buffer
from itertools import chain
from tf_agents.networks import actor_distribution_network,actor_distribution_rnn_network,value_rnn_network,value_network



def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units,activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))

def obs_selector(args='DQN'):
    if args is 'DQN':
        return tf_agents.agents.dqn.dqn_agent.DqnAgent
    elif args is 'PPO':
        return tf_agents.agents.ppo.ppo_agent.PPOAgent
    elif args is 'DDPG':
        return tf_agents.agents.ddpg.ddpg_agent.DdpgAgent
    elif args is 'REINFORCE':
        return tf_agents.agents.reinforce.reinforce_agent.ReinforceAgent
    elif args is 'TD3':
        return tf_agents.agents.td3.td3_agent.Td3Agent
    elif args is 'SAC':
        return tf_agents.agents.sac.sac_agent.SacAgent




class DeepAgent():

    def __init__(self,env,alpha,agent_arg='PPO'):
        self.alpha = alpha
        self.learning_method(env,alpha,agent_arg)

    def observation_and_action_constraint_splitter(self,observation):
        return observation['obs'], observation['mask']

    def learning_method(self,env,alpha,agent_arg):
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        if agent_arg is 'DQN':
            layer_params = (100,)
            self.fc_layer_params = layer_params
            dense_layers = [dense_layer(num_units) for num_units in self.fc_layer_params]
            q_values_layer = tf.keras.layers.Dense(env.nr_actions, activation=None,
                                                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03,
                                                                                                          maxval=0.03),
                                                   bias_initializer=tf.keras.initializers.Constant(-0.2))
            q_net = tf_agents.networks.sequential.Sequential(dense_layers + [q_values_layer])
            self.agent = obs_selector(agent_arg)(env.time_step_spec, env.act_spec, q_network=q_net, optimizer=optimizer,
                             td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss,
                             train_step_counter=train_step_counter,
                             observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)
        elif agent_arg is 'PPO':
            actor_fc_layers=(200, 100)
            value_fc_layers=(200, 100)
            self.fc_layer_params = actor_fc_layers
            use_rnns=False
            lstm_size=(20,)
            if use_rnns:
                if type(env.obs_spec) is dict:
                    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                        env.obs_spec['obs'],
                        env.action_spec(),
                        input_fc_layer_params=actor_fc_layers,
                        output_fc_layer_params=None,
                        lstm_size=lstm_size)
                    value_net = value_rnn_network.ValueRnnNetwork(
                        env.obs_spec['obs'],
                        input_fc_layer_params=value_fc_layers,
                        output_fc_layer_params=None)
                else:
                    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                        env.obs_spec,
                        env.act_spec,
                        input_fc_layer_params=actor_fc_layers,
                        output_fc_layer_params=None,
                        lstm_size=lstm_size)
                    value_net = value_rnn_network.ValueRnnNetwork(
                        env.obs_spec,
                        input_fc_layer_params=value_fc_layers,
                        output_fc_layer_params=None)

            else:
                if type(env.obs_spec) is dict:
                    actor_net =  actor_distribution_network.ActorDistributionNetwork(
                        env.obs_spec['obs'],
                        env.act_spec,
                        fc_layer_params=actor_fc_layers)
                    value_net = value_network.ValueNetwork(
                        env.obs_spec['obs'],
                        fc_layer_params=value_fc_layers)
                else:
                    actor_net =  actor_distribution_network.ActorDistributionNetwork(
                        env.obs_spec,
                        env.act_spec,
                        fc_layer_params=actor_fc_layers)
                    value_net = value_network.ValueNetworkk(
                        env.obs_spec,
                        fc_layer_params=value_fc_layers)
            self.agent = obs_selector(agent_arg)(
                env.time_step_spec,
                env.act_spec,
                optimizer,
                actor_net=actor_net,
                value_net=value_net,
                entropy_regularization=0.0,
                importance_ratio_clipping=0.2,
                normalize_observations=False,
                normalize_rewards=False,
                use_gae=True,
                observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
                train_step_counter=train_step_counter)
        elif agent_arg is 'DDPG':
            return tf_agents.agents.ddpg.ddpg_agent.DdpgAgent
        elif agent_arg is 'REINFORCE':
            return tf_agents.agents.reinforce.reinforce_agent.ReinforceAgent
        elif agent_arg is 'TD3':
            return tf_agents.agents.td3.td3_agent.Td3Agent
        elif agent_arg is 'SAC':
            return tf_agents.agents.sac.sac_agent.SacAgent


class ReplayMemory:
    def __init__(self, config):
        self.config = config
        self.actions = np.empty((self.config['mem_size']-1), dtype=np.int32)
        self.rewards = np.empty((self.config['mem_size']-1), dtype=np.int32)
        self.observations = np.empty((self.config['mem_size'], self.config['obs_dims']), dtype=np.int32)
        self.count = 0
        self.current = 0

    def reset(self,obs):
        self.actions = np.empty((self.config['mem_size'] - 1), dtype=np.int32)
        self.rewards = np.empty((self.config['mem_size'] - 1), dtype=np.int32)
        self.observations = np.empty((self.config['mem_size'], self.config['obs_dims']), dtype=np.int32)
        self.count = 0
        self.current = 0
        self.initial_add(obs)

    def initial_add(self,obs):
        self.observations[-1,:] = obs

    def add(self, action, reward,next_obs):
        if self.config['mem_size']>1:
            self.count = max(self.count, self.current + 1)
            for i in range(self.count - 1):
                self.actions[i] = self.actions[i]
                self.rewards[i] = self.rewards[i]
                self.observations[i] = self.observations[i + 1]
            self.observations[-1,:] = next_obs
            self.actions[-1] = action
            self.rewards[-1] = reward
            self.current = (self.current + 1) % self.config['mem_size']
        else:
            self.observations[-1, :] = next_obs

    def getObs(self):
        return self.observations.reshape(-1)

    def getSeq(self):
        seq_out = list(chain.from_iterable(zip(self.observations[:-1],self.actions)))
        seq_out.append(self.observations[-1])
        return np.array(seq_out).reshape(-1)

    def getRewSeq(self):
        seq_out = list(chain.from_iterable(zip(self.observations[:-1], self.actions,self.rewards)))
        seq_out.append(self.observations[-1])
        return np.array(seq_out).reshape(-1)
    # def __call__(self,policy, obs, allowed_actions=None):
    #     act_logits,_ = policy.distribution(obs)
    #     if allowed_actions:
    #         mask = np.zeros_like(act_logits.numpy(), dtype=bool)
    #         for i in allowed_actions:
    #             mask[0, i] = True
    #         new_logits = tf_agents.distributions.masked.MaskedCategorical(logits=act_logits, mask=mask)
    #         action = tf.random.categorical(logits=new_logits.logits, num_samples=1, dtype=None, seed=None,
    #                                            name=None).numpy()[0, 0]
    #     else:
    #         action = tf.random.categorical(logits=act_logits, num_samples=1, dtype=None, seed=None,
    #                                            name=None).numpy()[0, 0]
    #     return tf_agents.trajectories.policy_step.PolicyStep(action=tf.constant([action]))


