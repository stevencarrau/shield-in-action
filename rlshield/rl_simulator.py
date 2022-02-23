import random

import numpy as np
import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp
from tf_agents.drivers import py_driver
from tf_agents.metrics import py_metrics
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import reverb_replay_buffer,reverb_utils
from tf_agents.specs import tensor_spec
import tensorflow as tf
from tf_agents.utils import common
import tf_agents
from deep_policy import DeepAgent
import re
import tf_agents
import reverb

from model_simulator import SimulationExecutor
import logging
logger = logging.getLogger(__name__)

def collect_episode(env,policy,num_episodes,buffer,max_steps):
    driver = py_driver.PyDriver(
        env,
        tf_agents.policies.py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
        [buffer],
        max_episodes=num_episodes,max_steps=max_steps)
    initial_time_step = env.reset()
    driver.run(initial_time_step)
    #
    # episode_counter = 0
    # env.reset()
    # while episode_counter < num_episodes:
    #     time_step = env.current_time_step()
    #     action_step = policy.action(time_step)
    #     next_time_step = env.step(env.apply_action(action_step.action))
    #     traj = trajectory.from_transition(time_step,action_step,next_time_step)
    #     buffer.add_batch(traj)
    #     if next_time_step.step_type == ts.StepType.LAST:
    #         episode_counter += 1


def collect_step(env,agent,policy,buffer):
    time_step = env.current_time_step()
    actions = env._simulator.available_actions()
    safe_actions = env._shield.shielded_actions(range(len(actions)))
    action_step = policy.action(time_step)
    next_time_step = env.step(env.apply_action(action_step.action))
    traj = trajectory.from_transition(time_step,action_step,next_time_step)
    buffer.add_batch(traj)

def collect_data(env,agent,policy,buffer,steps):
    for i in range(steps):
        collect_step(env,agent,policy,buffer)

def compute_avg_return(env, agent,policy, num_episodes=10,max_steps=100):
    total_return = 0.0
    episodes = []
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0
        steps = 0
        while not time_step.is_last():
            actions = env._simulator.available_actions()
            safe_actions = env._shield.shielded_actions(range(len(actions)))
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward[0]
            steps += 1
            if steps >= max_steps:
                break
        total_return += episode_return
        episodes.append(episode_return.numpy())
    episode_rewards = np.array(episodes)
    avg_return = total_return / num_episodes  
    return avg_return.numpy(),np.quantile(episode_rewards,0.1),np.quantile(episode_rewards,0.3173),np.quantile(episode_rewards,0.6827),np.quantile(episode_rewards,0.9)

def record_track(recorder,executor,agent,policy,maxsteps):
    state = executor._simulator.restart()
    finished = False
    executor._shield.reset()
    recorder.start_path()
    recorder.record_state(state)
    recorder.record_belief(executor._shield.list_support())
    for n in range(maxsteps):
        actions = executor._simulator.available_actions()
        safe_actions = executor._shield.shielded_actions(range(len(actions)))
        logger.debug(f"Number of actions: {actions}. Safe action indices: {safe_actions}")
        time_step = executor.current_time_step()
        action_step = policy.action(time_step) #agent(policy,time_step, allowed_actions=safe_actions)
        action = executor.apply_action(action_step.action)
        executor._simulator.step(action)
        state = executor._simulator._report_state()
        executor._shield.track(action, executor._model.get_observation(state))
        assert state in executor._shield.list_support()

        recorder.record_available_actions(actions)
        recorder.record_allowed_actions(safe_actions)
        recorder.record_selected_action(action)
        recorder.record_state(state)
        recorder.record_belief(executor._shield.list_support())
        if executor._simulator.is_done():
            logger.info(f"Done after {n} steps!")
            finished = True
            break
    actions = executor._simulator.available_actions()
    safe_actions = executor._shield.shielded_actions(range(len(actions)))
    recorder.record_available_actions(actions)
    recorder.record_allowed_actions(safe_actions)
    recorder.end_path(finished)

class TF_Environment(SimulationExecutor):
    def __init__(self,model,shield,obs_length=1,valuations=False,obs_type='BELIEF_SUPPORT',maxsteps=100,goal_value=1000):
        super().__init__(model,shield)
        self.shield_on = True
        self.decay = 1
        self.obs_type = obs_type
        self.batch_size = 64
        self.goal_value = goal_value
        self.batched = False
        action_keywords = set()
        for s_i in range(self._model.nr_states):
            n_act = self._model.get_nr_available_actions(s_i)
            for a_i in range(n_act):
                action_keywords = action_keywords.union(self._model.choice_labeling.get_labels_of_choice(self._model.get_choice_index(s_i,a_i)))
        # action_spec_count = [self._model.get_nr_available_actions(i) for i in range(1,self._model.nr_states)]
        self.action_indices = dict([[j,i] for i,j in enumerate(action_keywords)])
        self.act_keywords = dict([[self.action_indices[i],i] for i in self.action_indices])
        self.nr_actions = len(action_keywords)
        self.valuations = valuations
        if valuations:
            self.keywords = self.get_observation_keywords()
            self.obs_length = len(self.keywords)
            obs_shape = np.array(self.observe()).shape
        else:
            self.obs_length = obs_length
            obs_shape = np.array(self.observe()).shape
        self.act_spec = tf_agents.specs.BoundedTensorSpec(dtype='int32', name='action', minimum=0, maximum=self.nr_actions - 1,shape=tf.TensorShape(()))
        self.disc_spec = tf_agents.specs.BoundedTensorSpec(name='discount', dtype='float32', minimum=0, maximum=1,shape=tf.TensorShape(()))
        self.obs_spec = {'obs': tf_agents.specs.TensorSpec(name='observation', dtype='int32', shape=tf.TensorShape(obs_shape)),'mask': tf_agents.specs.TensorSpec(shape=(self.nr_actions,), dtype='bool',name='mask')}
        self.rew_spec = tf_agents.specs.TensorSpec(name='reward', dtype='float32', shape=tf.TensorShape(()))
        self.step_spec = tf_agents.specs.TensorSpec(name='step_type', dtype='int32', shape=tf.TensorShape(()))
        self.time_step_spec = ts.TimeStep(discount=self.disc_spec, observation=self.obs_spec, reward=self.rew_spec, step_type=self.step_spec)
        replay_config = {'mem_size':obs_length,'obs_dims':self.obs_length}
        # self.replay_memory = ReplayMemory(replay_config)
        # self.replay_memory.initial_add(self.observe())
        self.step_count = 0
        self.episode_count = 0
        self.cost_ind = list(self._model.reward_models.keys()).index('costs')
        self.gain_ind = list(self._model.reward_models.keys()).index('gains')
        self.first = True
        self.maxsteps = maxsteps


    def restart(self):
        self._simulator.restart()
        self._shield.reset()
        # self.replay_memory.reset(self.observe())
        self.step_count = 0
        self.sink_flag = False
        self.episode_count += 1

    def reset(self):
        self.restart()
        # self._simulator.step(0)
        # self._shield.track(0,self._simulator._report_observation())
        return self.current_time_step()

    def is_done(self):
        # if self._model.is_sink_state(self._simulator._engine.get_current_state()) and not self.sink_flag:
        #     self.sink_flag = True
        #     return False
        return self._model.is_sink_state(self._simulator._engine.get_current_state()) or self.step_count==self.maxsteps

    def current_time_step(self,rew=None):
        if rew is not None:
            r = tf.constant(rew)
        else:
            r = tf.constant(self._simulator._report_rewards()[0]) if len(self._simulator._report_rewards()) != 0 else tf.constant(0.)
        discount = tf.constant(1.)
        actions = self._simulator.available_actions()
        if self.shield_on:
            safe_actions = self._shield.shielded_actions(range(len(actions)))
        else:
            # if np.random.random(1)[0] > self.decay:
            #     safe_actions = self._shield.shielded_actions(range(len(actions))) ## Shield on
            # else:
            safe_actions = actions

        if len(self._model.choice_labeling.get_labels_of_choice(self._model.get_choice_index(self._simulator._report_state(),0)))==0:
            mask_inds = [0]
        else:
            mask_inds = [self.action_indices[self._model.choice_labeling.get_labels_of_choice(self._model.get_choice_index(self._simulator._report_state(),a_i)).pop()] for a_i in safe_actions]
        mask = np.zeros(shape=(self.nr_actions,),dtype=bool)
        for i in mask_inds:
            mask[i] = True
        mask = tf.logical_and(tf.ones(shape=(self.nr_actions,),dtype=tf.bool),mask)
        # obs_in = []
        # for o_i in range(self.obs_length-1):
        #
        # print(self._simulator._report_state())
        # observation = {'obs':tf.constant([self.replay_memory.getObs()],dtype='int32'),'mask':mask}
        observation = {'obs':tf.constant(self.observe(),dtype='int32'),'mask':mask}
        if self.first:
            self.first = False
            return ts.TimeStep(reward=r, observation=observation, discount=discount,step_type=tf.constant(ts.StepType.FIRST))
        elif self.is_done():
            self.restart()
            # self.first=True
            return ts.TimeStep(reward=r,observation=observation,discount=discount,step_type=tf.constant(ts.StepType.LAST))
        else:
            return ts.TimeStep(reward=r, observation=observation,discount=discount, step_type=tf.constant(ts.StepType.MID))

    def step(self,action):
        try:
            act = self.apply_action(action)
        except:
            act = 0
            print('What')
        state, rew, labels = self._simulator.step(act)
        self._shield.track(act, self._simulator._report_observation())
        # obs = self.observe()
        self.step_count += 1
        if (self.is_done() and 'traps' in labels):
            rew[self.cost_ind] += self.goal_value if self.goal_value > 100 else 0
        elif (self.is_done() and 'goal' in labels):
            rew[self.gain_ind] += self.goal_value
        elif (self.is_done()):
            rew[self.cost_ind] += 0
        current_step = self.current_time_step(rew=self.cost_fn(rew))
        return current_step

    def observe(self):
        if self.valuations:
            return np.array(self.get_observation_valuation(),dtype=int)
        else:
            if self.obs_type == 'STATE_LEVEL':
                return [self._simulator._report_state()]
            elif self.obs_type == 'BELIEF_SUPPORT':
                support = np.zeros((self._model.nr_states,),dtype=int)
                for i in self._shield.list_support():
                    support[i] = 1
                return support.tolist()
            else:
                return [self._simulator._report_observation()]

    def cost_fn(self,rew_in):
        if len(rew_in)>1:
            return rew_in[self.gain_ind]-rew_in[self.cost_ind]
        else:
            return -rew_in[0]

    def get_observation_valuation(self):
        if self.obs_type == 'STATE_LEVEL':
            return [json_to_int(self._model.state_valuations.get_json(self._simulator._report_state())[i]) for i in self.keywords]
        else:
            return [json_to_int(self._model.observation_valuations.get_json(self._simulator._report_observation())[i]) for i in self.keywords]

    def get_observation_keywords(self):
        if self.obs_type == 'STATE_LEVEL':
            keywords = set([i[1:-2] for i in str(self._model.state_valuations.get_json(0)).split()[1:-1:2]])
        else:
            keywords = set([i[1:-2] for i in str(self._model.observation_valuations.get_json(0)).split()[1:-1:2]])
        # if 'start' in keywords:
        #     keywords.remove('start')
        # if 'turn' in keywords:
        #     keywords.remove('turn')
        return keywords

    def get_choice_labels(self):
        return [self._model.choice_labeling.get_labels_of_choice(self._model.get_choice_index(self._simulator._report_state(),a_i)).pop() for a_i in range(self._simulator.nr_available_actions())]

    def apply_action(self,act_in):
        act_keyword = self.act_keywords[int(act_in)]
        choice_list = self.get_choice_labels()
        return choice_list.index(act_keyword)


def simulate_deep_RL(env, recorder, total_nr_runs=5,eval_interval=1000,eval_episodes=10, maxsteps=30,eval_env=None,agent_arg='DQN'):
    gamma = 1.0
    alpha = 3e-2
    log_interval = 100
    initial_collect_steps = 100
    replay_buffer_max_length = 2000
    eval_interval = eval_interval
    train_sequence_length = 1
    RL_agent = DeepAgent(env,alpha,agent_arg)
    table_name = 'deep_rl_buffer'
    replay_buffer_signature = tensor_spec.from_spec(RL_agent.agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature
    )
    reverb_server = reverb.Server([table])
    if agent_arg == "REINFORCE":
        buffer = reverb_replay_buffer.ReverbReplayBuffer(
            RL_agent.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=None,
            local_server=reverb_server
        )
        rb_observer = reverb_utils.ReverbAddEpisodeObserver(
            buffer.py_client,
            table_name,
            replay_buffer_max_length
        )
        collect_driver = py_driver.PyDriver(
            env,
            tf_agents.policies.py_tf_eager_policy.PyTFEagerPolicy(
                RL_agent.agent.collect_policy, use_tf_function=True),
            [rb_observer],
            max_steps=maxsteps,max_episodes=1)
    else:
        buffer = reverb_replay_buffer.ReverbReplayBuffer(
            RL_agent.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=train_sequence_length + 1,
            local_server=reverb_server
        )
        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            buffer.py_client, table_name, sequence_length=train_sequence_length+1
        )
        collect_driver = py_driver.PyDriver(
            env,
            tf_agents.policies.py_tf_eager_policy.PyTFEagerPolicy(
                RL_agent.agent.collect_policy, use_tf_function=True),
            [rb_observer],
            max_steps=train_sequence_length+1, max_episodes=1)
        dataset = buffer.as_dataset(num_parallel_calls=1, sample_batch_size=64,
                                    num_steps=train_sequence_length + 1).prefetch(3)
        iterator = iter(dataset)
    rand_pol = tf_agents.policies.random_tf_policy.RandomTFPolicy(env.time_step_spec, env.act_spec,
                                                                  observation_and_action_constraint_splitter=RL_agent.observation_and_action_constraint_splitter)
    metric = py_metrics.AverageReturnMetric()
    py_driver.PyDriver(
        env,
        tf_agents.policies.py_tf_eager_policy.PyTFEagerPolicy(
            rand_pol, use_tf_function=True),
        [rb_observer,metric],
        max_steps=initial_collect_steps).run(env.reset())

    # avg_return = compute_avg_return(eval_env, RL_agent.agent,RL_agent.agent.policy,num_eval_episodes,max_steps=maxsteps)
    # if recorder:
    #     record_track(recorder, eval_env, RL_agent.agent, RL_agent.agent.policy, maxsteps)
    time_step = env.reset()
    # collect_data(env, RL_agent.agent,RL_agent.agent.collect_policy, buffer,steps =2)
    RL_agent.agent.train = common.function(RL_agent.agent.train)
    returns = [(0,)+metric.result()]
    # print(f'Random policy return: {compute_avg_return(eval_env,RL_agent.agent,rand_pol,10,max_steps=maxsteps)}')

    for _ in range(total_nr_runs):
        if agent_arg == 'REINFORCE':
            # time_step,_ = collect_driver.run(time_step)
            # experience, unused_info = next(iterator)
            collect_episode(env,RL_agent.agent.collect_policy,num_episodes=1,buffer=rb_observer,max_steps=maxsteps)
            iterator = iter(buffer.as_dataset(sample_batch_size=1))
            trajectories, _ = next(iterator)
            train_loss =  RL_agent.agent.train(experience=trajectories)
            buffer.clear()
        else:
            time_step ,_ = collect_driver.run(time_step)
            experience, unused_info = next(iterator)
            train_loss = RL_agent.agent.train(experience).loss
            # collect_data(env,RL_agent.agent,  RL_agent.agent.collect_policy, buffer, collect_steps_per_iteration)
            # experience, unused_info = next(iterator)
            # train_loss = RL_agent.agent.train(experience).loss



        step = int(RL_agent.agent.train_step_counter.numpy()/25) if agent_arg=='PPO' else RL_agent.agent.train_step_counter.numpy()
        # step = self.episode_count

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            # avg_return = compute_avg_return(eval_env, RL_agent.agent, RL_agent.agent.policy, num_eval_episodes,max_steps=maxsteps)
            print('step = {0}: Average Return = {1}'.format(step, metric.result()))
            returns.append((step,)+metric.result())

    if recorder:
        record_track(recorder,eval_env,RL_agent.agent,RL_agent.agent.policy,maxsteps)
    return returns


def json_to_int(i):
    try:
        return int(i)
    except:
        return 0 if i=='false' else 1
