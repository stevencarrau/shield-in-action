# discrete_sac_agent.py
"""
A Soft Actor-Critic Agent.

Implements the discrete version of Soft Actor-Critic (SAC) algorithm based on
"Discrete and Continuous Action Representation for Practical RL in Video Games" by Olivier Delalleau, Maxim Peter, Eloi Alonso, Adrien Logut (2020).
Paper: https://montreal.ubisoft.com/en/discrete-and-continuous-action-representation-for-practical-reinforcement-learning-in-video-games/
"""
# Using Type Annotations.
from __future__ import absolute_import, division, print_function

import collections
from typing import Callable, Optional, Text

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from six.moves import zip
from tf_agents.agents import data_converter, tf_agent
from tf_agents.networks import encoding_network, network, utils
from tf_agents.policies import actor_policy, tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common, eager_utils, nest_utils, object_identity


SacLossInfo = collections.namedtuple(
    'SacLossInfo', ('critic_loss', 'actor_loss', 'alpha_loss'))


@gin.configurable
class DiscreteSacAgent(tf_agent.TFAgent):
    """A SAC Agent that supports discrete action spaces."""

    def __init__(self,
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 critic_network: network.Network,
                 actor_network: network.Network,
                 actor_optimizer: types.Optimizer,
                 critic_optimizer: types.Optimizer,
                 alpha_optimizer: types.Optimizer,
                 actor_loss_weight: types.Float = 1.0,
                 critic_loss_weight: types.Float = 0.5,
                 alpha_loss_weight: types.Float = 1.0,
                 actor_policy_ctor: Callable[
                     ..., tf_policy.TFPolicy] = actor_policy.ActorPolicy,
                 critic_network_2: Optional[network.Network] = None,
                 target_critic_network: Optional[network.Network] = None,
                 target_critic_network_2: Optional[network.Network] = None,
                 target_update_tau: types.Float = 1.0,
                 target_update_period: types.Int = 1,
                 td_errors_loss_fn: types.LossFn = tf.math.squared_difference,
                 gamma: types.Float = 1.0,
                 reward_scale_factor: types.Float = 1.0,
                 initial_log_alpha: types.Float = 0.0,
                 use_log_alpha_in_alpha_loss: bool = True,
                 target_entropy: Optional[types.Float] = None,
                 gradient_clipping: Optional[types.Float] = None,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False,
                 train_step_counter: Optional[tf.Variable] = None,
                 observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
                 name: Optional[Text] = None):
        """Creates a SAC Agent.

        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions.
          critic_network: A function critic_network((observations, actions)) that
            returns the q_values for each observation and action.
          actor_network: A function actor_network(observation, action_spec) that
            returns action distribution.
          actor_optimizer: The optimizer to use for the actor network.
          critic_optimizer: The default optimizer to use for the critic network.
          alpha_optimizer: The default optimizer to use for the alpha variable.
          actor_loss_weight: The weight on actor loss.
          critic_loss_weight: The weight on critic loss.
          alpha_loss_weight: The weight on alpha loss.
          actor_policy_ctor: The policy class to use.
          critic_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
            the second critic network during Q learning.  The weights from
            `critic_network` are copied if this is not provided.
          target_critic_network: (Optional.)  A `tf_agents.network.Network` to be
            used as the target critic network during Q learning. Every
            `target_update_period` train steps, the weights from `critic_network`
            are copied (possibly withsmoothing via `target_update_tau`) to `
            target_critic_network`.  If `target_critic_network` is not provided, it
            is created by making a copy of `critic_network`, which initializes a new
            network with the same structure and its own layers and weights.
            Performing a `Network.copy` does not work when the network instance
            already has trainable parameters (e.g., has already been built, or when
            the network is sharing layers with another).  In these cases, it is up
            to you to build a copy having weights that are not shared with the
            original `critic_network`, so that this can be used as a target network.
            If you provide a `target_critic_network` that shares any weights with
            `critic_network`, a warning will be logged but no exception is thrown.
          target_critic_network_2: (Optional.) Similar network as
            target_critic_network but for the critic_network_2. See documentation
            for target_critic_network. Will only be used if 'critic_network_2' is
            also specified.
          target_update_tau: Factor for soft update of the target networks.
          target_update_period: Period for soft update of the target networks.
          td_errors_loss_fn:  A function for computing the elementwise TD errors
            loss.
          gamma: A discount factor for future rewards.
          reward_scale_factor: Multiplicative scale for the reward.
          initial_log_alpha: Initial value for log_alpha.
          use_log_alpha_in_alpha_loss: A boolean, whether using log_alpha or alpha
            in alpha loss. Certain implementations of SAC use log_alpha as log
            values are generally nicer to work with.
          target_entropy: The target average policy entropy, for updating alpha. The
            default value is negative of the total number of actions.
          gradient_clipping: Norm length to clip gradients.
          debug_summaries: A bool to gather debug summaries.
          summarize_grads_and_vars: If True, gradient and network variable summaries
            will be written during training.
          train_step_counter: An optional counter to increment every time the train
            op is run.  Defaults to the global_step.
          name: The name of this agent. All variables in this module will fall under
            that name. Defaults to the class name.
        """
        tf.Module.__init__(self, name=name)
        self._observation_and_action_constraint_splitter = (observation_and_action_constraint_splitter)
        net_observation_spec = time_step_spec.observation
        if observation_and_action_constraint_splitter:
            net_observation_spec, _ = observation_and_action_constraint_splitter(net_observation_spec)
        flat_action_spec = tf.nest.flatten(action_spec)
        self._num_actions = np.sum([
            single_spec.maximum-single_spec.minimum+1
            for single_spec in flat_action_spec
        ])

        self._check_action_spec(action_spec)

        self._critic_network_1 = critic_network
        self._critic_network_1.create_variables(
            net_observation_spec)
        if target_critic_network:
            target_critic_network.create_variables(
                net_observation_spec)
        self._target_critic_network_1 = (
            common.maybe_copy_target_network_with_checks(self._critic_network_1,
                                                         target_critic_network,
                                                         'TargetCriticNetwork1'))

        if critic_network_2 is not None:
            self._critic_network_2 = critic_network_2
        else:
            self._critic_network_2 = critic_network.copy(name='CriticNetwork2')
            # Do not use target_critic_network_2 if critic_network_2 is None.
            target_critic_network_2 = None
        self._critic_network_2.create_variables(
            net_observation_spec)
        if target_critic_network_2:
            target_critic_network_2.create_variables(
                net_observation_spec)
        self._target_critic_network_2 = (
            common.maybe_copy_target_network_with_checks(self._critic_network_2,
                                                         target_critic_network_2,
                                                         'TargetCriticNetwork2'))

        if actor_network:
            actor_network.create_variables(net_observation_spec)
        self._actor_network = actor_network

        policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=False,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter
        )

        self._train_policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=True,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter
        )

        self._log_alpha = common.create_variable(
            'initial_log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)

        if target_entropy is None:
            target_entropy = self._get_default_target_entropy(action_spec)

        self._use_log_alpha_in_alpha_loss = use_log_alpha_in_alpha_loss
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._alpha_optimizer = alpha_optimizer
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._alpha_loss_weight = alpha_loss_weight
        self._td_errors_loss_fn = td_errors_loss_fn
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._target_entropy = target_entropy
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._update_target = self._get_target_updater(
            tau=self._target_update_tau, period=self._target_update_period)

        train_sequence_length = 2 if not critic_network.state_spec else None

        super().__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter
        )

        self._as_transition = data_converter.AsTransition(
            self.data_context, squeeze_time_dim=(train_sequence_length == 2))

    def _check_action_spec(self, action_spec):
        flat_action_spec = tf.nest.flatten(action_spec)
        for spec in flat_action_spec:
            if spec.dtype.is_floating:
                raise NotImplementedError(
                    'DiscreteSacAgent does not support continuous actions. '
                    'Action spec: {}'.format(action_spec))

    def _get_default_target_entropy(self, action_spec):
        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        # ratio=0.98 is thevalue used by Christodoulou, 2019 so we use this by default
        target_entropy = - np.log(1/self._num_actions) * 0.98
        return target_entropy

    def _actions_dist(self, time_steps):
        """Get actions distributions from policy."""
        # Get raw action distribution from policy, and initialize bijectors list.
        batch_size = nest_utils.get_outer_shape(
            time_steps, self._time_step_spec)[0]
        policy_state = self._train_policy.get_initial_state(batch_size)
        action_distribution = self._train_policy.distribution(
            time_steps, policy_state=policy_state).action

        return action_distribution

    def _initialize(self):
        """Returns an op to initialize the agent.

        Copies weights from the Q networks to the target Q network.
        """
        common.soft_variables_update(
            self._critic_network_1.variables,
            self._target_critic_network_1.variables,
            tau=1.0)
        common.soft_variables_update(
            self._critic_network_2.variables,
            self._target_critic_network_2.variables,
            tau=1.0)

    def _train(self, experience, weights):
        """Returns a train op to update the agent's networks.

        This method trains with the provided batched experience.

        Args:
          experience: A time-stacked trajectory object.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.

        Returns:
          A train_op.

        Raises:
          ValueError: If optimizers are None and no default value was provided to
            the constructor.
        """
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        trainable_critic_variables = list(object_identity.ObjectIdentitySet(
            self._critic_network_1.trainable_variables +
            self._critic_network_2.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_critic_variables, ('No trainable critic variables to '
                                                'optimize.')
            tape.watch(trainable_critic_variables)
            critic_loss = self._critic_loss_weight*self.critic_loss(
                time_steps,
                actions,
                next_time_steps,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)

        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
        self._apply_gradients(critic_grads, trainable_critic_variables,
                              self._critic_optimizer)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                               'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self._actor_loss_weight*self.actor_loss(
                time_steps, weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables,
                              self._actor_optimizer)

        alpha_variable = [self._log_alpha]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert alpha_variable, 'No alpha variable to optimize.'
            tape.watch(alpha_variable)
            alpha_loss = self._alpha_loss_weight*self.alpha_loss(
                time_steps, weights=weights)
        tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
        alpha_grads = tape.gradient(alpha_loss, alpha_variable)
        self._apply_gradients(alpha_grads, alpha_variable,
                              self._alpha_optimizer)

        total_loss = critic_loss + actor_loss + alpha_loss

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='critic_loss', data=critic_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='alpha_loss', data=alpha_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='total_loss', data=total_loss, step=self.train_step_counter)

        self.train_step_counter.assign_add(1)
        self._update_target()

        extra = SacLossInfo(
            critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _apply_gradients(self, gradients, variables, optimizer):
        # list(...) is required for Python3.
        grads_and_vars = list(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                             self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

        For each weight w_s in the original network, and its corresponding
        weight w_t in the target network, a soft update is:
        w_t = (1- tau) x w_t + tau x ws

        Args:
          tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
          period: Step interval at which the target network is updated.

        Returns:
          A callable that performs a soft update of the target network parameters.
        """
        with tf.name_scope('update_target'):

            def update():
                """Update target network."""
                critic_update_1 = common.soft_variables_update(
                    self._critic_network_1.variables,
                    self._target_critic_network_1.variables,
                    tau,
                    tau_non_trainable=1.0)

                critic_2_update_vars = common.deduped_network_variables(
                    self._critic_network_2, self._critic_network_1)

                target_critic_2_update_vars = common.deduped_network_variables(
                    self._target_critic_network_2, self._target_critic_network_1)

                critic_update_2 = common.soft_variables_update(
                    critic_2_update_vars,
                    target_critic_2_update_vars,
                    tau,
                    tau_non_trainable=1.0)

                return tf.group(critic_update_1, critic_update_2)

            return common.Periodically(update, period, 'update_targets')

    def critic_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    td_errors_loss_fn: types.LossFn,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: bool = False) -> types.Tensor:
        """Computes the critic loss for SAC training.

        Args:
            time_steps: A batch of timesteps.
            actions: A batch of actions.
            next_time_steps: A batch of next timesteps.
            td_errors_loss_fn: A function(td_targets, predictions) to compute
            elementwise (per-batch-entry) loss.
            gamma: Discount for future rewards.
            reward_scale_factor: Multiplicative factor to scale rewards.
            weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
            training: Whether this loss is being used for training.

        Returns:
            critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            nest_utils.assert_same_structure(actions, self.action_spec)
            nest_utils.assert_same_structure(
                time_steps, self.time_step_spec)
            nest_utils.assert_same_structure(
                next_time_steps, self.time_step_spec)

            alpha = tf.math.exp(self._log_alpha)
            next_dist = self._actions_dist(next_time_steps)

            target_q_values1, _ = self._target_critic_network_1(
                next_time_steps.observation['obs'], next_time_steps.step_type, training=False)
            target_q_values2, _ = self._target_critic_network_2(
                next_time_steps.observation['obs'], next_time_steps.step_type, training=False)
            v_approx_next_state = tf.minimum(
                target_q_values1, target_q_values2)

            next_probs = next_dist.probs_parameter()
            v_approx_next_state = tf.reduce_sum(
                v_approx_next_state * next_probs, axis=-1)  # (?, )
            v_approx_next_state += alpha * next_dist.entropy()  # (?, )

            discounts = next_time_steps.discount * \
                tf.constant(gamma, dtype=tf.float32)

            # Mask is 0.0 at end of each episode to restart cumulative sum
            #   end of each episode.
            episode_mask = common.get_episode_mask(next_time_steps)

            td_targets = tf.stop_gradient(
                reward_scale_factor * next_time_steps.reward + discounts * v_approx_next_state*episode_mask)  # (?, 1)

            pred_td_targets1, _ = self._critic_network_1(
                time_steps.observation['obs'], time_steps.step_type, training=training)
            pred_td_targets2, _ = self._critic_network_2(
                time_steps.observation['obs'], time_steps.step_type, training=training)

            # Actually selected Q-values (from the actions batch).
            temp_one_hot = tf.one_hot(actions, depth=self._num_actions,
                                      dtype=tf.float32)  # (?, nb_actions)

            pred_td_targets1 = tf.reduce_sum(
                pred_td_targets1 * temp_one_hot, axis=-1)  # (?, 1)
            pred_td_targets2 = tf.reduce_sum(
                pred_td_targets2 * temp_one_hot, axis=-1)  # (?, 1)

            critic_loss1 = td_errors_loss_fn(
                td_targets, pred_td_targets1)
            critic_loss2 = td_errors_loss_fn(
                td_targets, pred_td_targets2)
            critic_loss = critic_loss1 + critic_loss2  # (?, ) or (?, 1)

            if critic_loss.shape.rank > 1:
                # Sum over the time dimension.
                critic_loss = tf.reduce_sum(
                    critic_loss, axis=range(1, critic_loss.shape.rank))

            agg_loss = common.aggregate_losses(
                per_example_loss=critic_loss,
                sample_weight=weights,
                regularization_loss=(self._critic_network_1.losses +
                                     self._critic_network_2.losses))
            critic_loss = agg_loss.total_loss

            self._critic_loss_debug_summaries(td_targets, pred_td_targets1,
                                              pred_td_targets2)

            return critic_loss

    def actor_loss(self,
                   time_steps: ts.TimeStep,
                   weights: Optional[types.Tensor] = None) -> types.Tensor:
        """Computes the actor_loss for SAC training.

        Args:
        time_steps: A batch of timesteps.
        weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.

        Returns:
        actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            nest_utils.assert_same_structure(
                time_steps, self.time_step_spec)

            dist = self._actions_dist(time_steps)
            alpha = tf.exp(self._log_alpha)

            target_q_values1, _ = self._critic_network_1(
                time_steps.observation['obs'], time_steps.step_type, training=False)
            target_q_values2, _ = self._critic_network_2(
                time_steps.observation['obs'], time_steps.step_type, training=False)
            target_q_values = tf.minimum(
                target_q_values1, target_q_values2)  # (?, q_outputs)

            logits_q = tf.stop_gradient(
                target_q_values / alpha)
            dist_d_qs = tfp.distributions.Categorical(logits=logits_q)  # (?, )
            kl = tfp.distributions.kl_divergence(dist, dist_d_qs)  # (?, )
            actor_loss = alpha * kl  # (?, )

            if actor_loss.shape.rank > 1:
                # Sum over the time dimension.
                actor_loss = tf.reduce_sum(
                    actor_loss, axis=range(1, actor_loss.shape.rank))
            reg_loss = self._actor_network.losses if self._actor_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=actor_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)
            actor_loss = agg_loss.total_loss
            self._actor_loss_debug_summaries(actor_loss, dist,
                                             target_q_values, time_steps)

            return actor_loss

    def alpha_loss(self,
                   time_steps: ts.TimeStep,
                   weights: Optional[types.Tensor] = None) -> types.Tensor:
        """Computes the alpha_loss for EC-SAC training (discrete actions).

        Args:
          time_steps: A batch of timesteps.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.

        Returns:
          alpha_d_loss: A scalar alpha loss (discrete action).
        """
        with tf.name_scope('alpha_loss'):
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            # alpha_loss = alpha * (H - H_target)
            #            = -alpha * (pi - log(pi) + H_target) equivalent to Christodoulou, 2019
            dist = self._actions_dist(time_steps)
            entropy = dist.entropy()
            entropy_diff = tf.stop_gradient(
                entropy - self._target_entropy)
            if self._use_log_alpha_in_alpha_loss:
                alpha_loss = self._log_alpha*entropy_diff
            else:
                alpha_loss = tf.exp(self._log_alpha)*entropy_diff

            if alpha_loss.shape.rank > 1:
                # Sum over the time dimension.
                alpha_loss = tf.reduce_mean(
                    alpha_loss, axis=range(1, alpha_loss.shape.rank))

            agg_loss = common.aggregate_losses(
                per_example_loss=alpha_loss, sample_weight=weights)
            alpha_loss = agg_loss.total_loss

            self._alpha_loss_debug_summaries(
                alpha_loss, entropy_diff)

            return alpha_loss

    def _actor_loss_debug_summaries(self, actor_loss, dist,
                                    target_q_values, time_steps):
        if self._debug_summaries:
            common.generate_tensor_summaries('actor_loss', actor_loss,
                                             self.train_step_counter)
            try:
                tf.compat.v2.summary.histogram(
                    name='actions_log_prob_discrete',
                    data=dist.logits,
                    step=self.train_step_counter)
            except ValueError:
                pass  # Guard against internal SAC variants that do not directly
                # generate actions.

            common.generate_tensor_summaries('target_q_values', target_q_values,
                                             self.train_step_counter)
            common.generate_tensor_summaries('act_mode', dist.mode(),
                                             self.train_step_counter)
            try:
                common.generate_tensor_summaries('entropy_action',
                                                 dist.entropy(),
                                                 self.train_step_counter)
            except NotImplementedError:
                pass  # Some distributions do not have an analytic entropy.

    def _alpha_loss_debug_summaries(self, alpha_loss, entropy_diff):
        if self._debug_summaries:
            common.generate_tensor_summaries(f'alpha_loss', alpha_loss,
                                             self.train_step_counter)
            common.generate_tensor_summaries(f'entropy_diff', entropy_diff,
                                             self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name=f'log_alpha', data=self._log_alpha, step=self.train_step_counter)

    def _critic_loss_debug_summaries(self, td_targets, pred_td_targets1,
                                     pred_td_targets2):
        if self._debug_summaries:
            td_errors1 = td_targets - pred_td_targets1
            td_errors2 = td_targets - pred_td_targets2
            td_errors = tf.concat([td_errors1, td_errors2], axis=0)
            common.generate_tensor_summaries('td_errors', td_errors,
                                             self.train_step_counter)
            common.generate_tensor_summaries('td_targets', td_targets,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2,
                                             self.train_step_counter)

class DiscreteSacCriticNetwork(network.Network):
    """Creates a critic network."""

    def __init__(self,
                 input_tensor_spec,
                 observation_preprocessing_layers=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=(75, 40),
                 observation_dropout_layer_params=None,
                 action_fc_layer_params=None,
                 action_dropout_layer_params=None,
                 joint_fc_layer_params=(75, 40),
                 joint_dropout_layer_params=None,
                 activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 kernel_initializer=None,
                 last_kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='CriticNetwork'):
        """Creates an instance of `CriticNetwork`.

        Args:
           input_tensor_spec: A tuple of (observation, action) each a nest of
            `tensor_spec.TensorSpec` representing the inputs.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations.
            All of these layers must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them. Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
            This layer must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          observation_conv_layer_params: Optional list of convolution layer
            parameters for observations, where each item is a length-three tuple
            indicating (num_units, kernel_size, stride).
          observation_fc_layer_params: Optional list of fully connected parameters
            for observations, where each item is the number of units in the layer.
          observation_dropout_layer_params: Optional list of dropout layer
            parameters, each item is the fraction of input units to drop or a
            dictionary of parameters according to the keras.Dropout documentation.
            The additional parameter `permanent`, if set to True, allows to apply
            dropout at inference for approximated Bayesian inference. The dropout
            layers are interleaved with the fully connected layers; there is a
            dropout layer after each fully connected layer, except if the entry in
            the list is None. This list must have the same length of
            observation_fc_layer_params, or be None.
          action_fc_layer_params: Optional list of fully connected parameters for
            actions, where each item is the number of units in the layer.
          action_dropout_layer_params: Optional list of dropout layer parameters,
            each item is the fraction of input units to drop or a dictionary of
            parameters according to the keras.Dropout documentation. The additional
            parameter `permanent`, if set to True, allows to apply dropout at
            inference for approximated Bayesian inference. The dropout layers are
            interleaved with the fully connected layers; there is a dropout layer
            after each fully connected layer, except if the entry in the list is
            None. This list must have the same length of action_fc_layer_params, or
            be None.
          joint_fc_layer_params: Optional list of fully connected parameters after
            merging observations and actions, where each item is the number of units
            in the layer.
          joint_dropout_layer_params: Optional list of dropout layer parameters,
            each item is the fraction of input units to drop or a dictionary of
            parameters according to the keras.Dropout documentation. The additional
            parameter `permanent`, if set to True, allows to apply dropout at
            inference for approximated Bayesian inference. The dropout layers are
            interleaved with the fully connected layers; there is a dropout layer
            after each fully connected layer, except if the entry in the list is
            None. This list must have the same length of joint_fc_layer_params, or
            be None.
          activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
          output_activation_fn: Activation function for the last layer. This can be
            used to restrict the range of the output. For example, one can pass
            tf.keras.activations.sigmoid here to restrict the output to be bounded
            between 0 and 1.
          kernel_initializer: kernel initializer for all layers except for the value
            regression layer. If None, a VarianceScaling initializer will be used.
          last_kernel_initializer: kernel initializer for the value regression
             layer. If None, a RandomUniform initializer will be used.
          batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
          dtype: The dtype to use by the layers.
          name: A string representing name of the network.

        Raises:
          ValueError: If `observation_spec` or `action_spec` contains more than one
            observation.
        """
        observation_spec, action_spec = input_tensor_spec

        super().__init__(
            input_tensor_spec=observation_spec,
            state_spec=(),
            name=name)

        if kernel_initializer is None:
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform')
        if last_kernel_initializer is None:
            last_kernel_initializer = tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003)

        self._observation_encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=observation_preprocessing_layers,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            dropout_layer_params=observation_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype)

        flat_action_spec = tf.nest.flatten(action_spec)
        q_output_size = np.sum([
            single_spec.maximum-single_spec.minimum+1
            for single_spec in flat_action_spec
        ])
        self._output_layer = tf.keras.layers.Dense(
            q_output_size,
            activation=output_activation_fn,
            kernel_initializer=last_kernel_initializer,
            name='value')

    def call(self, inputs, step_type=(), network_state=(), training=False):
        state, network_state = self._observation_encoder(
            inputs, step_type=step_type, network_state=network_state,
            training=training)

        q_values = self._output_layer(state)
        return q_values, network_state