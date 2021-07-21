import numpy as np
import sys
import tensorflow as tf
import tf_agents
from collections import namedtuple

def penalized_loss(ret):
	def loss(y_true,y_pred):
		return tf.reduce_mean(-ret* tf.losses.sparse_categorical_crossentropy(y_pred=y_pred,y_true=y_true,from_logits=True))
	return loss

class ReplayMemory:
	def __init__(self, config):
		self.config = config
		self.actions = np.empty((self.config.mem_size), dtype=np.int32)
		self.rewards = np.empty((self.config.mem_size), dtype=np.int32)
		self.observations = np.empty((self.config.mem_size, self.config.obs_dims), dtype=np.int32)
		self.count = 0
		self.current = 0

	def add(self, observation, reward, action):
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.count = max(self.count, self.current + 1)
		for i in range(self.count - 1):
			self.observations[i] = self.observations[i + 1]
		self.observations[-1] = observation
		self.current = (self.current + 1) % self.config.mem_size

	def getState(self, index):
		return self.observations.reshape(-1)

class PiApproximationWithNN():
	def __init__(self,state_dims,num_actions,alpha,mem_size):
		self.state_dims = state_dims
		self.obs_dims = state_dims
		self.buffer_size = mem_size
		self.num_actions = num_actions
		input = tf.keras.Input(shape=(self.obs_dims * self.buffer_size,), name='X')
		x = tf.keras.layers.Dense(32, activation='relu', name='dense_1')(input)
		x = tf.keras.layers.Dense(32, activation='relu', name='dense_2')(x)
		output_logit = tf.keras.layers.Dense(num_actions, activation='linear', name='predictions_logits')(x)

		self.model = tf.keras.Model(inputs=input, outputs=output_logit)
		self.ret = tf.Variable(initial_value=[1.0], dtype='float32')
		self.model.compile(optimizer='adam',loss=penalized_loss(self.ret),experimental_run_tf_function=False)


	def __call__(self, s,allowed_actions):
		act_logits = self.model(s.reshape(1,-1))  #self.sess.run(self.actSel, feed_dict={self.X: s.reshape(-1, self.obs_dims * self.buffer_size)})
		mask = np.zeros_like(act_logits.numpy(),dtype=bool)
		for i in allowed_actions:
			mask[0,i] = True
		new_logits = tf_agents.distributions.masked.MaskedCategorical(logits=act_logits,mask=mask)
		action = tf.random.categorical(logits=new_logits.logits, num_samples=1, dtype=None, seed=None, name=None).numpy()[0,0]
		return action

	def update(self, s, a, gamma_t, delta):
		ret = np.array(gamma_t * delta).reshape(-1,)
		self.ret.assign(ret)
		self.model.fit({'X':s.reshape(1,-1)},np.array([[a]], dtype=int),verbose=0)

	def add_config(self, config):
		self.config = config


class Baseline(object):
	"""
	The dumbest baseline; a constant for every state
	"""
	def __init__(self, b):
		self.b = b

	def __call__(self, s):
		return self.b

	def update(self, s, G):
		pass

def REINFORCE(simulator,recorder,logger,gamma, nr_good_runs = 1, total_nr_runs = 5, maxsteps=30):
	alpha = 3e-4
	mem_size = 1

	good_runs = 0
	nr_actions = max([simulator._model.get_nr_available_actions(i) for i in range(simulator._model.nr_states)])
	pi = PiApproximationWithNN(1,nr_actions,alpha,mem_size=mem_size)
	V = Baseline(0.)

	G_0 = []
	result = []
	struc = namedtuple("struc", ['mem_size', 'obs_dims'])
	config = struc(mem_size,1)
	rep = ReplayMemory(config)
	pi.add_config(config)
	good_runs = 0
	# obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
	for e_i in range(total_nr_runs):
		finished = False
		state = simulator._simulator.restart()
		simulator._shield.reset()
		logger.info("Start new episode.")
		recorder.start_path()
		recorder.record_state(state)
		recorder.record_belief(simulator._shield.list_support())
		traj = []
		for n in range(maxsteps):
			actions = simulator._simulator.available_actions()
			safe_actions = simulator._shield.shielded_actions(range(len(actions)))
			if len(safe_actions) == 0:
				a = 0
				r_t = -100
				rep.add(simulator._simulator._report_observation(), r_t, a)
				rep_zprime = rep.getState(rep.current)
				rep_z = rep.getState(rep.current)
				traj.append((rep_z, a, r_t, rep_zprime))
				finished = True
				break
			else:
				a = pi(rep.getState(rep.current),safe_actions)
				z_prime, r_t = simulator._simulator.step(int(a))
			rep_z = rep.getState(rep.current)
			if len(r_t)==0:
				r_t = 0
			else:
				r_t = r_t[0]-r_t[1] # -r_t[2]
			rep.add(z_prime, r_t, a)
			rep_zprime = rep.getState(rep.current)
			traj.append((rep_z, a, r_t, rep_zprime))
			state = z_prime
			simulator._shield.track(int(a), simulator._model.get_observation(state))
			recorder.record_available_actions(actions)
			recorder.record_allowed_actions(safe_actions)
			recorder.record_selected_action(int(a))
			recorder.record_state(simulator._simulator._report_state())
			recorder.record_belief(simulator._shield.list_support())
			if simulator._simulator.is_done():
				logger.info(f"Done after {n} steps!")
				finished = True
				good_runs += 1
				break
		recorder.record_available_actions(actions)
		recorder.record_allowed_actions(safe_actions)
		recorder.end_path(finished)
		result.append(simulator._simulator.is_done())
		for i, t_tup in enumerate(traj):
			G = sum([gamma ** j * s_tup[2] for j, s_tup in enumerate(traj[i:])])
			if i == 0: G_0.append(G)
			delta = G - V(t_tup[0])
			V.update(t_tup[0], G)
			pi.update(t_tup[0], t_tup[1], gamma ** i, delta)
		if good_runs == nr_good_runs:
			break

	return G_0, pi,result


# num_iter = 10
# with_buffer2 = []
# print('***************************************')
# for q in range(num_iter):
# 	print("----------------> With Buffer = 2: {}".format(q))
# 	training_progress = test_reinforce_Buffer(env, 2, q)
# 	with_buffer2.append(training_progress[0])
# 	pi_buff = training_progress[1]
# print('***************************************')
# with_buffer2 = np.mean(with_buffer2, axis=0)