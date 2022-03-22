import gym
from gym import error, spaces


class ShieldToGymWrapper(gym.Env):
    def __init__(self,
                 model,
                 shield
    ):
        super().__init__(model, shield)
        self.nr_actions = max(action_spec_count)self.step_count = 0
        self.episode_count = 0
        self.cost_ind = list(self._model.reward_models.keys()).index('costs')
        self.gain_ind = list(self._model.reward_models.keys()).index('gains')
        self.first = True
        self.maxsteps = maxsteps

    def restart(self):
        self._simulator.restart()
        self._shield.reset()
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
            r = tf.constant([rew])
        else:
            r = tf.constant([self._simulator._report_rewards()[0]]) if len(self._simulator._report_rewards()) != 0 else tf.constant([0.])
        discount = tf.constant([1.])
        actions = self._simulator.available_actions()
        safe_actions = self._shield.shielded_actions(range(len(actions)))
        mask = np.zeros(shape=(self.nr_actions,),dtype=bool)
        for i in safe_actions:
            mask[i] = True
        mask = tf.logical_and(tf.ones(shape=(1,self.nr_actions),dtype=tf.bool),mask)
        observation = {'obs':tf.constant([self.observe()],dtype='int32'),'mask':mask}
        if self.first:
            self.first = False
            return ts.TimeStep(reward=r, observation=observation, discount=discount,step_type=tf.constant([ts.StepType.FIRST]))
        elif self.is_done():
            self.restart()
            self.first=True
            return ts.TimeStep(reward=r,observation=observation,discount=discount,step_type=tf.constant([ts.StepType.LAST]))
        else:
            return ts.TimeStep(reward=r, observation=observation,discount=discount, step_type=tf.constant([ts.StepType.MID]))

    def step(self,action):
        state, rew = self._simulator.step(action)
        self._shield.track(action, self._simulator._report_observation())
        # obs = self.observe()
        self.step_count += 1
        if (self.is_done() and 'traps' in self._model.states[state].labels):
            rew[self.cost_ind] += 1000
        elif (self.is_done() and 'goal' in self._model.states[state].labels):
            rew[self.gain_ind] += 1000
        elif (self.is_done()):
            rew[self.cost_ind] += 100
        current_step = self.current_time_step(rew=self.cost_fn(rew))
        # self.replay_memory.add(action,self.cost_fn(rew),obs)
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
        return keywords

def json_to_int(i):
    try:
        return int(i)
    except:
        return 0 if i=='false' else 1