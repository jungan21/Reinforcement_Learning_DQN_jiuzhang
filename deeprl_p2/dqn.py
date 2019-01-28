#coding:utf-8

import numpy as np
import sys
import keras.models
import keras.backend as K
from deeprl_p2.utils import *
from PIL import Image

"""Main DQN agent."""

class DQNAgent:
	"""Class implementing DQN.

	This is a basic outline of the functions/parameters you will need
	in order to implement the DQNAgnet. This is just to get you
	started. You may need to tweak the parameters, add new ones, etc.

	Feel free to change the functions and funciton parameters that the
	class provides.

	We have provided docstrings to go along with our suggested API.

	Parameters
	----------
	q_network: keras.models.Model
	  Your Q-network model.
	preprocessor: deeprl_p2.core.Preprocessor
	  The preprocessor class. See the associated classes for more
	  details.
	memory: deeprl_p2.core.Memory
	  Your replay memory.
	gamma: float
	  Discount factor.
	target_update_freq: float
	  Frequency to update the target network. You can either provide a
	  number representing a soft target update (see utils.py) or a
	  hard target update (see utils.py and Atari paper.)
	num_burn_in: int
	  Before you begin updating the Q-network your replay memory has
	  to be filled up with some number of samples. This number says
	  how many.
	train_freq: int
	  How often you actually update your Q-Network. Sometimes
	  stability is improved if you collect a couple samples for your
	  replay memory, for every Q-network update that you run.
	batch_size: int
	  How many samples in each minibatch.
	"""
	def __init__(self,
				 q_network,
				 q_values_func,
				 preprocessor,
				 memory,
				 policy,
				 gamma,
				 target_update_freq,
				 num_burn_in,
				 train_freq,
				 batch_size,
				 save_path
				 ):

		self.q_network = q_network
		# q_network: used for loss function right side i.e. Q(s, a, w) 预测的值。 让这个网络里的w先学习，
		# target_network： 和q_network同一个网络，只不过这里的w先固定住一段时间，等到上面 q_network 学习好了，再把学习好了的w带入这个target_network
		self.target_network = keras.models.clone_model(q_network)
		self.target_network.set_weights(q_network.get_weights()) # 等到 q_network 学习好了，再把学习好了的w带入这个target_network
		self.target_q_values_func = K.function([self.target_network.layers[0].input], [self.target_network.layers[5].output])

		self.q_values_func = q_values_func
		self.preprocessor = preprocessor
		self.memory = memory
		self.policy = policy
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.num_burn_in = num_burn_in
		self.train_freq = train_freq
		self.batch_size = batch_size
		self.save_path = save_path
		self.num_steps = 0

		# can be train, test or init
		self.mode = 'init'

	def compile(self, optimizer, loss_func):
		"""Setup all of the TF graph variables/ops.

		This is inspired by the compile method on the
		keras.models.Model class.

		This is a good place to create the target network, setup your
		loss function and any placeholders you might need.

		You should use the mean_huber_loss function as your
		loss_function. You can also experiment with MSE and other
		losses.

		The optimizer can be whatever class you want. We used the
		keras.optimizers.Optimizer class. Specifically the Adam
		optimizer.
		"""
		self.q_network.compile(optimizer=optimizer, loss=loss_func)
		self.target_network.compile(optimizer=optimizer, loss=loss_func)

	def load_weights(self, weights_path):
		self.q_network.load_weights(weights_path)
		self.update_target_network()

	def update_target_network(self):
		self.target_network.set_weights(self.q_network.get_weights())


	def calc_q_values(self, state):
		"""Given a state (or batch of states) calculate the Q-values.

		Basically run your network on these states.

		Return
		------
		Q-values for the state(s)
		"""
		return self.q_values_func([state])[0]

	def cal_target_q_values(self, state):
		return self.target_q_values_func([state])[0]

	def select_action(self, state, **kwargs):
		"""Select the action based on the current state.

		You will probably want to vary your behavior here based on
		which stage of training your in. For example, if you're still
		collecting random samples you might want to use a
		UniformRandomPolicy.

		If you're testing, you might want to use a GreedyEpsilonPolicy
		with a low epsilon.

		If you're training, you might want to use the
		LinearDecayGreedyEpsilonPolicy.

		This would also be a good place to call
		process_state_for_network in your preprocessor.

		Returns
		--------
		selected action
		"""
		preprocessed_state = self.preprocessor.process_state_for_network(state)
		q_values = self.calc_q_values(preprocessed_state)
		#print (np.argmax(q_values), q_values)
		return self.policy[self.mode].select_action(q_values), preprocessed_state


	def update_predict_network(self):
		"""Update your predict network.

		Behavior may differ based on what stage of training your
		in. If you're in training mode then you should check if you
		should update your network parameters based on the current
		step and the value you set for train_freq.

		Inside, you'll want to sample a minibatch, calculate the
		target values, update your network, and then update your
		target values.

		You might want to return the loss and other metrics as an
		output. They can help you monitor how training is going.
		"""

		# retrive from memory
		# 从经验数组D中随机抽取一批转移样本 ( 𝜙t , 𝑎t , 𝑟t , 𝜙(t+1))
		states, actions, rewards, new_states, is_terminals = self.memory.sample(self.batch_size)

		preprocessed_states, preprocessed_new_states = self.preprocessor.process_batch(states, new_states)

		actions = self.preprocessor.process_action(actions)


		# update network
		# 下面几行 对应算法里的： 计算Q值 step
		q_values = self.cal_target_q_values(preprocessed_new_states)
		max_q_values = np.max(q_values, axis=1)
		max_q_values[is_terminals] = 0.0
		targets = rewards + self.gamma * max_q_values
		targets = np.expand_dims(targets, axis=1)

		# 先固定住target_network, 这里train q_network(i.e.就是loss function的右半边的Q(𝜙j,𝑎j; 𝜔)), 得到loss, 并梯度下降得到新的w
		self.q_network.train_on_batch([preprocessed_states, actions], targets) # q_network 也就是keras 的Model

		if self.num_steps % self.target_update_freq == 0: # 控制 update target_network w值的frequency, 已达到延迟跟新target_network的效果
			print("Update target network at %d steps" % self.num_steps)
			self.update_target_network() # 其实就一句话： self.target_network.set_weights(self.q_network.get_weights())

	def fit(self, env, num_iterations, max_episode_length=None):
		"""Fit your model to the provided environment.

		Its a good idea to print out things like loss, average reward,
		Q-values, etc to see if your agent is actually improving.

		You should probably also periodically save your network
		weights and any other useful info.

		This is where you should sample actions from your network,
		collect experience samples and add them to your replay memory,
		and update your network parameters.

		Parameters
		----------
		env: gym.Env
		  This is your Atari environment. You should wrap the
		  environment using the wrap_atari_env function in the
		  utils.py
		num_iterations: int
		  How many samples/updates to perform.
		max_episode_length: int
		  How long a single episode should last before the agent
		  resets. Can help exploration.
		"""
		print ('initializing replay memory...')
		sys.stdout.flush()
		self.mode = 'init'
		self.memory.clear()
		self.preprocessor.reset()
		self.num_steps = 0
		num_updates = 0
		num_episodes = 0
		while num_updates < num_iterations:
			state = env.reset()
			self.preprocessor.reset()
			num_episodes += 1
			t = 0
			total_reward = 0
			#print ('episode start')
			while True:
				#env.render()
				self.num_steps += 1
				t += 1
				action, _ = self.select_action(state) # 选择Q值最大的动作:𝑎(t) =max𝑄∗(𝜙(S(t)),𝑎;𝜔),并有一定概率随机选一个动作
				next_state, reward, is_terminal, debug_info = env.step(action) # 在环境里执行动作得到奖励r以及新的画面X(t+1)

				reward = self.preprocessor.process_reward(reward)
				total_reward += reward

				preprocessed_state = self.preprocessor.process_state_for_memory(state) # 把新的画面X(t+1)加到St之后并进行预处理得到𝜙(t+1) = 𝜙(S(t+1)) .

				self.memory.append(preprocessed_state, action, reward, is_terminal) # 把状态转移信息( 𝜙t , 𝑎t , 𝑟t , 𝜙(t+1)) 放入经验数组D中

				if self.num_steps > self.num_burn_in:
					if self.mode != 'train':
						print("Finish Burn-in, Start Training!")

					self.mode = 'train'
					if self.num_steps % self.train_freq == 0:
						self.update_predict_network()
						num_updates += 1
						if num_updates % 10000 == 0:
							self.q_network.save_weights('%s/model_weights_%d.h5' % (self.save_path, num_updates // 10000))

				if is_terminal or (max_episode_length is not None and t > max_episode_length):
					break

				state = next_state
			print ('episode %d ends, lasts for %d steps (total steps:%d), gets %d reward. (%d/%d updates).' % (num_episodes, t, self.num_steps, total_reward, num_updates, num_iterations))

	def evaluate(self, env, num_episodes, max_episode_length=None):
		"""Test your agent with a provided environment.

		You shouldn't update your network parameters here. Also if you
		have any layers that vary in behavior between train/test time
		(such as dropout or batch norm), you should set them to test.

		Basically run your policy on the environment and collect stats
		like cumulative reward, average episode length, etc.

		You can also call the render function here if you want to
		visually inspect your policy.
		"""
		self.mode = 'test'

		average_episode_length = 0.
		rewards = []

		for i in range(num_episodes):
			state = env.reset()
			t = 0
			episode_reward = 0.0
			while True:
				#env.render()
				t += 1
				action, _ = self.select_action(state)
				next_state, reward, is_terminal, debug_info = env.step(action)
				#preprocessed_reward = self.preprocessor.process_reward(reward)
				episode_reward += reward
				average_episode_length += 1

				if is_terminal or (max_episode_length is not None and t > max_episode_length):
					break
				state = next_state
			rewards.append(episode_reward)
		self.mode = 'train'

		return np.mean(rewards), np.std(rewards), average_episode_length / num_episodes
