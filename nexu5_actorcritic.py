import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import gymnasium as gym
from gymnasium.spaces import Box 
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from vizdoom import gymnasium_wrapper
import cv2
import main

action_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    



#env = ContinuousViZDoomWrapper(env)


gamma = 0.99
tau = 0.005
actor_lr  = 0.0001
critic_lr = 0.001


state_dim = (7056,)
action_dim = action_space.shape  
max_action = action_space.high













def create_actor(state_dim, action_dim, max_action):
    print(f"State dimension: {state_dim}")
    inputs = layers.Input(shape=state_dim)
    out = layers.Dense(256, activation="relu")(inputs)
    #out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)

    print(f"Action dimension: {action_dim}")
    outputs = layers.Dense(action_dim[0], activation="tanh")(out)

    scaled_outputs = layers.Lambda(lambda x: x * max_action)(outputs)

    model = tf.keras.Model(inputs, scaled_outputs)
    return model









def create_critic(state_dim, action_dim):
    regularizer = tf.keras.regularizers.l2(1e-2)

    state_input = layers.Input(shape=state_dim)
    action_input = layers.Input(shape=action_dim)


    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(512, activation="relu", kernel_regularizer=regularizer)(concat)
    #out = layers.Dense(512, activation="relu", kernel_regularizer=regularizer)(out)
    out = layers.Dense(512, activation="relu", kernel_regularizer=regularizer)(out)
    outputs = layers.Dense(1)(out)  

    model = tf.keras.Model([state_input, action_input], outputs)
    return model









class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        print(max_size)


        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def store(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done


        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices],
        )









#-----------------------------------------------------------------------------



def ddpg_train_step(
    actor_model, critic_model, target_actor, target_critic,
    actor_optimizer, critic_optimizer, replay_buffer,
    batch_size, gamma, tau):






    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)


    state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
    action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
    reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
    next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
    done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)


    with tf.GradientTape() as tape:

        target_actions = target_actor(next_state_batch)


        target_q = reward_batch + gamma * (1 - done_batch) * target_critic([next_state_batch, target_actions])
        target_q = tf.stop_gradient(target_q)


        current_q = critic_model([state_batch, action_batch])


        critic_loss = tf.math.reduce_mean(tf.square(current_q - target_q))


    critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
    #critic_grads = tf.clip_by_norm(critic_grads, clip_norm=1.0)
    critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))


    with tf.GradientTape() as tape:

        actions = actor_model(state_batch)
        actor_loss = -tf.math.reduce_mean(critic_model([state_batch, actions]))


    actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)

    actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))


    for (var, target_var) in zip(actor_model.variables, target_actor.variables):
        target_var.assign(tau * var + (1 - tau) * target_var)

    for (var, target_var) in zip(critic_model.variables, target_critic.variables):
        target_var.assign(tau * var + (1 - tau) * target_var)

    print(f"Critic Loss: {critic_loss.numpy()}, Actor Loss: {actor_loss.numpy()}")
    #print(f"current_q: {current_q}, target_q: {target_q}")

    return critic_loss, actor_loss

#------------------------------------------------------------------------------------



def preprocess_observation(obs):
    if isinstance(state_, dict):
        if "screen" in state_:  
            obs = state_["screen"]  
        else:
            raise ValueError(f"Unexpected dict keys in observation: {state_.keys()}")
    else:
        obs = state_  

    obs = np.array(obs, dtype=np.uint8)  


    if obs.ndim == 3: 
        import cv2
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


    obs = cv2.resize(obs, (84, 84))

    normalized = obs.astype(np.float32) / 255.0  
    flattend = normalized.flatten()
    return flattend

def convert_continuous_action(action):

    move_right = action[0] > 0.5
    move_left = action[0] < -0.5
    move_forward = action[1] > 0.5
    move_backward = action[1] < -0.5
    turn_right = action[2] > 0.5
    turn_left = action[2] < -0.5
    attack = action[3] > 0.5  
    if move_right and not move_left:
        if move_forward and not move_backward:
            return 0  
        elif not move_forward and move_backward:
            return 1  
        elif not move_forward and not move_backward:
            return 2  

    if move_left and not move_right:
        if move_forward and not move_backward:
            return 3  
        elif not move_forward and move_backward:
            return 4  
        elif not move_forward and not move_backward:
            return 5  
    if turn_right and not turn_left:
        return 6  

    if turn_left and not turn_right:
        return 7  

    
    return 0 


actor = create_actor(state_dim, action_dim, max_action)
critic = create_critic(state_dim, action_dim)
target_actor = create_actor(state_dim, action_dim, max_action)
target_critic = create_critic(state_dim, action_dim)


actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm =1.0)


replay_buffer = ReplayBuffer(100000, state_dim[0], action_dim[0])


target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())


batch_size = 100

MAX_STEPS_PER_EPISODE = 300
episode_steps = 0 
episode_rew=[]
avg_rew=[]


critic_losses = []
actor_losses = []

plot_lim = 0
save_frequency = 1000
panel = 0.4
rew = 0
sigma = 0.3


def train_loop(

			   replay_buffer,   episode_steps        ,
			   episode_rew ,   avg_rew,critic_losses,
			   actor_losses,   plot_lim             ,
			   rew         ,   batch_size           ,
			   critic      ,   actor                ,
			   MAX_STEPS_PER_EPISODE, step
			   
			   ):



	episode_rews = 0  
	global state_
	if step ==0:
		state_,info = main.reset()
		print(f"Type of state_: {type(state_)}")
		print(f"Shape of state_: {getattr(state_, 'shape', 'Unknown')}")
		episode_steps = 0  
		episode_rews = 0
		rew = 0.0
		#action = target_actor.predict(np.array([state]))[0]
	state = preprocess_observation(state_)

	action = actor.predict(np.array([state]))[0]
	noise_scale = max(0.1, 0.3 * (1 - step / 200000))  
	noise = np.random.normal(0, noise_scale, size=action_dim)
	actions = np.clip(action + noise, -max_action, max_action)
	final_action = convert_continuous_action(actions)




	next_state, reward, done, truncated, info = main.step(final_action)
	if done or truncated or episode_steps >= MAX_STEPS_PER_EPISODE:
		state_,info = main.reset()
	reward = reward-rew
	print(rew)

	reward = np.clip(reward, -2.0, 2.0)  
	episode_rews+=reward




	next_state = preprocess_observation(next_state)
	replay_buffer.store(state, action, reward, next_state, done or truncated)
	rep = str(replay_buffer)
	print(rep)
		#done = 0

	state = next_state
	plot_lim +=1
	episode_steps += 1
	if replay_buffer.size > batch_size:
		critic_loss, actor_loss = ddpg_train_step(
		        actor, critic, target_actor, target_critic,
		        actor_optimizer, critic_optimizer, replay_buffer,
		        batch_size=batch_size, gamma=gamma, tau=tau)
		critic_losses.append(critic_loss.numpy())
		actor_losses.append(actor_loss.numpy())









	episode_rew.append(episode_rews)
	replay_size = replay_buffer.size
	print(plot_lim)
	print(replay_size)

	







	if plot_lim >= 100:



		plt.figure(figsize=(10, 5)) 
		plt.plot(actor_losses, label='Actor Loss' )
		plt.plot(critic_losses, label='Critic Loss')
		plt.xlabel('Training Steps')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig("actor_critic_losses3.png")
		plt.figure(figsize=(10, 5))
		plt.plot(episode_rew, label='Episode Reward')
		plt.xlabel('Training Steps')
		plt.ylabel('Episode Reward')
		plt.legend()
		plt.savefig("my_plot3.png")
		plot_lim = 0
	print(info)
	rew = 0
	return final_action
		


def play():

    obs_space = train_loop(
			   replay_buffer,   episode_steps        ,
			   episode_rew ,   avg_rew,critic_losses,
			   actor_losses,   plot_lim             ,
			   rew         ,   batch_size           ,
			   critic      ,   actor                ,
			   MAX_STEPS_PER_EPISODE, step           
          )
    #return obs_space
for step in range(100000):
    train_loop(
			   replay_buffer,   episode_steps        ,
			   episode_rew ,   avg_rew,critic_losses,
			   actor_losses,   plot_lim             ,
			   rew         ,   batch_size           ,
			   critic      ,   actor                ,
			   MAX_STEPS_PER_EPISODE, step           
          )
