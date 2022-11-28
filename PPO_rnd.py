import gym
import gymnasium
import argparse
from minigrid.wrappers import RGBImgPartialObsWrapper, RGBImgObsWrapper
from MiniGrid_Wrappers import StateBonus, GrayImgObsWrapper, FrameStackWrapper, MaxStepWrapper, NormalizeObsWrapper

import tensorflow as tf

print(tf.executing_eagerly())
import tensorflow_probability as tfp
from tensorflow.keras.layers import MaxPooling3D, Conv3D, GlobalAveragePooling2D, concatenate, add, Multiply, Permute, Softmax, AveragePooling2D, MaxPooling2D, Convolution2D, LeakyReLU, Reshape, Lambda, Conv2D, LSTMCell, LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, multiply, Concatenate, Flatten, Activation, dot, Dot, Dropout
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import sys
import numpy
from model_ppo_tf2 import RNDmodel
from model_ppo_tf2 import DQN, DRQN, Conv_Transformer, ConvTransformer, ViTrans, MFCA, MultiscaleTransformer, OnlyMultiscale
from Atari_Warppers import NoopResetEnv, NormalizedEnv, ResizeObservation, SyncVectorEnv, ClipRewardEnv, EpisodicLifeEnv, FireResetEnv
import wandb
from wandb.keras import WandbCallback

# 设置显存按需获取
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 设置每个GPU可访问显存的量的上限
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
    try:
        for i in range(len(gpu_list)):
            tf.config.experimental.set_virtual_device_configuration(
                gpu_list[i], [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=8192)
                ])
    except RuntimeError as e:
        print(e)
else:
    print("NO GPUs")

# strategy = tf.distribute.MirroredStrategy()

# 输入参数
parser = argparse.ArgumentParser(description='Training parameters')

parser.add_argument('--game',
                    type=str,
                    default='MiniGrid-MemoryS11-v0',
                    help="Games in Atari")
parser.add_argument('--model', type=str, default='DQN', help="Model we use")
parser.add_argument('--action_dim',
                    type=int,
                    default=3,
                    help="Action dimension")
parser.add_argument('--rnd', action='store_true', help='If use RND model')
parser.set_defaults(render=False)

args = parser.parse_args()


#
class Utils():
    # 计算新的平均值
    def count_new_mean(self, prevMean, prevLen, newData):
        return ((prevMean * prevLen) +
                tf.math.reduce_sum(newData, 0)) / (prevLen + newData.shape[0])

    # 计算新的标准差
    def count_new_std(self, prevStd, prevLen, newData):
        return tf.math.sqrt(
            ((tf.math.square(prevStd) * prevLen) +
             (tf.math.reduce_variance(newData, 0) * newData.shape[0])) /
            (prevStd + newData.shape[0]))

    # 标准化data中的数据，并完成剪切
    def normalize(self, data, mean=None, std=None, clip=None):
        if isinstance(mean, tf.Tensor) and isinstance(std, tf.Tensor):
            data_normalized = (data - mean) / (std + 1e-8)
        else:
            data_normalized = (data - tf.math.reduce_mean(data)) / (
                tf.math.reduce_std(data) + 1e-8)

        if clip:
            data_normalized = tf.clip_by_value(data_normalized, -clip, clip)

        return data_normalized


# 骨架模型
backbone = eval(args.model)()
# 参数初始化
initializer = tf.keras.initializers.Orthogonal(gain=0.1)


class Actor_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model, self).__init__()
        self.backbone = backbone
        self.actor_dense = Dense(
            256,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            activation='relu')
        # self.outputs = Dense(action_dim,activation='linear',name='output',kernel_initializer=initializer)
        self.outputs = Dense(action_dim,
                             activation='softmax',
                             name='output',
                             kernel_initializer=initializer)

    def call(self, x):
        x = self.backbone(x)
        x = self.actor_dense(x)
        x = self.outputs(x)
        return x


class Critic_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model, self).__init__()
        self.backbone = backbone
        self.critic_dense = Dense(
            256,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            activation='relu')
        self.outputs = Dense(1,
                             kernel_initializer=initializer,
                             activation='relu')

    def call(self, x):
        x = self.backbone(x)
        x = self.critic_dense(x)
        x = self.outputs(x)
        return x


class RND_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(RND_Model, self).__init__()
        # 提取最后一帧数据
        self.last_frame = Lambda(lambda x: x[:, -1, :, :, :])
        self.conv1 = Conv2D(32,
                            8, (4, 4),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)
        self.conv2 = Conv2D(64,
                            4, (2, 2),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)
        self.conv3 = Conv2D(64,
                            3, (1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)
        self.flat = Flatten()
        self.outputs = Dense(5, activation='linear')

    def call(self, x):
        x = self.last_frame(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.outputs(x)
        return x


# 专门存observation的结构
class ObsMemory():
    def __init__(self, state_dim):
        self.observations = []

        self.mean_obs = tf.zeros(state_dim, dtype=tf.float32)
        self.std_obs = tf.zeros(state_dim, dtype=tf.float32)
        self.std_in_rewards = tf.zeros(1, dtype=tf.float32)
        self.total_number_obs = tf.zeros(1, dtype=tf.float32)
        self.total_number_rwd = tf.zeros(1, dtype=tf.float32)

    def __len__(self):
        return len(self.observations)

    def get_all(self):
        return tf.constant(self.observations, dtype=tf.float32)

    def get_all_tensor(self):
        observations = tf.constant(self.observations, dtype=tf.float32)
        # 返回dataset对象
        return tf.data.Dataset.from_tensor_slices(observations)

    # 将obs存入数组
    def save_eps(self, obs):
        self.observations.append(obs)

    # 保存obs和reward的std和avg
    def save_observation_normalize_parameter(self, mean_obs, std_obs,
                                             total_number_obs):
        self.mean_obs = mean_obs
        self.std_obs = std_obs
        self.total_number_obs = total_number_obs

    def save_rewards_normalize_parameter(self, std_in_rewards,
                                         total_number_rwd):
        self.std_in_rewards = std_in_rewards
        self.total_number_rwd = total_number_rwd

    # 删掉所有的obs
    def clear_memory(self):
        del self.observations[:]


# 作为记忆存储的结构
class Memory():
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.dones)

    # 将存储的内容转换为tf.constant然后返回
    def get_all_tensor(self):
        states = tf.constant(self.states, dtype=tf.float32)
        actions = tf.constant(self.actions, dtype=tf.float32)
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype=tf.float32),
                                 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype=tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype=tf.float32)

        return tf.data.Dataset.from_tensor_slices(
            (states, actions, rewards, dones, next_states))

    # 存储
    def save_eps(self, state, action, reward, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)

    # 清理
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]


class Distributions():
    def sample(self, datas):
        # probs给概率P，使用sample方法根据概率进行抽样
        distribution = tfp.distributions.Categorical(probs=datas)
        # distribution = tfp.distributions.Categorical(logits=datas)
        return distribution.sample()

    def entropy(self, datas):
        # 返回probs这一堆概率值的entropy
        distribution = tfp.distributions.Categorical(probs=datas)
        # distribution = tfp.distributions.Categorical(logits=datas)
        return distribution.entropy()

    def logprob(self, datas, value_data):
        # 返回概率的log值
        distribution = tfp.distributions.Categorical(probs=datas)
        # distribution = tfp.distributions.Categorical(logits=datas)

        return tf.expand_dims(distribution.log_prob(value_data), 1)

    def kl_divergence(self, datas1, datas2):
        # 返回两个分布的KL距离
        distribution1 = tfp.distributions.Categorical(probs=datas1)
        # distribution1 = tfp.distributions.Categorical(logits=datas1)
        distribution2 = tfp.distributions.Categorical(probs=datas2)
        # distribution2 = tfp.distributions.Categorical(logits=datas2)

        return tf.expand_dims(
            tfp.distributions.kl_divergence(distribution1, distribution2), 1)


# 根据reward，value，next value以及done的值，采用三种方法计算value
class PolicyFunction():
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns = []
        # 从后往前算cumulative reward
        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (
                1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)

        return tf.stack(returns)

    def temporal_difference(self, reward, next_value, done):
        # one-step TD
        q_values = reward + (1 - done) * self.gamma * next_value
        return q_values

    def generalized_advantage_estimation(self, values, rewards, next_values,
                                         dones):

        gae = 0
        adv = []
        # one-step TD - predict value
        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        for step in reversed(range(len(rewards))):
            # 对delta进行一个类似cumulative reward的算
            gae = delta[step] + (1.0 -
                                 dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)

        return tf.stack(adv)


class Agent():
    def __init__(self, state_dim, action_dim, is_training_mode, value_clip,
                 entropy_coef, vf_loss_coef, minibatch, PPO_epochs, gamma, lam,
                 learning_rate, n_episode):
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.minibatch = minibatch
        self.PPO_epochs = PPO_epochs
        self.RND_epochs = 5
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim
        self.actor = Actor_Model(state_dim, action_dim)
        self.actor_old = Actor_Model(state_dim, action_dim)

        self.ex_critic = Critic_Model(state_dim, action_dim)
        self.ex_critic_old = Critic_Model(state_dim, action_dim)

        if args.rnd:
            self.in_critic = Critic_Model(state_dim, action_dim)
            self.in_critic_old = Critic_Model(state_dim, action_dim)

            self.rnd_predict = RND_Model(state_dim, action_dim)
            self.rnd_target = RND_Model(state_dim, action_dim)
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            learning_rate, 10000, 0.1 * learning_rate, power=1)
        self.ppo_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_fn, epsilon=1e-05, clipnorm=0.5)
        if args.rnd:
            self.rnd_optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, epsilon=1e-05, clipnorm=0.5)

        self.memory = Memory()
        self.obs_memory = ObsMemory(state_dim)
        self.utils = Utils()

        self.policy_function = PolicyFunction(gamma, lam)
        self.distributions = Distributions()
        self.update_times = 0
        if args.rnd:
            self.ex_advantages_coef = 2
            self.in_advantages_coef = 1
        self.clip_normalization = 5

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def save_observation(self, obs):
        self.obs_memory.save_eps(obs)

    # 更新obs_memory的标准化系数
    def update_obs_normalization_param(self, obs):
        obs = tf.constant(obs, dtype=tf.float32)

        mean_obs = self.utils.count_new_mean(self.obs_memory.mean_obs,
                                             self.obs_memory.total_number_obs,
                                             obs)
        std_obs = self.utils.count_new_std(self.obs_memory.std_obs,
                                           self.obs_memory.total_number_obs,
                                           obs)
        total_number_obs = len(obs) + self.obs_memory.total_number_obs

        self.obs_memory.save_observation_normalize_parameter(
            mean_obs, std_obs, total_number_obs)

    def update_rwd_normalization_param(self, in_rewards):
        std_in_rewards = self.utils.count_new_std(
            self.obs_memory.std_in_rewards, self.obs_memory.total_number_rwd,
            in_rewards)
        total_number_rwd = len(in_rewards) + self.obs_memory.total_number_rwd

        self.obs_memory.save_rewards_normalize_parameter(
            std_in_rewards, total_number_rwd)

    # RND Loss
    if args.rnd:

        def get_rnd_loss(self, state_pred, state_target):
            # Don't update target state value
            state_target = tf.stop_gradient(state_target)

            # Mean Squared Error Calculation between state and predict
            forward_loss = tf.math.reduce_mean(
                tf.math.square(state_target - state_pred) * 0.5)
            return forward_loss

    # Loss for PPO
    def get_PPO_loss(self, action_probs, ex_values, old_action_probs,
                     old_ex_values, next_ex_values, actions, ex_rewards, dones,
                     state_preds, state_targets, in_values, old_in_values,
                     next_in_values, std_in_rewards):

        # stop_gradient将原来的tensor当作一个独立的节点来看待，切断反向传播的路径
        Old_ex_values = tf.stop_gradient(old_ex_values)

        # 将外部奖励扔进去计算GAE
        External_Advantages = self.policy_function.generalized_advantage_estimation(
            ex_values, ex_rewards, next_ex_values, dones)
        # 加上原来的value作为return
        External_Returns = tf.stop_gradient(External_Advantages + ex_values)
        # 进行一个标准化
        External_Advantages = tf.stop_gradient(
            (External_Advantages - tf.math.reduce_mean(External_Advantages)) /
            (tf.math.reduce_std(External_Advantages) + 1e-6))

        # 计算内在奖励，除以方差的标准化
        in_rewards = tf.math.square(state_targets - state_preds) * 0.5 / (
            tf.math.reduce_mean(std_in_rewards) + 1e-8)
        # 计算GAE（这个value是由专门计算in_value的critic网络算的）
        Internal_Advantages = self.policy_function.generalized_advantage_estimation(
            in_values, in_rewards, next_in_values, dones)
        # 和外部奖励一样的步骤
        Internal_Returns = tf.stop_gradient(Internal_Advantages + in_values)
        Internal_Advantages = tf.stop_gradient(
            (Internal_Advantages - tf.math.reduce_mean(Internal_Advantages)) /
            (tf.math.reduce_std(Internal_Advantages) + 1e-6))

        # 对内外部奖励进行一个权重的加法
        Advantages = tf.stop_gradient(
            self.ex_advantages_coef * External_Advantages +
            self.in_advantages_coef * Internal_Advantages)

        # 计算log之后相减再exp作为比值（引入误差？）
        # actions = actions.reshape(-1,1)
        # index = np.arange(len(actions)).reshape(-1, 1)
        # index = np.hstack((index, actions))
        # probs = tf.gather_nd(action_probs, index)
        # old_probs = tf.gather_nd(old_action_probs, index)
        # ratios = probs / old_probs
        logprobs = self.distributions.logprob(action_probs, actions)
        Old_logprobs = tf.stop_gradient(
            self.distributions.logprob(old_action_probs, actions))
        ratios = tf.math.exp(logprobs - Old_logprobs)

        # cal PPO-clip, 0.2 is hyper-parameter
        pg_loss = tf.minimum(
            ratios * Advantages,
            tf.clip_by_value(ratios, 1 - 0.2, 1 + 0.2) * Advantages)
        # add duel-clip
        duel_clip_loss = tf.where(tf.math.less(Advantages, 0),
                                  tf.maximum(pg_loss, 3 * Advantages), pg_loss)
        pg_loss = tf.math.reduce_mean(duel_clip_loss)
        # 从prob中获得entropy
        dist_entropy = tf.math.reduce_mean(
            self.distributions.entropy(action_probs))

        # 对预测的外部value进行一个裁剪
        ex_vpredclipped = Old_ex_values + tf.clip_by_value(
            ex_values - Old_ex_values, -self.value_clip, self.value_clip)
        # 分别计算clip后于clip前预测的外部value与外部return的差
        ex_vf_losses1 = tf.math.square(External_Returns -
                                       ex_values)  # Mean Squared Error
        ex_vf_losses2 = tf.math.square(External_Returns -
                                       ex_vpredclipped)  # Mean Squared Error
        # 把这个差当作loss
        critic_ext_loss = tf.math.reduce_mean(
            tf.math.maximum(ex_vf_losses1, ex_vf_losses2))
        # 把内在奖励与in_values的差当作loss
        critic_int_loss = tf.math.reduce_mean(
            tf.math.square(Internal_Returns - in_values))
        # critic loss等于两个差的和，权重相等
        critic_loss = (critic_ext_loss + critic_int_loss) * 0.5

        # 对于PG loss，它实际上是目标函数 我们要最大化它 也就是最小化它的负数
        # 对于critic loss 我们要最小化它
        # 对于熵 熵越大概率分布越均匀 所以我们要最大化熵 也就是最小化它的负数
        loss = (critic_loss * self.vf_loss_coef) - (
            dist_entropy * self.entropy_coef) - pg_loss
        return loss, critic_loss, dist_entropy, pg_loss

    def get_loss(self, action_probs, ex_values, old_action_probs,
                 old_ex_values, next_ex_values, actions, ex_rewards, dones):
        # stop_gradient将原来的tensor当作一个独立的节点来看待，切断反向传播的路径
        Old_ex_values = tf.stop_gradient(old_ex_values)

        # 将外部奖励扔进去计算GAE
        External_Advantages = self.policy_function.generalized_advantage_estimation(
            ex_values, ex_rewards, next_ex_values, dones)
        # 加上原来的value作为return
        External_Returns = tf.stop_gradient(External_Advantages + ex_values)
        # 进行一个标准化
        External_Advantages = tf.stop_gradient(
            (External_Advantages - tf.math.reduce_mean(External_Advantages)) /
            (tf.math.reduce_std(External_Advantages) + 1e-6))

        logprobs = self.distributions.logprob(action_probs, actions)
        Old_logprobs = tf.stop_gradient(
            self.distributions.logprob(old_action_probs, actions))
        ratios = tf.math.exp(logprobs - Old_logprobs)

        # cal PPO-clip, 0.2 is hyper-parameter
        pg_loss = tf.minimum(
            ratios * External_Advantages,
            tf.clip_by_value(ratios, 1 - 0.2, 1 + 0.2) * External_Advantages)
        # add duel-clip
        duel_clip_loss = tf.where(tf.math.less(External_Advantages, 0),
                                  tf.maximum(pg_loss, 3 * External_Advantages),
                                  pg_loss)
        pg_loss = tf.math.reduce_mean(duel_clip_loss)
        # 从prob中获得entropy
        dist_entropy = tf.math.reduce_mean(
            self.distributions.entropy(action_probs))

        # 对预测的外部value进行一个裁剪
        ex_vpredclipped = Old_ex_values + tf.clip_by_value(
            ex_values - Old_ex_values, -self.value_clip, self.value_clip)
        # 分别计算clip后于clip前预测的外部value与外部return的差
        ex_vf_losses1 = tf.math.square(External_Returns -
                                       ex_values)  # Mean Squared Error
        ex_vf_losses2 = tf.math.square(External_Returns -
                                       ex_vpredclipped)  # Mean Squared Error
        # 把这个差当作loss
        critic_ext_loss = tf.math.reduce_mean(
            tf.math.maximum(ex_vf_losses1, ex_vf_losses2))

        # 对于PG loss，它实际上是目标函数 我们要最大化它 也就是最小化它的负数
        # 对于critic loss 我们要最小化它
        # 对于熵 熵越大概率分布越均匀 所以我们要最大化熵 也就是最小化它的负数
        loss = (critic_ext_loss * self.vf_loss_coef) - (
            dist_entropy * self.entropy_coef) - pg_loss
        return loss, critic_ext_loss, dist_entropy, pg_loss

    # @tf.function
    def act(self, state):
        state = tf.expand_dims(tf.cast(state, dtype=tf.float32), 0)
        action_probs = self.actor(state)
        critic = self.ex_critic(state)
        wandb.log({'max_p': tf.math.reduce_max(action_probs).numpy()},
                  commit=False)
        wandb.log({'min_p': tf.math.reduce_min(action_probs).numpy()},
                  commit=False)
        wandb.log({'value': critic.numpy()})
        if self.is_training_mode:
            action = self.distributions.sample(action_probs)
        else:
            action = tf.math.argmax(action_probs, 1)

        return action

    # @tf.function
    def compute_intrinsic_reward(self, obs, mean_obs, std_obs):
        obs = self.utils.normalize(obs, mean_obs, std_obs)

        state_pred = self.rnd_predict(obs)
        state_target = self.rnd_target(obs)

        return (state_target - state_pred)

    # 进行一个反向传播
    # @tf.function
    def training_rnd(self, obs, mean_obs, std_obs):
        obs = self.utils.normalize(obs, mean_obs, std_obs)
        with tf.GradientTape() as tape:
            state_pred = self.rnd_predict(obs)
            state_target = self.rnd_target(obs)

            loss = self.get_rnd_loss(state_pred, state_target)

        gradients = tape.gradient(loss, self.rnd_predict.trainable_variables)
        self.rnd_optimizer.apply_gradients(
            zip(gradients, self.rnd_predict.trainable_variables))

    # 进行一个反向传播
    # @tf.function
    def training_ppo_rnd(self, states, actions, rewards, dones, next_states,
                         mean_obs, std_obs, std_in_rewards):
        obs = tf.stop_gradient(
            self.utils.normalize(next_states, mean_obs, std_obs))
        state_preds = self.rnd_predict(obs)
        state_targets = self.rnd_target(obs)

        with tf.GradientTape() as tape:
            action_probs, ex_values, in_values = self.actor(
                states), self.ex_critic(states), self.in_critic(states)
            old_action_probs, old_ex_values, old_in_values = self.actor_old(
                states), self.ex_critic_old(states), self.in_critic_old(states)
            next_ex_values, next_in_values = self.ex_critic(
                next_states), self.in_critic(next_states)

            loss, critic_loss, dist_entropy, pg_loss = self.get_PPO_loss(
                action_probs, ex_values, old_action_probs, old_ex_values,
                next_ex_values, actions, rewards, dones, state_preds,
                state_targets, in_values, old_in_values, next_in_values,
                std_in_rewards)
        wandb.log({"critic_loss": critic_loss.numpy()}, commit=False)
        wandb.log({"entropy": dist_entropy.numpy()}, commit=False)
        wandb.log({"pg_loss": pg_loss.numpy()}, commit=False)
        gradients = tape.gradient(
            loss, self.actor.trainable_variables +
            self.ex_critic.trainable_variables +
            self.in_critic.trainable_variables)
        self.ppo_optimizer.apply_gradients(
            zip(
                gradients, self.actor.trainable_variables +
                self.ex_critic.trainable_variables +
                self.in_critic.trainable_variables))

    def training_ppo(self, states, actions, rewards, dones, next_states):
        with tf.GradientTape() as tape:
            action_probs, values = self.actor(states), self.ex_critic(states)
            old_action_probs, old_values = self.actor_old(
                states), self.ex_critic_old(states)
            next_values = self.ex_critic(next_states)

            loss, critic_loss, dist_entropy, pg_loss = self.get_loss(action_probs, values, old_action_probs,
                                 old_values, next_values, actions, rewards,
                                 dones)
        wandb.log({"critic_loss": critic_loss.numpy()}, commit=False)
        wandb.log({"entropy": dist_entropy.numpy()}, commit=False)
        wandb.log({"pg_loss": pg_loss.numpy()}, commit=False)
        gradients = tape.gradient(
            loss,
            self.actor.trainable_variables + self.ex_critic.trainable_variables)
        self.ppo_optimizer.apply_gradients(
            zip(
                gradients, self.actor.trainable_variables +
                self.ex_critic.trainable_variables))

    # 更新rnd模型
    def update_rnd(self):
        batch_size = self.minibatch

        # K epochs
        intrinsic_rewards = 0
        for _ in range(self.RND_epochs):
            for obs in self.obs_memory.get_all_tensor().batch(batch_size):
                self.training_rnd(obs, self.obs_memory.mean_obs,
                                  self.obs_memory.std_obs)
        # 训练完之后
        intrinsic_rewards = self.compute_intrinsic_reward(
            self.obs_memory.get_all(), self.obs_memory.mean_obs,
            self.obs_memory.std_obs)

        self.update_obs_normalization_param(self.obs_memory.observations)
        self.update_rwd_normalization_param(intrinsic_rewards)

        # Clear the memory
        self.obs_memory.clear_memory()

    # 更新模型
    def update_ppo(self):
        batch_size = self.minibatch

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states, actions, rewards, dones, next_states in self.memory.get_all_tensor(
            ).batch(batch_size):
                if args.rnd:
                    self.training_ppo_rnd(states, actions, rewards, dones, next_states,
                                    self.obs_memory.mean_obs,
                                    self.obs_memory.std_obs,
                                    self.obs_memory.std_in_rewards)
                else:
                    self.training_ppo(states, actions, rewards, dones, next_states)

        # 清除缓存数据
        self.memory.clear_memory()

        # 更新旧的模型
        self.actor_old.set_weights(self.actor.get_weights())
        self.ex_critic_old.set_weights(self.ex_critic.get_weights())
        if args.rnd:
            self.in_critic_old.set_weights(self.in_critic.get_weights())

    def save_weights(self):
        self.actor.save_weights('{args.game}/actor_ppo', save_format='tf')
        self.actor_old.save_weights('bipedalwalker_w/actor_old_ppo',
                                    save_format='tf')

        self.ex_critic.save_weights('{args.game}/ex_critic_ppo',
                                    save_format='tf')
        self.ex_critic_old.save_weights('{args.game}/ex_critic_old_ppo',
                                        save_format='tf')
        if args.rnd:
            self.in_critic.save_weights('{args.game}/in_critic_ppo',
                                        save_format='tf')
            self.in_critic_old.save_weights('{args.game}/in_critic_old_ppo',
                                            save_format='tf')

    def load_weights(self):
        self.actor.load_weights('{args.game}/actor_ppo')
        self.actor_old.load_weights('{args.game}/actor_old_ppo')

        self.ex_critic.load_weights('{args.game}/ex_critic_ppo')
        self.ex_critic_old.load_weights('{args.game}/ex_critic_old_ppo')
        if args.rnd:
            self.in_critic.load_weights('{args.game}/in_critic_ppo')
            self.in_critic_old.load_weights('{args.game}/in_critic_old_ppo')


def run_inits_episode(env, agent, state_dim, render, n_init_episode):
    env.reset()
    print("INIT EPISODES!")
    for _ in range(n_init_episode):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step(action)
        agent.save_observation(next_state)

        if render:
            env.render()

        if done:
            env.reset()

    agent.update_obs_normalization_param(agent.obs_memory.observations)
    agent.obs_memory.clear_memory()

    return agent


def make_env(gym_id):
    if "MiniGrid" in gym_id:
        env = gymnasium.make(gym_id)
        env = StateBonus(env)
        env = RGBImgPartialObsWrapper(env)
        env = GrayImgObsWrapper(env)
        env = FrameStackWrapper(env, num_stack=8)
        env = NormalizeObsWrapper(env)
        env = MaxStepWrapper(env, 100)
        return env
    else:
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TimeLimit(env, max_episode_steps=2000)
        env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        env = ResizeObservation(env, (80, 80))
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = NormalizedEnv(env, ob=True, ret=False)
        env = gym.wrappers.FrameStack(env, 8)
        return env


def run_episode(env, agent, state_dim, render, training_mode, t_updates,
                n_update):
    ############################################
    state = np.array(env.reset())
    done = False
    total_reward = 0
    eps_time = 0
    action_list = []
    ############################################total_reward

    while not done:
        action = int(agent.act(state))
        action_list.append(action)
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state)
        eps_time += 1
        t_updates += 1
        total_reward += reward

        if training_mode:
            agent.save_eps(state, float(action), float(reward), float(done),
                           next_state)
            agent.save_observation(next_state)

        state = next_state

        if render:
            env.render()

        if training_mode:
            if args.rnd:
                if t_updates % n_update == 0:
                    agent.update_rnd()
                    t_updates = 0

        if done:
            action_var = np.var(action_list)
            del action_list
            return total_reward, eps_time, t_updates, action_var


def main():
    #########################################
    load_weights = False
    save_weights = False
    training_mode = True

    render = False
    n_step_update = 128  # steps before you update the RND
    n_eps_update = 5  # episode before you update the PPO
    n_episode = 10000  # episode you want to run
    n_init_episode = 10
    n_saved = 10  # episode to run before saving the weights

    value_clip = 1.0  # Value clipping
    entropy_coef = 0.1  # entropy loss ratio
    vf_loss_coef = 1.0  # critic loss ratio
    minibatch = 128  # size of batch = n_update / minibatch
    PPO_epochs = 4  # epochs for an update

    gamma = 0.99
    lam = 0.95
    learning_rate = 2.5e-4
    env_name = args.game
    env = make_env(env_name)

    state_dim = list(env.observation_space.shape)
    if 'MiniGrid' in args.game:
        action_dim = args.action_dim
    else:
        action_dim = env.action_space.n
    ##############################################
    runs = wandb.init(project=args.game.split('/')[-1] + '_PPO_' +
                      str(n_episode) if '/' in args.game else args.game +
                      '_PPO_' + str(n_episode),
                      name=args.model + '_PPO',
                      config={
                          'learning_rate': learning_rate,
                          'num_actions': action_dim,
                          'gamma': gamma,
                          'total_episodes': n_episode,
                          'Num_stacking': 8,
                          'num_batches': 1,
                          'epoches': 4
                      },
                      save_code=True,
                      monitor_gym=True)
    config = wandb.config
    agent = Agent(state_dim, action_dim, training_mode, value_clip,
                  entropy_coef, vf_loss_coef, minibatch, PPO_epochs, gamma,
                  lam, learning_rate, n_episode)
    #############################################

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    t_updates = 0

    if training_mode:
        agent = run_inits_episode(env, agent, state_dim, render,
                                  n_init_episode)

    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_updates, action_var = run_episode(
            env, agent, state_dim, render, training_mode, t_updates,
            n_step_update)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(
            i_episode, total_reward, time))

        if i_episode % n_eps_update == 0:
            agent.update_ppo()

        if save_weights:
            if i_episode % n_saved == 0:
                agent.save_weights()
                print('weights saved')

        wandb.log({"rewards": int(total_reward)}, commit=False)
        wandb.log({"action_var": action_var}, commit=False)


if __name__ == '__main__':
    main()