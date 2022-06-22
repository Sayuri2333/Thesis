import numpy as np
import os
import gym
from yaml import parse
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras import activations
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import Sequential, Model, load_model
from tensorflow.compat.v1.keras.layers import concatenate, Add, Multiply, Permute, Softmax, AveragePooling2D, MaxPooling2D, Convolution2D, LeakyReLU, add, Reshape, Lambda, Conv2D, LSTMCell, LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, multiply, Concatenate, Flatten, Activation, dot, Dot, Dropout
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.compat.v1.keras import losses
from model_ppo import DQN, DRQN, Conv_Transformer, ConvTransformer, ViTrans, MFCA, MultiscaleTransformer
import argparse
from Atari_Warppers import MaxAndSkipEnv, SyncVectorEnv, FireResetEnv, ResizeObservation, NoopResetEnv, NormalizeObservation, NormalizeReward, NormalizedEnv, ClipRewardEnv, EpisodicLifeEnv

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import wandb

def multi_gpu_model(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,
                                axis=0, name=name))
    return Model(model.inputs, merged)

parser  = argparse.ArgumentParser(description='Training parameters')
# 
parser.add_argument('--steps', type=int, default=1024000, help="length of Replay Memory")
parser.add_argument('--epochs', type=int, default=4, help="epochs on training batch data")
parser.add_argument('--game', type=str, default='ALE/Breakout-v5', help="Games in Atari")
parser.add_argument('--model', type=str, default='DQN', help="Model we use")
parser.add_argument('--multi_gpu', action='store_true', help='If use multi GPU')
parser.add_argument('--num_steps', type=int, default=256)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--num_minibatches', type=int, default=4)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--clip_coef', type=float, default=0.1)
parser.set_defaults(render=False)

args = parser.parse_args()

def make_env(gym_id, seed, idx, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger = lambda x: x % 100 == 0)
        env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = ResizeObservation(env, (80, 80))
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = NormalizedEnv(env, ob=True, ret=True)
        env = gym.wrappers.FrameStack(env, 8)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# game = gym.wrappers.FrameStack(game, num_stack=8)

# test game
test_game = gym.make(args.game)
test_game = gym.wrappers.RecordVideo(test_game, f"videos/{args.game.split('/')[-1]}", episode_trigger = lambda x: x % 10 == 0)
# test_game = MaxAndSkipEnv(test_game, skip=4)
if "FIRE" in test_game.unwrapped.get_action_meanings():
    test_game = FireResetEnv(test_game)
test_game = ResizeObservation(test_game, (80, 80))
test_game = gym.wrappers.GrayScaleObservation(test_game, keep_dim=True)
test_game = gym.wrappers.FrameStack(test_game, 8)
# test_game = NormalizedEnv(test_game, ob=True, ret=False)

# Initialize runningmean and std for train and test game
test_game.reset()
for i in range(1):
    done = False
    step = 0
    while not done:
        step += 1
        random_action = test_game.action_space.sample()
        pic, reward, done, _ = test_game.step(random_action)
    test_game.reset()

# game.reset()
# pic = None
# for i in range(2000):
#     done = False
#     step = 0
#     step += 1
#     random_action = game.action_space.sample()
#     pic, reward, done, _ = game.step(random_action)

# OBS = pic

if args.multi_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

STEPS = args.steps
TEST_STEPS = 50000
LOSS_CLIPPING = 0.2
EPOCHS = args.epochs
GAMMA = 0.99

BATCH_SIZE = args.num_envs * args.num_steps
MINIBATCH_SIZE = BATCH_SIZE // args.num_minibatches
print(MINIBATCH_SIZE)
NUM_UPDATES = STEPS // BATCH_SIZE
NUM_ACTIONS = test_game.action_space.n
ENTROPY_LOSS = 0.01
LR = 0.00025  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))
ENV_DUMMY_ACTION, ENV_DUMMY_VALUE = np.zeros((args.num_envs, NUM_ACTIONS)), np.zeros((args.num_envs, 1))

initializer = tf.keras.initializers.Orthogonal(gain=0.1)



# 计算actor网络的loss
def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        print(f'y_true_1: {y_true}')
        # old_prediction 以前网络输出的p值 y_pred 以前网络输出的动作的one-hot向量
        # y_true以及y_pred为loss自带的输入
        # y_pred为当前网络的action概率分布，old_prob为以前actor网络的action概率分布
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        # 计算两个分布在实际采取的动作上的概率比值
        r = prob/(old_prob + 1e-10)
        # 带有熵损失的actor网络loss，输出的动作概率越接近0.5（越不确定），loss越大
        # 5. Policy Entropy
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss

# 计算critic网络的loss
def adjust_MSE_loss(value):
    def loss(y_true, y_pred):
        v_loss_unclipped = (y_true - y_pred) ** 2
        v_clipped = value + K.clip(y_pred - value, min_value=-args.clip_coef, max_value=args.clip_coef)
        v_loss_clipped = (y_true - v_clipped) ** 2
        v_loss_max = K.maximum(v_loss_clipped, v_loss_unclipped)
        return 0.5 * K.mean(v_loss_max)
    return loss

class Agent:
    def __init__(self):
        self.runs = wandb.init(project=args.game.split('/')[-1] + '_PPO_' + str(STEPS) if '/' in args.game else args.game + '_PPO_' + str(STEPS),
                         name = args.model + '_PPO',
                         config = {
                             'learning_rate': LR,
                             'num_actions': NUM_ACTIONS,
                             'Num_Testing_steps': TEST_STEPS,
                             'gamma': GAMMA,
                             'total_steps': STEPS,
                             'Num_stacking': 4,
                             'num_envs': BATCH_SIZE,
                             'num_steps': args.num_steps,
                             'num_envs': args.num_envs
                         },
                         save_code=True,
                         monitor_gym=True
                        )
        config = wandb.config
        self.recorder_minp = []
        self.recorder_maxp = []
        self.recorder_minaction = []
        self.step = 0
        self.backbone = eval(args.model)()
        self.Num_stacking = config.Num_stacking
        self.actor, self.critic = self.build_actor_critic()
        self.batch = [[], [], [], []]
        self.env = game
        self.episode = 0
        self.reward = []
        self.reward_over_time = []
        self.path = self.get_path()
        self.obs =self.env.reset()
        self.obs = self.obs
        self.lmbda = args.lmbda

    def get_path(self):
        name = 'PPO_Results/' + args.game + '_' + str(args.steps) + '/' + args.model + '/'
        return name
    
    def build_actor_critic(self):
        backbone = eval(args.model)()
        # actor
        advantage = Input(shape=(1,))
        value = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))
        input_actor = Input(shape=(self.Num_stacking, 80, 80, 1))
        feature_actor = backbone(input_actor)
        actor_dense = Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), activation='relu')(feature_actor)
        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output', kernel_initializer=initializer)(actor_dense)
        model_actor = Model(inputs=[input_actor, advantage, old_prediction], outputs=[out_actions])
        if args.multi_gpu:
            model_actor = multi_gpu_model(model_actor, 2)
        # 6. Learning Rate Decay
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            LR,
            10000,
            0.1 * LR,
            power=1)
        # 9. Adam Epsilon Parameter
        # 7. Gradient Clip
        model_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-05, clipnorm=0.5),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        
        # critic
        input_critic = Input(shape=(self.Num_stacking, 80, 80, 1))
        feature_critic = backbone(input_critic)
        critic_dense = Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), activation='relu')(feature_critic)
        out_value = Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), activation='relu')(critic_dense)
        model_critic = Model(inputs=[input_critic, value], outputs=[out_value])
        if args.multi_gpu:
            model_critic = multi_gpu_model(model_critic, 2)
        # 9. Adam Epsilon Parameter
        # 7. Gradient Clip
        learning_rate_fn1 = tf.keras.optimizers.schedules.PolynomialDecay(
            LR,
            10000,
            0.1 * LR,
            power=1)
        model_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn1, epsilon=1e-05, clipnorm=0.5),
        loss = [adjust_MSE_loss(value=value)]
        )
        return model_actor, model_critic

    def get_action(self, obs):
        # actor网络使用softmax函数输出概率
        # p = (num_envs, action)
        p = self.actor.predict([obs, ENV_DUMMY_VALUE, ENV_DUMMY_ACTION])
        self.recorder_maxp.append(np.mean(np.max(p, axis=-1)))
        self.recorder_minp.append(np.mean(np.min(p, axis=-1)))
        # self.recorder_minaction.append(np.argmin(p[0]))
        # action = (num_envs) -> array
        action = [np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[i])) for i in range(args.num_envs)]
        # action_matrix = (num_envs, action)
        action_matrix = np.zeros((args.num_envs, NUM_ACTIONS))
        # 根据选择的动作生成one-hot动作向量
        for i in range(args.num_envs):
            action_matrix[i, action[i]] = 1
        return action, action_matrix, p

    def transform_reward(self, rewards, dones, obses):
        # self.reward存储当前episode每一步的reward
        # 测试与训练的区别在于action选择的策略不同

        # print('Reward of episode ' + str(self.episode) + ' is: ' + str(sum(self.reward)))
        # # 这个就是普通的训练用episode的总reward
        # wandb.log({'episode_reward': np.array(self.reward).sum()}, step=self.step)
        wandb.log({'mean_max_p': np.array(self.recorder_maxp).mean()}, step=self.step)
        wandb.log({'mean_min_p': np.array(self.recorder_minp).mean()}, step=self.step)
        # wandb.log({'min_action_var': np.array(self.recorder_minaction).var()}, step=self.step)
        self.recorder_maxp = []
        self.recorder_minp = []
        # self.recorder_minaction = []
        last_v = self.critic.predict([self.obs, ENV_DUMMY_VALUE])
        # values = (num_steps+1, num_envs, 1)
        values = []
        for obs in obses:
            values.append(self.critic.predict([obs, ENV_DUMMY_VALUE]))
        values.append(last_v)
        rewards = [np.expand_dims(rewards[i], axis=-1) for i in range(len(rewards))]
        dones = [np.expand_dims(dones[i], axis=-1) for i in range(len(dones))]
        gae = np.zeros((args.num_envs, 1))
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * self.lmbda * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
        # returns = (num_steps+1, num_envs, 1)
        return returns

    def get_batch(self):
        obses = []
        actions = []
        preds = []
        rewards = []
        dones = []
        for i in range(args.num_steps):
            action, action_matrix, predicted_action = self.get_action(self.obs)
            next_obs, reward, done, info = self.env.step(action)
            self.step += args.num_envs * 1
            obses.append(self.obs)
            actions.append(action_matrix)
            preds.append(predicted_action)
            rewards.append(reward)
            dones.append(done)
            self.obs = next_obs
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={self.step}, episodic_return={item['episode']['r']}")
                    wandb.log({"episodic_return": item["episode"]["r"]}, step=self.step)
                    wandb.log({"episodic_length": item["episode"]["l"]}, step=self.step)
                    break
        # obses = (num_steps, num_envs, num_stack, H, W, C)
        # actions = (num_steps, num_envs, action)
        # preds = (num_steps, num_envs, action)
        # rewards = (num_steps, num_envs)
        # dones = (num_steps, num_envs)
        rewards = self.transform_reward(rewards, dones, obses)
        # rewards = (num_steps, num_envs, 1)
        

        # reshape
        obses = np.array(obses).reshape((args.num_steps * args.num_envs, self.Num_stacking, 80, 80, 1))
        actions = np.array(actions).reshape((args.num_steps * args.num_envs, NUM_ACTIONS))
        preds = np.array(preds).reshape((args.num_steps * args.num_envs, NUM_ACTIONS))
        rewards = np.array(rewards).reshape((args.num_steps * args.num_envs, 1))
        return obses, actions, preds, rewards

    def run(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for i in range(NUM_UPDATES):
            obses, actions, preds, rewards = self.get_batch()
            print("max obses: " + str(np.max(obses)))
            print("min obses: " + str(np.min(obses)))
            pred_values = self.critic.predict([obses, np.zeros((args.num_envs * args.num_steps, 1))])
            wandb.log({"average_critic_score": np.mean(pred_values)}, step=self.step)
            advantages = rewards - pred_values
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages)+ 1e-8)
            actor_result = self.actor.fit([obses, advantages, preds], [actions], batch_size=MINIBATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            wandb.log({'actor_loss': np.mean(actor_result.history['loss'])}, step=self.step)
            critic_result = self.critic.fit([obses, pred_values], [rewards], batch_size=MINIBATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            wandb.log({'critic_loss': np.mean(critic_result.history['loss'])}, step=self.step)            
            # b_inds = np.arange(BATCH_SIZE)
            # actor_loss_list = []
            # critic_loss_list = []
            # for epoch in range(EPOCHS):
            #     np.random.shuffle(b_inds)
            #     for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            #         end = start + MINIBATCH_SIZE
            #         mb_inds = b_inds[start:end]
            #         batch_obses, batch_actions, batch_preds, batch_rewards, batch_pred_values = obses[mb_inds], actions[mb_inds], preds[mb_inds], rewards[mb_inds], pred_values[mb_inds]
            #         batch_advantage = batch_rewards - batch_pred_values
            #         batch_advantage = (batch_advantage - np.mean(batch_advantage)) / (np.std(batch_advantage)+ 1e-8)
            #         actor_result = self.actor.fit([batch_obses, batch_advantage, batch_preds], [batch_actions], batch_size=MINIBATCH_SIZE, epochs=1, verbose=False)
            #         critic_result = self.critic.fit([batch_obses, batch_pred_values], [batch_rewards], batch_size=MINIBATCH_SIZE, epochs=1, verbose=False)
            #         actor_loss_list.append(actor_result.history['loss'])
            #         critic_loss_list.append(critic_result.history['loss'])
                    
            # wandb.log({'actor_loss': np.mean(np.array(actor_loss_list))}, step=self.step)
            # wandb.log({'critic_loss': np.mean(np.array(critic_loss_list))}, step=self.step)
        
        self.env = test_game
        obs = self.env.reset()
        self.reward = []
        episode_reward = []
        self.actor.save(self.path + '/model.h5')
        print('saved!')
        # use test_game to test
        while self.step < STEPS + TEST_STEPS:
            action, _, _ = self.actor.predict([np.array([obs]), DUMMY_VALUE, DUMMY_ACTION])
            obs, reward, done, _ = self.env.step(action)
            self.step += 1
            self.reward.append(reward)
            if done:
                #wandb
                print('Test Episode ' + str(self.episode) + ' reward: ' + str(sum(self.reward)))
                episode_reward.append(sum(self.reward))
                obs = self.env.reset()
                self.reward = []
        print('Average Test Reward: ' + str(sum(episode_reward) / len(episode_reward)))
        self.runs.finish()

if __name__ == '__main__':
    game = SyncVectorEnv(
        [make_env(args.game, 1 + i, i, args.game.split('/')[-1]) for i in range(args.num_envs)]
    )
    ag = Agent()
    ag.run()