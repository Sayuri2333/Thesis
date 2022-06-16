# replay memory 调整
import numpy as np
import os
import gym
from yaml import parse
# import tensorflow
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
import cv2
from utils import RewardScaling, Normalization

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import wandb
from wandb.keras import WandbCallback

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
parser.add_argument('--episodes', type=int, default=15000, help="length of Replay Memory")
parser.add_argument('--epochs', type=int, default=2, help="epochs on training batch data")
parser.add_argument('--game', type=str, help="Games in Atari")
parser.add_argument('--model', type=str, help="Model we use")
parser.add_argument('--multi_gpu', action='store_true', help='If use multi GPU')
parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
parser.add_argument('--memory_size', type=int, default=1024)
parser.set_defaults(render=False)

args = parser.parse_args()

def resize_input(pic):
    if args.game == 'SeaquestDeterministic-v4':
        pic = pic[7:182, 10:160, :]
    elif args.game == 'BreakoutDeterministic-v4':
        pic = pic
    elif args.game == 'FrostbiteDeterministic-v4':
        pic = pic[10:190, :, :]
    elif args.game == 'BeamRiderDeterministic-v4':
        pic = pic[10:190, :, :]
    elif args.name == 'GopherDeterministic-v4':
        pic = pic[45:225, :, :]
    # RGB to Grey
    pic = cv2.resize(pic, (80, 80))
    pic = pic[:, :, 0:1] * 0.2989 + pic[:, :, 1:2] * 0.5870 + pic[:, :, 2:3] * 0.1140
    return pic

game = gym.make(args.game)
game = gym.wrappers.RecordVideo(game, 'video', episode_trigger = lambda x: x % 10 == 0)

if args.multi_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EPISODES = args.episodes
TEST_EPISODE = 500
LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = args.epochs
BUFFER_SIZE = args.memory_size
GAMMA = 0.99

BATCH_SIZE = args.batch_size
NUM_ACTIONS = game.action_space.n
STEPS_PER_EPOCH = 5
TOTAL_TRAINING_STEP = EPOCHS * (BUFFER_SIZE / BATCH_SIZE) * EPISODES
ENTROPY_LOSS = 0.01
LR = 0.00025  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))

initializer = tf.keras.initializers.Orthogonal(gain=0.1)



# 计算actor网络的loss
def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        # y_true以及y_pred为loss自带的输入，其中，y_true是one-hot向量
        # y_pred为当前网络的action概率分布，old_prob为以前actor网络的action概率分布
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        # 计算两个分布在实际采取的动作上的概率比值
        r = prob/(old_prob + 1e-10)
        # 带有熵损失的actor网络loss，输出的动作概率越接近0.5（越不确定），loss越大
        # 5. Policy Entropy
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


class Agent:
    def __init__(self):
        self.runs = wandb.init(project=args.game + '_PPO_' + str(EPISODES),
                         name = args.model + '_PPO',
                         config = {
                             'learning_rate': LR,
                             'num_actions': NUM_ACTIONS,
                             'Num_Training_episode': EPISODES,
                             'Num_Testing_episode': TEST_EPISODE,
                             'gamma': GAMMA,
                             'Num_stacking': 8,
                             'batch_size': BATCH_SIZE,
                         },
                         save_code=True,
                         monitor_gym=True
                        )
        config = wandb.config
        self.recorder_minp = []
        self.recorder_maxp = []
        self.recorder_minaction = []
        self.state_norm = Normalization(shape=(80, 80, 1))
        self.backbone = eval(args.model)()
        self.actor, self.critic = self.build_actor_critic()
        self.batch = [[], [], [], []]
        self.env = game
        self.Num_stacking = 8
        self.episode = 0
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.path = self.get_path()
        self.gradient_steps = 0
        self.initialization(resize_input(self.env.reset()))

        self.reward_scal = RewardScaling()

    def get_path(self):
        name = 'PPO_Results/' + args.game + '_' + str(args.episodes) + '/' + args.model + '/'
        return name

    # stack state into a state_set
    def initialization(self, state):  # 接收初始状态并初始化state_set
        # 1. state normalization
        state = self.state_norm(state)
        self.state_set = []
        for i in range(self.Num_stacking):
            self.state_set.append(state.copy())

    def skip_and_stack_frame(self, state):  # 将给定state存入state_set并返回最近8 frames
        # 2. state normalization
        state = self.state_norm(state)
        self.state_set.append(state.copy())
        if len(self.state_set) > self.Num_stacking:
            del self.state_set[:-self.Num_stacking]
    
    def build_actor_critic(self):
        backbone = eval(args.model)()
        # actor
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))
        input_actor = Input(shape=(8, 80, 80, 1))
        feature_actor = backbone(input_actor)
        actor_dense = Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), activation='relu')(feature_actor)
        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output', kernel_initializer=initializer)(actor_dense)
        model_actor = Model(inputs=[input_actor, advantage, old_prediction], outputs=[out_actions])
        if args.multi_gpu:
            model_actor = multi_gpu_model(model_actor, 2)
        # 6. Learning Rate Decay
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            LR,
            TOTAL_TRAINING_STEP,
            0.1 * LR,
            power=1)
        # 9. Adam Epsilon Parameter
        # 7. Gradient Clip
        model_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-05, clipnorm=0.5),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        
        # critic
        input_critic = Input(shape=(8, 80, 80, 1))
        feature_critic = backbone(input_critic)
        critic_dense = Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), activation='relu')(feature_critic)
        out_value = Dense(1, kernel_initializer=initializer, activation='relu')(critic_dense)
        model_critic = Model(inputs=[input_critic], outputs=[out_value])
        if args.multi_gpu:
            model_critic = multi_gpu_model(model_critic, 2)
        # 9. Adam Epsilon Parameter
        # 7. Gradient Clip
        model_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-05, clipnorm=0.5), loss='mse')
        return model_actor, model_critic

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0 or self.episode >= EPISODES:
            self.val = False
        else:
            self.val = False
        self.initialization(resize_input(self.env.reset()))
        self.reward = []

    def get_action(self):
        # actor网络使用softmax函数输出概率
        p = self.actor.predict([np.array([np.concatenate([np.expand_dims(self.state_set[i], axis=0) for i in range(self.Num_stacking)], axis=0)]), 
        DUMMY_VALUE, DUMMY_ACTION])
        self.recorder_maxp.append(np.max(p[0]))
        self.recorder_minp.append(np.min(p[0]))
        self.recorder_minaction.append(np.argmin(p[0]))
        if self.val is False:
            # 按actor网络输出的概率进行选择
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            # 直接选择最大概率的
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        # 根据选择的动作生成one-hot动作向量
        action_matrix[action] = 1
        return action, action_matrix, p

    def transform_reward(self):
        # self.reward存储当前episode每一步的reward
        # 测试与训练的区别在于action选择的策略不同
        print('Reward of episode ' + str(self.episode) + ' is: ' + str(sum(self.reward)))
        if self.val is True:
            # self.val在episode结束时进行判定，决定下一个episode是否为测试用
            wandb.log({'validation_reward': np.array(self.reward).sum()})
        else:
            # 这个就是普通的训练用episode的总reward
            wandb.log({'episode_reward': np.array(self.reward).sum()})
        wandb.log({'mean_max_p': np.array(self.recorder_maxp).mean()})
        wandb.log({'mean_min_p': np.array(self.recorder_minp).mean()})
        wandb.log({'min_action_var': np.array(self.recorder_minaction).var()})
        self.recorder_maxp = []
        self.recorder_minp = []
        self.recorder_minaction = []
        # 4. reward scaling
        for i in range(len(self.reward)):
            self.reward[i] = self.reward_scal(self.reward[i])
        self.reward_scal.reset()
        # 对于倒数第二步开始往前数直到第一步
        for j in range(len(self.reward) - 2, -1, -1):
            # 这一步的reward等于实际reward加上下一步的reward乘上GAMMA
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        done = False
        tmp_batch = [[], [], []]
        step = 0
        while not done:
            action, action_matrix, predicted_action = self.get_action()
            observation, reward, done, _ = self.env.step(action)
            step += 1
            self.reward.append(reward)
            # 存储当前状态，当前执行动作one-hot向量以及当前actor网络对于状态输出的动作概率向量
            tmp_batch[0].append(np.concatenate([np.expand_dims(self.state_set[i], axis=0) for i in range(self.Num_stacking)], axis=0))
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            if len(tmp_batch[0]) > BUFFER_SIZE:
                del tmp_batch[0][:-BUFFER_SIZE]
                del tmp_batch[1][:-BUFFER_SIZE]
                del tmp_batch[2][:-BUFFER_SIZE]
                del self.reward[:-BUFFER_SIZE]
            # 更新当前state到下一步
            self.skip_and_stack_frame(resize_input(observation))

        if done:
            print('This episode steps: ' + str(step))
            step = 0
            # 跑完一个episode之后，存储s, a, 实际v(s)以及预测a
            self.transform_reward()
            if self.val is False:
                for i in range(len(tmp_batch[0])):
                    obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                    r = self.reward[i]
                    self.batch[0].append(obs)
                    self.batch[1].append(action)
                    self.batch[2].append(pred)
                    self.batch[3].append(r)
            del tmp_batch
            self.reset_env()
        # cut old
        if len(self.batch[0]) > BUFFER_SIZE:
            del self.batch[0][:-BUFFER_SIZE]
            del self.batch[1][:-BUFFER_SIZE]
            del self.batch[2][:-BUFFER_SIZE]
            del self.batch[3][:-BUFFER_SIZE]
        obs, action, pred, reward = np.array(self.batch[0]), np.array(self.batch[1]), np.array(self.batch[2]), np.reshape(np.array(self.batch[3]), (len(self.batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        while self.episode < EPISODES:
            # 跑n个episode，记录需要的信息
            obs, action, pred, reward = self.get_batch()
            # 因为要基于这个batch做更新，所以把基于改进前策略的动作概率向量当作old_pred
            old_prediction = pred
            # 使用critic网络预测value值
            pred_values = self.critic.predict(obs)
            # 实际的v(s)减去预测的v(s)得到adv
            advantage = reward - pred_values
            # 1. Advantage Normalization
            advantage = (advantage - np.mean(advantage)) / np.std(advantage)

            _ = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False, callbacks=[WandbCallback()])
            _ = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False, callbacks=[WandbCallback()])

            self.gradient_steps += 1
        self.reset_env()
        episode_reward = []
        self.actor.save(self.path + '/model.h5')
        while self.episode < EPISODES + TEST_EPISODE:
            action, _, _ = self.get_action()
            observation, reward, done, _ = self.env.step(action)
            self.reward.append(reward)
            self.skip_and_stack_frame(resize_input(observation))
            if done:
                #wandb
                print('Test Episode ' + str(self.episode - EPISODES) + ' reward: ' + str(sum(self.reward)))
                episode_reward.append(sum(self.reward))
                self.reward = []
                self.reset_env()
        print('Average Test Reward: ' + str(sum(episode_reward) / len(episode_reward)))
        self.runs.finish()
if __name__ == '__main__':
    ag = Agent()
    ag.run()