import random
import numpy as np
from numpy.core.fromnumeric import squeeze
import argparse
import cv2
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
from model import DQN, DRQN, Conv_Transformer, ConvTransformer, ViTrans, MFCA, MultiscaleTransformer
from utils import NoisyDense
import gym
import os
from PER import Memory

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
parser.add_argument('--memory', type=int, default=20000, help="length of Replay Memory")
parser.add_argument('--training_steps', type=int, default=500000, help="training steps")
parser.add_argument('--with_PER', action='store_true', help="Use Prioritized Experience Replay")
parser.add_argument('--game', type=str, help="Games in Atari")
parser.add_argument('--model', type=str, help="Model we use")
parser.add_argument('--multi_gpu', action='store_true', help='If use multi GPU')
parser.add_argument('--dueling', action='store_true', help='If use dueling architecture')
parser.add_argument('--noisy', action='store_true', help='If use noisy network')
parser.set_defaults(render=False)

args = parser.parse_args()

def resize_input(pic):
    if args.game == 'SeaquestDeterministic-v4':
        pic = pic[7:182, 10:160, :]
    elif args.game == 'BreakoutDeterministic-v4':
        pic = pic[20:200, :, :]
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

if args.multi_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class DARQN:
    def __init__(self):
        #wandb init
        self.run = wandb.init(project=args.game + '_' + str(args.training_steps),
                         name = args.model,
                         config = {
                             'learning_rate': 0.00025,
                             'loss_function': 'mse',
                             'num_actions': game.action_space.n,
                             'Num_Exploration': 5000,
                             'Num_Training': args.training_steps,
                             'Num_Testing': 100000,
                             'gamma': 0.99,
                             'first_epsilon': 0.9,
                             'final_epsilon': 0.1,
                             'Num_stacking': 8,
                             'batch_size': 32,
                             'replay_memory': args.memory
                         }
                        )
        config = wandb.config
        self.progress = ''
        self.Num_action = config.num_actions
        
        self.Num_Exploration = config.Num_Exploration
        self.Num_Training = config.Num_Training
        self.Num_Testing = config.Num_Testing
        self.test_score = []

        self.learning_rate = config.learning_rate
        self.gamma = config.gamma

        self.first_epsilon = config.first_epsilon
        self.final_epsilon = config.final_epsilon
        self.epsilon = self.first_epsilon
        self.Num_update_target = 10
        self.Is_train = True
        self.load_path = './All_Results/'+args.game + '_' + str(args.training_steps)+'/'+args.model+'/'

        self.step = 1
        self.score = 0
        self.episode = 0

        self.state_set = []
        self.Num_stacking = config.Num_stacking

        self.Num_replay_memory = config.replay_memory
        self.Num_batch = config.batch_size
        if args.with_PER:
            self.replay_memory = Memory(capacity=config.replay_memory)  # DQN通过Memory类间接操控SumTree
        else:
            self.replay_memory = []
        
        self.maxQ = 0
        self.Qdiff = 0
        self.step_old = 0

        self.input_state, self.output, self.model = self.network()
        self.input_target_state, self.output_target, self.target_model = self.network()
        network_weights = self.model.trainable_weights
        target_network_weights = self.target_model.trainable_weights

        self.update_target_network = [target_network_weights[i].assign(network_weights[i]) for i in range(len(target_network_weights))]
        self.sess = self.init_sess()
                    
    def main(self):
        # 从game对象中获得state
        game_state = game.reset()
        game_state = resize_input(game_state)

        # Initialization
        state = self.initialization(game_state)  # 初始化state_set并获得state
        stacked_state = self.skip_and_stack_frame(state)

        while True:
            self.progress = self.get_progress()  # 判断程序在什么阶段

            action = self.select_action(stacked_state)  # 根据最近 frames选择动作

            # 执行动作获得状态并stack
            next_state, reward, terminal, _ = game.step(action)
            # reward scaling
            self.score += reward
            if reward > 0: reward = 1
            elif reward < 0: reward = -1
            next_state = resize_input(next_state)
            stacked_next_state = self.skip_and_stack_frame(next_state)

            # 存入经验回放池
            self.experience_replay(stacked_state, action, reward, stacked_next_state, terminal)

            # 如果在训练progress
            if self.progress == 'Training':
                # 到了更新target network的时候
                if self.episode % self.Num_update_target == 0:
                    self.sess.run(self.update_target_network)

                # 训练一手
                self.train(self.model, self.target_model)
                # 保存模型
                self.save_model()  # 虽然每一步都在调用 但实际上只有在刚好训练完成时才保存模型

            # 更新状态奖励以及步数
            stacked_state = stacked_next_state
            self.step += 1

            # 如果到了终止状态 那就将stacked_state恢复到初始状态 并输出图像
            if terminal:
                game_state = game.reset()
                game_state = resize_input(game_state)
                stacked_state = self.if_terminal(game_state)
            if self.progress == 'Finished':
                print('Average score in Testing: ' + str(sum(self.test_score) / len(self.test_score)))
                print('Finished!')
                self.run.finish()
                break
    
    def init_sess(self):
        # 初始化
        config = tf.ConfigProto()
        # jit_level = tf.OptimizerOptions.ON_1
        # config.graph_options.optimizer_options.global_jit_level = jit_level
        config.gpu_options.per_process_gpu_memory_fraction =0.5
        config.gpu_options.allow_growth = True

        sess = tf.InteractiveSession(config=config)
        tf.keras.backend.set_session(sess)
        # 新建用于存储数据的文件夹
        if not os.path.exists(self.load_path):
            os.makedirs(self.load_path)

        init = tf.global_variables_initializer()
        sess.run(init)  # 初始化变量

        check_save = input('Load Model? (1=yes/2=no): ')

        if check_save == '1':
            # Restore variables from disk.
            self.model = load_model(self.load_path+'/model.h5')
            print("Model restored.")
            check_train = input('Testing or Training? (1=Testing / 2=Training): ')
            if check_train == '1':  # 如果只需要推理
                self.Num_Exploration = 0
                self.Num_Training = 0  # 探索和训练需要的步数都设置为0

        return sess

    def initialization(self, state):  # 接收初始状态并初始化state_set
        state = (state - 0) / 255.0
        self.state_set = []
        for i in range(self.Num_stacking):
            self.state_set.append(state.copy())

        return state  # 返回初始状态啥都不做之后的状态
    
    def skip_and_stack_frame(self, state):  # 将给定state存入state_set并返回最近2 frames
        state = (state - 0) / 255.0
        self.state_set.append(state.copy())
        if len(self.state_set) > self.Num_stacking:
            # self.state_set = self.state_set[-self.Num_stacking:]
            del self.state_set[:-self.Num_stacking]
        state_in = np.concatenate([np.expand_dims(self.state_set[i], axis=0) for i in range(self.Num_stacking)], axis=0)
        return state_in
    
    def get_progress(self):  # 根据self.step判断当前处于什么阶段
        progress = ''
        if self.step <= self.Num_Exploration:
            progress = 'Exploring'
        elif self.step <= self.Num_Exploration + self.Num_Training:
            progress = 'Training'
        elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
            progress = 'Testing'
        else:
            progress = 'Finished'

        return progress
    
    def network(self):
        input_state, feature = eval(args.model)()
        if args.noisy:
            if args.dueling:
                dense_action = NoisyDense(512, activation='relu')(feature)
                dense_value = NoisyDense(512, activation='relu')(feature)
                out_action = NoisyDense(self.Num_action)(dense_action)
                out_value = NoisyDense(1)(dense_value)
                action_scaled = out_action - tf.reduce_mean(out_action, axis=1, keepdims=True)
                out = out_value + action_scaled
            elif args.model == 'DRQN':
                out = NoisyDense(self.Num_action)(feature)
            else:
                dense = NoisyDense(512, activation='relu')(feature)
                out = NoisyDense(self.Num_action)(dense)
        else:
            if args.dueling:
                dense_action = Dense(512, activation='relu')(feature)
                dense_value = Dense(512, activation='relu')(feature)
                out_action = Dense(self.Num_action)(dense_action)
                out_value = Dense(1)(dense_value)
                action_scaled = out_action - tf.reduce_mean(out_action, axis=1, keepdims=True)
                out = out_value + action_scaled
            elif args.model == 'DRQN':
                out = Dense(self.Num_action)(feature)
            else:
                dense = Dense(512, activation='relu')(feature)
                out = Dense(self.Num_action)(dense)
        model = Model(inputs=input_state, outputs=out)
        if args.multi_gpu:
            multi_model = multi_gpu_model(model, 2)
            if args.dueling:
                multi_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-02, clipnorm=10))
            else:
                multi_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-02))
            return input_state, out, multi_model
        else:
            if args.dueling:
                model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-02, clipnorm=10))
            else:
                model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-02))
            return input_state, out, model
        
    def select_action(self, stacked_state):  # 根据给定state以及当前阶段选择动作
        action = -1

        # 探索阶段随机选择
        if self.progress == 'Exploring':
            # 随机选
            action_index = random.randint(0, self.Num_action - 1)
            action = action_index

        # training阶段使用ε-greedy方法选择
        elif self.progress == 'Training':
            if args.noisy:
                Q_value = self.output.eval(feed_dict={self.input_state: [stacked_state]})
                action_index = np.argmax(Q_value)
                action = action_index
                self.maxQ = np.max(Q_value)
                wandb.log({'maxQ': self.maxQ}, step=self.step)
                self.Qdiff = np.max(Q_value) - np.min(Q_value)
                wandb.log({'Qdiff': self.Qdiff}, step=self.step)
            else:
                if random.random() < self.epsilon:
                    # 随机选
                    action_index = random.randint(0, self.Num_action - 1)
                    action = action_index
                else:
                    # 最优选
                    Q_value = self.output.eval(feed_dict={self.input_state: [stacked_state]})
                    action_index = np.argmax(Q_value)
                    action = action_index
                    self.maxQ = np.max(Q_value)
                    wandb.log({'maxQ': self.maxQ}, step=self.step)
                    self.Qdiff = np.max(Q_value) - np.min(Q_value)
                    wandb.log({'Qdiff': self.Qdiff}, step=self.step)
                

                # ε的值随着每次select递减
                if self.epsilon > self.final_epsilon:
                    self.epsilon -= self.first_epsilon / self.Num_Training

        elif self.progress == 'Testing':
            # 测试阶段直接选择最优动作
            Q_value = self.output.eval(feed_dict={self.input_state: [stacked_state]})
            action_index = np.argmax(Q_value)
            action = action_index
            self.maxQ = np.max(Q_value)

            self.epsilon = 0
        return action
    
    def experience_replay(self, state, action, reward, next_state, terminal):
        if not args.with_PER:
            if len(self.replay_memory) > self.Num_replay_memory:
                del self.replay_memory[0]
            self.replay_memory.append([state, action, reward, next_state, terminal])
        else:
            self.replay_memory.store([state, action, reward, next_state, terminal])


    def train(self, model, target_model):
        # sample several episodes
        if not args.with_PER:
            minibatch = random.sample(self.replay_memory, self.Num_batch)
        else:
            tree_idx, minibatch, ISWeights = self.replay_memory.sample(self.Num_batch)
        # get s,a,pre_a,r,s',terminal
        state_batch = np.array([batch[0] for batch in minibatch])
        action_batch = np.array([batch[1] for batch in minibatch])
        reward_batch = np.array([batch[2] for batch in minibatch])
        next_state_batch = np.array([batch[3] for batch in minibatch])
        terminal_batch = np.array([batch[4] for batch in minibatch])
        # predict Q(s) with current model
        Q_batch = model.predict(state_batch)
        origin_Q_batch = Q_batch.copy()
        # predict Q'(s') with target model
        target_Q_batch = target_model.predict(next_state_batch)
        # predict Q(s') with current model
        Q_next_batch = model.predict(next_state_batch)
        next_action_batch = np.argmax(Q_next_batch, axis=1)
        # set Q(s,a) = r + gamma * Q'(s', a*), a* based on current model
        Q_batch[range(len(state_batch)), action_batch] = reward_batch + self.gamma * (1 - terminal_batch) * target_Q_batch[range(len(target_Q_batch)),next_action_batch ]
        if not args.with_PER:
            # train on this episode
            _ = model.fit(x=state_batch, y=Q_batch, batch_size=self.Num_batch, verbose = 0, callbacks=[WandbCallback()])
        else:
            # calculate abs TD error
            abs_error = np.sum(np.abs(origin_Q_batch-Q_batch), axis=1)
            # train on this episode
            _ = model.fit(x=state_batch, y=Q_batch, batch_size=self.Num_batch, sample_weight=ISWeights, verbose = 0, callbacks=[WandbCallback()])
            self.replay_memory.batch_update(tree_idx, abs_error)
    
    def save_model(self):
        # 保存网络到本地
        if self.step == self.Num_Exploration + self.Num_Training:
            self.model.save(self.load_path + '/model.h5')
            print("Model saved")
    
    def if_terminal(self, game_state):
        # Show Progress
        print('Step: ' + str(self.step) + ' / ' +
              'Episode: ' + str(self.episode) + ' / ' +
              'Progress: ' + self.progress + ' / ' +
              'Epsilon: ' + str(self.epsilon) + ' / ' +
              'Score: ' + str(self.score)
              
             )

        if self.progress != 'Exploring':
            self.episode += 1
        wandb.log({'return_per_episode': self.score}, step=self.step)
        if self.progress == 'Testing':
            self.test_score.append(self.score)
        self.score = 0
        # If game is finished, initialize the state
        state = self.initialization(game_state)
        stacked_state = self.skip_and_stack_frame(state)

        return stacked_state

if __name__ == '__main__':
    agent = DARQN()
    agent.main()