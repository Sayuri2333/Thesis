import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling3D, Conv3D, GlobalAveragePooling2D, concatenate, add, Multiply, Permute, Softmax, AveragePooling2D, MaxPooling2D, Convolution2D, LeakyReLU, Reshape, Lambda, Conv2D, LSTMCell, LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, multiply, Concatenate, Flatten, Activation, dot, Dot, Dropout
from utils_tf2 import TemporalToChannel, LocalSpaceTransformerBlock, SinusoidalPositionEmbedding, TransformerBlock, VisionTransformerBlock, CreatePatches, Add_Embedding_Layer, TemporalEmbedding, ConvTransformerBlock, multiFocusConvAttention, MultiscaleTransformerBlock, DownSampleTransformerBlock, SpatialEmbedding, SpaceTimeLocalTransformerBlock


initializer = tf.keras.initializers.Orthogonal(gain=1.0)

def DQN():
    input_state = Input(shape=(8,80,80,1))
    input_state1 = TemporalToChannel()(input_state)
    conv1 = Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=initializer)(input_state1)
    conv2 = Conv2D(64, 4, (2,2), activation='relu', padding='same', kernel_initializer=initializer)(conv1)
    conv3 = Conv2D(64, 3, (1,1), activation='relu', padding='same', kernel_initializer=initializer)(conv2)
    flat = Flatten()(conv3)
    model = Model(inputs=input_state, outputs=flat)
    return model

def DRQN():
    input_state = Input(shape=(8,80,80,1))
    conv1 = TimeDistributed(Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=initializer))(input_state)
    conv2 = TimeDistributed(Conv2D(64, 4, (2,2), activation='relu', padding='same', kernel_initializer=initializer))(conv1)
    conv3 = TimeDistributed(Conv2D(64, 3, (1,1), activation='relu', padding='same', kernel_initializer=initializer))(conv2)
    flat = TimeDistributed(Flatten())(conv3)
    LSTMUnit = LSTM(512, activation='relu')(flat)
    model = Model(inputs=input_state, outputs=LSTMUnit)
    return model

def Conv_Transformer():
    input_state = Input(shape=(8,80,80,1))
    conv1 = TimeDistributed(Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=initializer))(input_state)
    conv2 = TimeDistributed(Conv2D(64, 4, (2,2), activation='relu', padding='same', kernel_initializer=initializer))(conv1)
    conv3 = TimeDistributed(Conv2D(128, 3, (2,2), activation='relu', padding='same', kernel_initializer=initializer))(conv2)
    GAP = TimeDistributed(GlobalAveragePooling2D())(conv3)
    with_pos = SinusoidalPositionEmbedding(128)(GAP)
    trans = TransformerBlock(num_heads=4, mlp_dim=512)(with_pos)
    trans2 = TransformerBlock(num_heads=4, mlp_dim=512)(trans)
    last_frame = Lambda(lambda x: x[:, -1, :])(trans2)
    model = Model(inputs=input_state, outputs=last_frame)
    return model

def ConvTransformer():
    input_state = Input(shape=(8,80,80,1))
    conv1 = TimeDistributed(Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=initializer))(input_state)
    conv2 = TimeDistributed(Conv2D(64, 4, (2,2), activation='relu', padding='same', kernel_initializer=initializer))(conv1)
    with_pos = TemporalEmbedding(output_dim=64)(conv2)
    convtrans = ConvTransformerBlock(num_heads=4)(with_pos)
    convtrans2 = ConvTransformerBlock(num_heads=4)(convtrans)
    last_frame = Lambda(lambda x : x[:,-1,:,:,:])(convtrans2)
    conv3 = Conv2D(128, 3, (2,2), activation='relu', padding='same')(last_frame)
    GAP = GlobalAveragePooling2D()(conv3)
    model = Model(inputs=input_state, outputs=GAP)
    return model

def ViTrans():
    input_state = Input(shape=(8, 80, 80, 1))
    conv1 = TimeDistributed(Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=initializer))(input_state)
    patches = CreatePatches(4)(conv1)
    flat = TimeDistributed(TimeDistributed(Flatten()))(patches)
    emb = TimeDistributed(TimeDistributed(Dense(128, use_bias=False)))(flat)
    with_token = TimeDistributed(Add_Embedding_Layer(num_patches=25, d_model=128))(emb)
    vit1 = TimeDistributed(VisionTransformerBlock(num_heads=4, mlp_dim=512))(with_token)
    vit2 = TimeDistributed(VisionTransformerBlock(num_heads=4, mlp_dim=512))(vit1)
    total_feature = Lambda(lambda x: x[:, :, 0, :])(vit2)
    with_pos = SinusoidalPositionEmbedding(output_dim=128)(total_feature)
    trans1 = TransformerBlock(num_heads=4, mlp_dim=512)(with_pos)
    trans2 = TransformerBlock(num_heads=4, mlp_dim=512)(trans1)
    last_frame = Lambda(lambda x: x[:, -1, :])(trans2)
    model = Model(inputs=input_state, outputs=last_frame)
    return model

def MFCA():
    input_state = Input(shape=(8,80,80,1))
    conv1 = TimeDistributed(Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=initializer))(input_state)
    conv2 = TimeDistributed(Conv2D(32, 4, (2,2), activation='relu', padding='same', kernel_initializer=initializer))(conv1)
    with_pos_large = TemporalEmbedding(output_dim=32)(conv1)
    with_pos_small = TemporalEmbedding(output_dim=32)(conv2)
    att_large = multiFocusConvAttention(num_heads=2, selector=2)(with_pos_large)
    att_small = multiFocusConvAttention(num_heads=2, selector=2)(with_pos_small)
    att = MaxPooling2D(2)(att_large)
    all_att = Concatenate(axis=-1)([att, att_small])
    GAP = GlobalAveragePooling2D()(all_att)
    model = Model(inputs=input_state, outputs=GAP)
    return model

def MultiscaleTransformer():
    input_state = Input(shape=(8, 80, 80, 1))
    encoded = Conv3D(64, kernel_size=[2, 7, 7], strides=[2, 4, 4], padding='same', kernel_initializer=initializer)(input_state)
    with_temp = TemporalEmbedding(output_dim=64)(encoded)
    with_ST = SpatialEmbedding(output_dim=64)(with_temp)
    _att1 = DownSampleTransformerBlock(num_heads=4, mlp_dim=256)(with_ST)
    _att2 = SpaceTimeLocalTransformerBlock(num_heads=4, kernel_size=[2, 4, 4], mlp_dim=256)(_att1)
    _att3 = MultiscaleTransformerBlock(num_heads=4, mlp_dim=512, is_pooling=True, is_expanding=False)(_att2)
    _att4 = MultiscaleTransformerBlock(num_heads=4, mlp_dim=512, is_pooling=False, is_expanding=True)(_att3)
    last_frame = Lambda(lambda x: x[:, -1, :, :, :])(_att4)
    GAP = GlobalAveragePooling2D()(last_frame)
    model = Model(inputs=input_state, outputs=GAP)
    return model

def OnlyMultiscale():
    input_state = Input(shape=(8, 80, 80, 1))
    encoded = Conv3D(64, kernel_size=[2, 7, 7], strides=[2, 4, 4], padding='same')(input_state)
    with_temp = TemporalEmbedding(output_dim=64)(encoded)
    with_ST = SpatialEmbedding(output_dim=64)(with_temp)
    _att1 = MultiscaleTransformerBlock(num_heads=4, mlp_dim=256, is_pooling=True, is_expanding=False)(with_ST)
    _att2 = MultiscaleTransformerBlock(num_heads=4, mlp_dim=256, is_pooling=False, is_expanding=True)(_att1)
    # _att1 = DownSampleTransformerBlock(num_heads=4, mlp_dim=256)(with_ST)
    # _att2 = SpaceTimeLocalTransformerBlock(num_heads=4, kernel_size=[2, 4, 4], mlp_dim=256)(_att1)
    _att3 = MultiscaleTransformerBlock(num_heads=4, mlp_dim=512, is_pooling=True, is_expanding=False)(_att2)
    _att4 = MultiscaleTransformerBlock(num_heads=4, mlp_dim=512, is_pooling=False, is_expanding=True)(_att3)
    last_frame = Lambda(lambda x: x[:, -1, :, :, :])(_att4)
    GAP = GlobalAveragePooling2D()(last_frame)
    model = Model(inputs=input_state, outputs=GAP)
    return model

def RNDmodel():
    last_frame = Lambda(lambda x: x[:, -1, :, :, :])
    conv1 = Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=initializer)(last_frame)
    conv2 = Conv2D(64, 4, (2,2), activation='relu', padding='same', kernel_initializer=initializer)(conv1)
    conv3 = Conv2D(64, 3, (1,1), activation='relu', padding='same', kernel_initializer=initializer)(conv2)
    flat = Flatten()(conv3)
    out = Dense(5, activation='linear')(flat)
    model = Model(inputs=last_frame, outputs=flat)
    return model    