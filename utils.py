from sqlite3 import Time
import tensorflow
import numpy as np
from collections import deque
from numpy.core.fromnumeric import squeeze
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras import activations
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import Sequential, Model, load_model
from tensorflow.compat.v1.keras.layers import Multiply, Conv3D, GlobalAveragePooling2D, ConvLSTM2D, Permute, Softmax, AveragePooling2D, MaxPooling2D, Convolution2D, LeakyReLU, add, Reshape, Lambda, Conv2D, LSTMCell, LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, multiply, Concatenate, Flatten, Activation, dot, Dot, Dropout
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.compat.v1.keras import losses

initializer = tf.keras.initializers.Orthogonal(gain=1.0)

class SinusoidalPositionEmbedding(tf.keras.layers.Layer):
    """定义Sin-Cos位置Embedding"""

    def __init__(
            self,
            output_dim,
            merge_mode='add',
            custom_position_ids=False,
            **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        self.num_heads = num_heads
        super(MultiHeadSelfAttention, self).__init__(*args, **kwargs)
        
    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        self.d_model = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.query_dense_array = []
        self.key_dense_array = []
        self.value_dense_array = []
        for i in range(num_heads):
            self.query_dense_array.append(Dense(self.d_model, name = f'query_{i}', kernel_initializer=initializer))
            self.key_dense_array.append(Dense(self.d_model, name=f'key_{i}', kernel_initializer=initializer))
            self.value_dense_array.append(Dense(self.d_model, name=f'value_{i}', kernel_initializer=initializer))
        self.combine_heads = Dense(hidden_size, name='out')
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        scaled_score = score / tf.math.sqrt(float(self.d_model))
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output
    def call(self, inputs):
        weighted_sums = []
        for i in range(self.num_heads):
            query = self.query_dense_array[i](inputs)
            key = self.key_dense_array[i](inputs)
            value = self.value_dense_array[i](inputs)
            weighted_sum = self.attention(query, key, value)
            weighted_sums.append(weighted_sum)
        concat = K.concatenate(weighted_sums, axis=-1)
        return self.combine_heads(concat)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, mlp_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
    
    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(num_heads=self.num_heads, name='MultiHeadSelfAttention_1')
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="relu",
                    name=f"{self.name}/Dense_0", kernel_initializer=initializer
                ),
                tf.keras.layers.Dense(input_shape[-1], activation="linear", name=f"{self.name}/Dense_1", kernel_initializer=initializer),
            ],
            name="MlpBlock_3",
        )
    
    def call(self, inputs):
        x = inputs
        x = self.att(x)
        x = x + inputs
        y = self.mlpblock(x)
        y = x + y
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(TemporalEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.pos_emb = SinusoidalPositionEmbedding(self.output_dim)
    
    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, time_steps, img_size, _, channel_depth = input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        zeros = tf.zeros([1, time_steps, self.output_dim])
        pos_vector = self.pos_emb(zeros) # [1, time_steps, output_dim]
        pos_vector = tf.expand_dims(pos_vector, axis=2)
        pos_vector = tf.expand_dims(pos_vector, axis=2)
        # [1, time_steps, 1, 1, output_dim]
        pos_3D = tf.broadcast_to(pos_vector, input_shape)
        
        return inputs + pos_3D
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class SpatialEmbedding(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(SpatialEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.pos_emb = SinusoidalPositionEmbedding(self.output_dim)
    
    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, time_steps, img_size, _, channel_depth = input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        zeros = tf.zeros([1, img_size*img_size, self.output_dim])
        pos_vector = self.pos_emb(zeros) # [1, img_size*img_size, output_dim]
        pos_vector = K.reshape(pos_vector, [1, img_size, img_size, self.output_dim])
        pos_vector = tf.expand_dims(pos_vector, axis=1)
        # [1, 1, img_size, img_size, output_dim]
        pos_3D = tf.broadcast_to(pos_vector, input_shape)
        
        return inputs + pos_3D
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class ConvSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        self.num_heads = num_heads
        super(ConvSelfAttention, self).__init__(*args, **kwargs)
    
    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        self.d_model = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.query_dense_array = []
        self.key_value_dense_array = []
        self.att_dense_array = []
        for i in range(num_heads):
            self.query_dense_array.append(TimeDistributed(Conv2D(self.d_model, 3, (1,1), padding='same', kernel_initializer=initializer)))
            self.key_value_dense_array.append(TimeDistributed(Conv2D(self.d_model, 3, (1,1), padding='same', kernel_initializer=initializer)))
            self.att_dense_array.append(TimeDistributed(Conv2D(1, 3, (1,1), padding='same', kernel_initializer=initializer)))
        
    def attention(self, inputs, index):
        # [batch, time, h, w, c] => [batch, time, time, h, w, c]
        input_shape = list(K.int_shape(inputs))
        input_shape.insert(1, input_shape[1])
#         input_shape[0] = -1
        input_shape[-1] = int(self.d_model)
        # calculate q, kv
        query = self.query_dense_array[index](inputs)
        kv = self.key_value_dense_array[index](inputs)
        # expand_dims
        expand_query = K.expand_dims(query, axis=2)
        expand_key = K.expand_dims(kv, axis=1)
        expand_value = K.expand_dims(kv, axis=1)
        # => [batch, time, time, h, w, c]
        expand_query = K.tile(expand_query, [1,1,input_shape[1],1,1,1])
        expand_key = K.tile(expand_key, [1,input_shape[1],1,1,1,1])
        expand_value = K.tile(expand_value, [1,input_shape[1],1,1,1,1])
        # [batch, time, time, h, w, c] => [batch, time*time, h, w, c]
        input_shape[1] = input_shape[1] * input_shape[2]
        del input_shape[2]
        input_shape[0] = -1
        reshaped_query = K.reshape(expand_query, input_shape)
        reshaped_key = K.reshape(expand_key, input_shape)
        # [batch, time*time, h, w, 2c]
        qk = K.concatenate([reshaped_key, reshaped_query], axis=-1)
        att_map = self.att_dense_array[index](qk)
        input_shape = list(K.int_shape(inputs))
        input_shape.insert(1, input_shape[1])
        input_shape[-1] = 1
        input_shape[0] = -1
        reshaped_att_map = K.reshape(att_map, input_shape)
        soft_reshaped_att_map = Softmax(axis=2)(reshaped_att_map)
        weighted_value_map = Multiply()([expand_value, soft_reshaped_att_map])
        output = K.sum(weighted_value_map, axis=2)
            
        return output
        
    def call(self, inputs):
        weighted_sums = []
        for i in range(self.num_heads):
            weighted_sum = self.attention(inputs, i)
            weighted_sums.append(weighted_sum)
        concat = K.concatenate(weighted_sums, axis=-1)
        return concat
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

@tf.keras.utils.register_keras_serializable()
class ConvTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
    
    def build(self, input_shape):
        self.att = ConvSelfAttention(num_heads=self.num_heads, name='MultiHeadSelfAttention_1')
        self.ffn = TimeDistributed(Conv2D(input_shape[-1], 3, (1,1), padding='same', kernel_initializer=initializer))
        # self.gn1 = GroupNormalization()
        # self.gn2 = GroupNormalization()
    
    def call(self, inputs):
        x = inputs
        x = self.att(x)
        x = x + inputs
        # x = self.gn1(x)
        y = self.ffn(x)
        y = x + y
        # y = self.gn2(y)
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
            }
        )
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class multiFocusConvAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, selector, **kwargs):
        self.num_heads = num_heads
        self.selector = selector
        super(multiFocusConvAttention, self).__init__(*args, **kwargs)
        self.selectors = []
        for i in range(self.selector):
            # 跟块大小及num_heads, input_shape[-1]有关
            self.selectors.append(self.add_weight(shape=[2, 2, 16], dtype=tf.float32))
        
    
    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        selector = self.selector
        self.d_model = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.key_dense_array = []
        self.value_dense_array = []
        self.att_dense_array = []
        self.dim_reduct = []
        for i in range(num_heads):
            self.key_dense_array.append(TimeDistributed(Conv2D(self.d_model, 2, (1,1), padding='same', kernel_initializer=initializer)))
            self.value_dense_array.append(TimeDistributed(Conv2D(self.d_model, 2, (1,1), padding='same', kernel_initializer=initializer)))
            self.att_dense_array.append(TimeDistributed(Conv2D(1, 2, (1,1), activation='relu', padding='same', kernel_initializer=initializer)))
        for i in range(selector):
            self.dim_reduct.append(Conv2D(self.d_model, 2, (1,1), activation='relu', padding='same', kernel_initializer=initializer))
        
    def return_tile_selectors(self, selector, inputs):
        width = K.int_shape(inputs)[2]
        height = K.int_shape(inputs)[3]
        tile_selector = K.tile(selector, [int(width/2)+1, int(height/2)+1, 1])
        tile_selectors = []
        tile_selectors.append(tile_selector[:width, :height, :])
        tile_selectors.append(tile_selector[:width, 1:(height+1), :])
        tile_selectors.append(tile_selector[1:(width+1), :height, :])
        tile_selectors.append(tile_selector[1:(width+1), 1:(height+1), :])
        
        return tile_selectors
        
    def attention(self, inputs, tile_selectors, index):
        batch_size = tf.shape(inputs)[0]
        timesteps = K.int_shape(inputs)[1]
        width = K.int_shape(inputs)[2]
        height = K.int_shape(inputs)[3]
        
        # [1, 1, 50, 40, 16]
        # 4 * [H, W, C]
        outputs = []
        for i in range(len(tile_selectors)):
            tile_selector = tile_selectors[i]
            tile_selector = K.expand_dims(tile_selector, axis=0)
            tile_selector = K.expand_dims(tile_selector, axis=0)
            # [batch, time_step, H, W, C]
            tile_selector = tf.broadcast_to(tile_selector, [batch_size, timesteps,
                                                           width, height, tf.shape(tile_selector)[-1]])
            # calculate k, v
            key = self.key_dense_array[index](inputs)
            value = self.value_dense_array[index](inputs)
            # concate
            select_key = K.concatenate([key, tile_selector], axis=-1)
            # calculate attention map(batch,time_step,H,W,1)
            att_map = self.att_dense_array[index](select_key)
            soft_att_map = Softmax(axis=1)(att_map)
            # point wise production
            weighted_value_map = Multiply()([value, soft_att_map])
            output = K.sum(weighted_value_map, axis=1)
            outputs.append(output)
        return K.concatenate(outputs, axis=-1)
        
    def call(self, inputs):
        weighted_sums = []
        for j in range(len(self.selectors)):
            tile_selectors = self.return_tile_selectors(self.selectors[j], inputs)
            for i in range(self.num_heads):
                weighted_sum = self.attention(inputs, tile_selectors, i)
                dim_reduct_sum = self.dim_reduct[j](weighted_sum)
                weighted_sums.append(dim_reduct_sum)
        concat = K.concatenate(weighted_sums, axis=-1)
        return concat
        
    def compute_output_shape(self, input_shape):
        return_shape = input_shape
        return_shape[-1] = input_shape[-1] * self.selector
        return input_shape
        
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"selector": self.selector})
        return config

class CreatePatches(tf.keras.layers.Layer):
    
    def __init__(self, patch_size, **kwargs):
        super(CreatePatches, self).__init__(**kwargs)
        self.patch_size = patch_size
    
    def get_config(self):  # override get_config
        config = {"patch_size": self.patch_size}
        base_config = super(CreatePatches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        patches = []
        # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
        input_image_size_i = inputs.shape[2]
        input_image_size_j = inputs.shape[3]
        for i in range(0, input_image_size_i, self.patch_size):
            for j in range(0, input_image_size_j, self.patch_size):
                a = inputs[:, :, i:i + self.patch_size, j:j + self.patch_size, :]
                a = K.expand_dims(a, axis=2)
                patches.append(a)
        patches = K.concatenate(patches, axis=2)
        return patches

class Add_Embedding_Layer(tf.keras.layers.Layer):
    def __init__(self, *args, num_patches=16, d_model=400, **kwargs):
        self.num_patches= num_patches
        self.d_model = d_model
        super(Add_Embedding_Layer, self).__init__(*args, **kwargs)
        self.patch_emb = self.add_weight(shape=[1,1,self.d_model], dtype=tf.float32)
        self.pos_emb = self.add_weight(shape=[1, self.num_patches+1, self.d_model], dtype=tf.float32)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patch_emb = tf.broadcast_to(self.patch_emb, [batch_size, 1, self.d_model])
        pos_emb = tf.broadcast_to(self.pos_emb, [batch_size, self.num_patches+1, self.d_model])
#         patch_emb = K.repeat_elements(self.patch_emb, batch_size, axis=0)
#         pos_emb = K.repeat_elements(self.pos_emb, batch_size, axis=0)
        return K.concatenate([patch_emb, inputs], axis=1) + pos_emb
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+1, input_shape[2])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "d_model": self.d_model,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class VisionTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, mlp_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
    
    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(num_heads=self.num_heads, name='MultiHeadSelfAttention_1')
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="relu",
                    name=f"{self.name}/Dense_0", kernel_initializer=initializer
                ),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1", kernel_initializer=initializer),
            ],
            name="MlpBlock_3",
        )
    
    def call(self, inputs):
        x = inputs
        x = self.att(x)
        x = x + inputs
        y = self.mlpblock(x)
        y = x + y
        return y
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config
    def compute_output_shape(self, input_shape):
        return input_shape
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class DownSample(tf.keras.layers.Layer):
    def __init__(self, *args, spatial_stride, **kwargs):
        self.spatial_stride = spatial_stride
        super(DownSample, self).__init__(*args, **kwargs)
    
    def call(self, inputs):
        patches = []
        # [batch, time, height, width, channel]
        input_shape = list(K.int_shape(inputs))
        for height_index in range(self. spatial_stride):
            for width_index in range(self.spatial_stride):
                snippets = inputs[:, :, height_index::self.spatial_stride, width_index::self.spatial_stride, :]
                snippets = K.expand_dims(snippets, axis=1)
                patches.append(snippets)
        # [batch, spatial_stride**2, time, height, width, channel]
        return K.concatenate(patches, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spatial_stride": self.spatial_stride,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class RebuiltFeature(tf.keras.layers.Layer):
    def __init__(self, *args, spatial_stride, **kwargs):
        self.spatial_stride = spatial_stride
        super(RebuiltFeature, self).__init__(*args, **kwargs)
    
    def call(self, inputs):
        # [batch, spatial_stride**2, time, height, width, channel]
        input_shape = list(K.int_shape(inputs))
        to_channel = inputs[:, 0, :, :, :, :]
        for i in range(1,input_shape[1]):
            to_channel = K.concatenate([to_channel, inputs[:, i, :, :, :, :]], axis=-1)
        # to_channel = [batch, time, height, width, channel*spatial_stride**2]
        res = []
        for i in range(input_shape[2]):
            a = tensorflow.nn.depth_to_space(to_channel[:, i, :, :, :], self.spatial_stride)
            a = K.expand_dims(a, axis=1)
            res.append(a)
        return K.concatenate(res, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spatial_stride": self.spatial_stride,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class DownSampleSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, spatial_stride, **kwargs):
        self.num_heads = num_heads
        self.spatial_stride = spatial_stride
        super(DownSampleSelfAttention, self).__init__(*args, **kwargs)
    
    def build(self, input_shape):
        self.downsample = DownSample(spatial_stride=self.spatial_stride)
        self.rebuiltfeature = RebuiltFeature(spatial_stride=self.spatial_stride)
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        self.d_model = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.query_dense_array = []
        self.key_dense_array = []
        self.value_dense_array = []
        for i in range(num_heads):
            self.query_dense_array.append(Dense(self.d_model, name = f'query_{i}', kernel_initializer=initializer))
            self.key_dense_array.append(Dense(self.d_model, name=f'key_{i}', kernel_initializer=initializer))
            self.value_dense_array.append(Dense(self.d_model, name=f'value_{i}', kernel_initializer=initializer))

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        scaled_score = score / tf.math.sqrt(float(self.d_model))
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    def one_head(self, inputs, i):
        groups = self.downsample(inputs)
        input_shape = list(K.int_shape(groups))
        groups = K.reshape(groups, [-1, input_shape[1], input_shape[2]*input_shape[3]*input_shape[4], input_shape[5]])
        res = []
        for group in range(input_shape[1]):
            group_in = groups[:, group, :, :]
            query = self.query_dense_array[i](group_in)
            key = self.key_dense_array[i](group_in)
            value = self.value_dense_array[i](group_in)
            re = self.attention(query, key, value)
            re = K.expand_dims(re, axis=1)
            res.append(re)
        res = K.concatenate(res, axis=1)
        reshaped_res = K.reshape(res, [-1, input_shape[1], input_shape[2], input_shape[3], input_shape[4], self.d_model])
        return reshaped_res
    
    def call(self, inputs):
        res = []
        for i in range(self.num_heads):
            res.append(self.one_head(inputs, i))
        res = K.concatenate(res, axis=-1)
        return self.rebuiltfeature(res)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spatial_stride": self.spatial_stride,
                "num_heads": self.num_heads,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

@tf.keras.utils.register_keras_serializable()
class DownSampleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, mlp_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
    
    def build(self, input_shape):
        self.att = DownSampleSelfAttention(num_heads=self.num_heads, spatial_stride=2)
        self.mlpblock = tf.keras.Sequential(
            [
                TimeDistributed(Conv2D(self.mlp_dim, 1, (1,1), activation='relu', kernel_initializer=initializer)),
                TimeDistributed(Conv2D(input_shape[-1], 1, (1,1), activation='relu', kernel_initializer=initializer)),
            ],
            name="MlpBlock_",
        )
    
    def call(self, inputs):
        x = inputs
        x = self.att(x)
        x = x + inputs
        y = self.mlpblock(x)
        y = x + y
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class MultiHeadPoolingAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, is_pooling, **kwargs):
        self.num_heads = num_heads
        self.is_pooling = is_pooling
        super().__init__(*args, **kwargs)
    
    def build(self, input_shape):
        channel_size = input_shape[-1]
        num_heads = self.num_heads
        self.d_model = channel_size // self.num_heads
        self.channel_size = channel_size
        if self.is_pooling:
            self.q_pooling = Conv3D(self.d_model, kernel_size=[2,3,3], strides=[1,2,2], activation='linear', padding='same', kernel_initializer=initializer)
            self.kv_pooling = Conv3D(self.d_model, kernel_size=[2,3,3], strides=[1,2,2], activation='linear', padding='same', kernel_initializer=initializer)
        else:
            self.q_pooling = Conv3D(self.d_model, kernel_size=[2,3,3], strides=[1,1,1], activation='linear', padding='same', kernel_initializer=initializer)
            self.kv_pooling = Conv3D(self.d_model, kernel_size=[2,3,3], strides=[1,2,2], activation='linear', padding='same', kernel_initializer=initializer)

        self.query_dense_array = []
        self.key_dense_array = []
        self.value_dense_array = []
        for i in range(num_heads):
            self.query_dense_array.append(Conv3D(self.d_model, kernel_size=[1,1,1], strides=[1,1,1], activation='linear', kernel_initializer=initializer))
            self.key_dense_array.append(Conv3D(self.d_model, kernel_size=[1,1,1], strides=[1,1,1], activation='linear', kernel_initializer=initializer))
            self.value_dense_array.append(Conv3D(self.d_model, kernel_size=[1,1,1], strides=[1,1,1], activation='linear', kernel_initializer=initializer))

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        scaled_score = score / tf.math.sqrt(float(self.d_model))
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    def one_head(self, inputs, i):
        input_seg = inputs[:, :, :, :, i*self.d_model: (i+1)*self.d_model]
        query = self.query_dense_array[i](input_seg)
        key = self.key_dense_array[i](input_seg)
        value = self.value_dense_array[i](input_seg)
        pooling_query = self.q_pooling(query)
        pooling_key = self.kv_pooling(key)
        pooling_value = self.kv_pooling(value)
        query_shape = list(K.int_shape(pooling_query))
        kv_shape = list(K.int_shape(pooling_key))
        reshaped_query = K.reshape(pooling_query, [-1, query_shape[1]*query_shape[2]*query_shape[3], query_shape[4]])
        reshaped_key = K.reshape(pooling_key, [-1, kv_shape[1]*kv_shape[2]*kv_shape[3], kv_shape[4]])
        reshaped_value = K.reshape(pooling_value, [-1, kv_shape[1]*kv_shape[2]*kv_shape[3], kv_shape[4]])
        att = self.attention(reshaped_query, reshaped_key, reshaped_value)

        resi = self.q_pooling(input_seg)
        resahped_att = K.reshape(att, [-1]+query_shape[1:])

        return resi + resahped_att
    
    def call(self, inputs):
        res = []
        for i in range(self.num_heads):
            res.append(self.one_head(inputs, i))
        res = K.concatenate(res, axis=-1)
        return res
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "is_pooling": self.is_pooling,
                "num_heads": self.num_heads,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

@tf.keras.utils.register_keras_serializable()
class MultiscaleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, mlp_dim, is_pooling, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.is_pooling = is_pooling
    
    def build(self, input_shape):
        self.att = MultiHeadPoolingAttention(num_heads=self.num_heads, is_pooling=self.is_pooling)
        if not self.is_pooling:
            self.mlpblock = tf.keras.Sequential(
                [
                    Conv3D(self.mlp_dim, kernel_size=[1,1,1], strides=[1,1,1], activation='relu', kernel_initializer=initializer),
                    Conv3D(input_shape[-1], kernel_size=[1,1,1], strides=[1,1,1], activation='relu', kernel_initializer=initializer),
                ],
                name="MlpBlock_",
            )
        else:
            self.mlpblock = tf.keras.Sequential(
                [
                    Conv3D(self.mlp_dim, kernel_size=[1,1,1], strides=[1,1,1], activation='relu', kernel_initializer=initializer),
                    Conv3D(input_shape[-1]*2, kernel_size=[1,1,1], strides=[1,1,1], activation='relu', kernel_initializer=initializer),
                ],
                name="MlpBlock_",
            )
            self.linear = Conv3D(input_shape[-1]*2, kernel_size=[1,1,1], strides=[1,1,1], activation='relu', kernel_initializer=initializer)
    
    def call(self, inputs):
        x = inputs
        x = self.att(x)
        y = self.mlpblock(x)
        if self.is_pooling:
            x = self.linear(x)
        y = x + y
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "is_pooling": self.is_pooling,
            }
        )
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class SpaceTimeLocalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, kernel_size, **kwargs):
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        super(SpaceTimeLocalSelfAttention, self).__init__(*args, **kwargs)
    
    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        self.d_model = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.query_dense_array = []
        self.key_dense_array = []
        self.value_dense_array = []
        for i in range(num_heads):
            self.query_dense_array.append(TimeDistributed(Dense(self.d_model, name = f'query_{i}', kernel_initializer=initializer)))
            self.key_dense_array.append(TimeDistributed(Dense(self.d_model, name=f'key_{i}', kernel_initializer=initializer)))
            self.value_dense_array.append(TimeDistributed(Dense(self.d_model, name=f'value_{i}', kernel_initializer=initializer)))

    def kernel_attention(self, inputs):
        # [batch, timeseg, heightseg, widseg, channel]
        input_shape = list(K.int_shape(inputs))
        reshaped_input = K.reshape(inputs, [-1, input_shape[1]*input_shape[2]*input_shape[3], input_shape[4]])
        out = []
        for head in range(self.num_heads):
            query = self.query_dense_array[head](reshaped_input)
            key = self.key_dense_array[head](reshaped_input)
            value = self.value_dense_array[head](reshaped_input)
            score = tf.matmul(query, key, transpose_b=True)
            scaled_score = score / tf.math.sqrt(float(self.d_model))
            weights = tf.nn.softmax(scaled_score, axis=-1)
            output = tf.matmul(weights, value)
            out.append(output)
        concat_out = K.concatenate(out, axis=-1)
        reshaped_out = K.reshape(concat_out, [-1]+input_shape[1:])
        return reshaped_out + inputs
    
    def layer_attention(self, inputs):
        # [batch, timeseg, height, width, channel]
        input_shape = list(K.int_shape(inputs))
        # when height == width and kernel[1] == kernel[2]
        dimension_step = input_shape[2] // self.kernel_size[1]
        total = []
        for i in range(dimension_step):
            hang = []
            for j in range(dimension_step):
                # [batch, timeseg, heightseg, widseg, channel]
                small_kernel = inputs[:, :, i*self.kernel_size[1] : (i+1)*self.kernel_size[1], j*self.kernel_size[1]: (j+1)*self.kernel_size[1], ]
                kernel_att = self.kernel_attention(small_kernel)
                hang.append(kernel_att)
            # [batch, timeseg, heightseg, width, channel]
            total.append(K.concatenate(hang, axis=3))
        return K.concatenate(total, axis=2)
    
    def recursive_call(self, inputs, temporal_length=None):
        if temporal_length == None:
            input_shape = list(K.int_shape(inputs))
            temporal_length = input_shape[1]
        if temporal_length > self.kernel_size[0]:
            pre = Lambda(lambda x: x[:, :-1, :, :, :])(inputs)
            after = Lambda(lambda x: x[:, -1:, :, :, :])(inputs)
            att = self.recursive_call(pre, temporal_length-1)
            pre_length = self.kernel_size[0] - 1
            #unused
            unused = Lambda(lambda x: x[:, :-(pre_length), :, :, :])(att)
            # used
            used = Lambda(lambda x: x[:, -(pre_length):, :, :, :])(att)
            combined = K.concatenate([used, after], axis=1)
            layer_att = self.layer_attention(combined)
            return K.concatenate([unused, layer_att], axis=1)
        elif temporal_length == self.kernel_size[0]:
            return self.layer_attention(inputs)
        

    def call(self, inputs):
        # [batch, time, height, width, channel]
        return self.recursive_call(inputs)

    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "num_heads": self.num_heads,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

@tf.keras.utils.register_keras_serializable()
class SpaceTimeLocalTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, kernel_size, mlp_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.att = SpaceTimeLocalSelfAttention(num_heads=self.num_heads, kernel_size=self.kernel_size)
        self.mlpblock = tf.keras.Sequential(
            [
                Conv3D(self.mlp_dim, kernel_size=[1,1,1], strides=[1,1,1], activation='relu', kernel_initializer=initializer),
                Conv3D(input_shape[-1], kernel_size=[1,1,1], strides=[1,1,1], activation='relu', kernel_initializer=initializer),
            ],
            name="MlpBlock_",
        )
    
    def call(self, inputs):
        x = inputs
        x = self.att(x)
        y = self.mlpblock(x)
        y = x + y
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "kernel_size": self.kernel_size,
            }
        )
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# PPO
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RewardScaling:
    def __init__(self, shape=1, gamma=0.99):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class NoisyDense(tf.keras.layers.Layer):
    """ Factorized Gaussian Noisy Dense Layer
    """
    def __init__(self, units, activation=None, trainable=True):
        super(NoisyDense, self).__init__()
        self.units = units
        self.trainable = trainable
        self.activation = tf.keras.activations.get(activation)
        self.sigma_0 = 0.5

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "trainable": self.trainable,
                "activation": self.activation,
                "sigma_0": self.sigma_0
            }
        )
        return config

    def build(self, input_shape):

        p = float(input_shape[-1])
        
        self.w_mu = self.add_weight(
            name="w_mu",
            shape=(int(input_shape[-1]), self.units),
            initializer=tf.keras.initializers.RandomUniform(
                -1. / np.sqrt(p), 1. / np.sqrt(p)),
            trainable=self.trainable)

        self.w_sigma = self.add_weight(
            name="w_sigma",
            shape=(int(input_shape[-1]), self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_0 / np.sqrt(p)),
            trainable=self.trainable)

        self.b_mu = self.add_weight(
            name="b_mu",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(
                -1.0 / np.sqrt(p), 1.0 / np.sqrt(p)),
            trainable=self.trainable)

        self.b_sigma = self.add_weight(
            name="b_sigma",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma_0 / np.sqrt(p)),
            trainable=self.trainable)

    def call(self, inputs, noise=True):
        
        epsilon_in = self.f(
            tf.random.normal(shape=(self.w_mu.shape[0], 1), dtype=tf.float32))

        epsilon_out = self.f(
            tf.random.normal(shape=(1, self.w_mu.shape[1]), dtype=tf.float32))

        w_epsilon = tf.matmul(epsilon_in, epsilon_out)
        b_epsilon = epsilon_out

        w = self.w_mu + self.w_sigma * w_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon

        out = tf.matmul(inputs, w) + b

        if self.activation is not None:
            out = self.activation(out)

        return out
    
    @staticmethod
    def f(x):
        x = tf.sign(x) * tf.sqrt(tf.abs(x))
        return x