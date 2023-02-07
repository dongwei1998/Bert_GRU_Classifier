# coding=utf-8
# =============================================
# @Time      : 2022-04-07 11:02
# @Author    : DongWei1998
# @FileName  : network.py
# @Software  : PyCharm
# =============================================
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import GRU, Bidirectional
from tqdm import tqdm
import time
import numpy as np
from utils import parameter

# 注意力计算
def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v) # （.., seq_len_v, depth）
    return output, attention_weights

# 归一化
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

# 前馈网络
def point_wise_feed_forward_network(embedding_size, feed_input_size):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(feed_input_size, activation='relu'),
        tf.keras.layers.Dense(embedding_size)
    ])

# 构造mutil head attention层
class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size

        # embedding_size 必须可以正确分为各个头
        assert embedding_size % num_heads == 0
        # 分头后的维度
        self.depth = embedding_size // num_heads

        self.wq = tf.keras.layers.Dense(embedding_size)
        self.wk = tf.keras.layers.Dense(embedding_size)
        self.wv = tf.keras.layers.Dense(embedding_size)

        self.dense = tf.keras.layers.Dense(embedding_size)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, embedding_size)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.embedding_size))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights

# Encoder层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, feed_input_size, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.mha = MutilHeadAttention(embedding_size, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_size, feed_input_size)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        # 多头注意力网络
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, embedding_size)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embedding_size)
        return out2

# 位置编码
def get_angles(pos, i, embedding_size):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(embedding_size))
    return pos * angle_rates

def positional_encoding(max_seq_length, embedding_size):
    angle_rads = get_angles(np.arange(max_seq_length)[:, np.newaxis],
                           np.arange(embedding_size)[np.newaxis,:],
                           embedding_size)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# 编码器
class Encoder(layers.Layer):
    def __init__(self, num_layers, embedding_size, num_heads, feed_input_size,input_vocab_size, max_seq_length, dropout_rate):

        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.embedding = layers.Embedding(input_vocab_size, embedding_size)
        self.pos_embedding = positional_encoding(max_seq_length, embedding_size)

        self.encode_layer = [EncoderLayer(embedding_size, num_heads, feed_input_size, dropout_rate) for _ in range(num_layers)]

        self.dropout = layers.Dropout(dropout_rate)


    def call(self, inputs, training, mark):
        # 构建embedding
        seq_len = inputs.shape[1]
        word_emb = self.embedding(inputs)
        word_emb *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        emb = word_emb + self.pos_embedding[:,:seq_len,:]
        x = self.dropout(emb, training=training)
        # 构建多层的Encoder
        for i in range(self.num_layers):
            x = self.encode_layer[i](x, training, mark)

        return x

class GruLayer(layers.Layer):
    def __init__(self, gru_size, drop_rate):
        super().__init__()
        # 前向
        fwd_GRU = GRU(gru_size, return_sequences=True, go_backwards=False, dropout=drop_rate, name="fwd_gru")
        # 后向
        bwd_GRU = GRU(gru_size, return_sequences=True, go_backwards=True, dropout=drop_rate, name="bwd_gru")
        self.bigru = Bidirectional(merge_mode="concat", layer=fwd_GRU, backward_layer=bwd_GRU, name="bigru")

    def call(self, inputs, training):

        outputs = self.bigru(inputs, training=training)

        return outputs


# 模型构建
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embedding_size, num_heads, feed_input_size,
                input_vocab_size, num_calss,
                max_seq_length, dropout_rate,gru_size):
        '''
        :param num_layers:          总的Encoder的层数
        :param embedding_size:      embedding输出的维度
        :param num_heads:           自注意力的头的个数
        :param feed_input_size:     前向网络的输入
        :param input_vocab_size:    输入词表的大小
        :param num_calss:           分类的label的数量
        :param max_seq_length:      每个批次最大长度 bert最大512
        :param dropout_rate:        丢弃概率
        '''
        super(Transformer, self).__init__()



        self.encoder = Encoder(num_layers, embedding_size, num_heads,feed_input_size,input_vocab_size, max_seq_length, dropout_rate)

        # 构建BI-LSTM
        self.gru_layer = GruLayer(gru_size,dropout_rate)
        # 丢弃
        self.dropout_layer = layers.Dropout(rate=dropout_rate)
        # 拉平
        self.fla = layers.Flatten()
        # 全连接
        self.final_layer = layers.Dense(num_calss)

        self.softmax_layer = layers.Dense(num_calss, activation="softmax",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="scores")

    def call(self, inputs,training, encode_padding_mask):
        '''
        :param inputs:                  输入的数据
        :param training:                唯一标识 train
        :param encode_padding_mask:     数据掩码
        :return:                        最终返回的 【-1,class_num】
        '''
        encode_out = self.encoder(inputs, training, encode_padding_mask)
        # todo 可以加一个lstm
        rnn_output = self.gru_layer(encode_out, training=training)
        # 全连接 不做激活 输出的是logits
        # 拉平
        fla_output = self.fla(rnn_output)
        # 丢弃
        h_drop = self.dropout_layer(fla_output)
        # 全连接
        predictions = self.final_layer(h_drop)

        scores = self.softmax_layer(predictions)
        return predictions,scores








if __name__ == '__main__':
    args = parameter.parser_opt('train')
    a=0
    bert_model = Transformer(
        num_layers=args.num_layers,
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        feed_input_size=args.feed_input_size,
        input_vocab_size=args.input_vocab_size,
        num_calss=args.num_calss,
        max_seq_length=args.max_seq_length,
        dropout_rate=args.dropout_rate,
        gru_size=args.gru_size
    )
    while a < 5000:
        inputs = tf.keras.layers.Input(shape=(None,args.max_seq_length), dtype=tf.int32)
        training = True
        encode_padding_mask = tf.keras.layers.Input(shape=(None,None,None), dtype=tf.float32)
        a+=1
