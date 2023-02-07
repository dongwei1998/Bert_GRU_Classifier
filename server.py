# coding=utf-8
# =============================================
# @Time      : 2022-04-08 14:52
# @Author    : DongWei1998
# @FileName  : server.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utils import parameter,data_help
import tensorflow as tf
from utils import network,tokenization
from flask import Flask, jsonify, request
import json
import numpy as np


# 自适应学习率
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_size, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embedding_size = tf.cast(embedding_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)

class Predictor(object):
    def __init__(self, args):
        self.args = args

        # 加载class_2_id
        with open(args.label_2_id_dir, 'r', encoding='utf-8') as r:
            self.label_2_id = json.loads(r.read())
            self.id_2_label = {v:k for k,v in self.label_2_id.items()}

        # 参数更新
        with open(args.vocab_file, 'r', encoding='utf-8') as r:
            input_vocab_size = r.readlines()
        args.input_vocab_size = len(input_vocab_size)
        args.num_calss = len(self.label_2_id)

        # 加载模型类
        self.transformer = network.Transformer(
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

        # 模型恢复
        ckpt = tf.train.Checkpoint(transformer=self.transformer)
        ckpt.restore(tf.train.latest_checkpoint(args.output_dir))

    # look-ahead mask 用于对未预测的token进行掩码
    # 这意味着要预测第三个单词，只会使用第一个和第二个单词。 要预测第四个单词，仅使用第一个，第二个和第三个单词，依此类推。
    def create_look_ahead_mark(self,size):
        # 1 - 对角线和取下三角的全部对角线（-1->全部）
        # 这样就可以构造出每个时刻未预测token的掩码
        mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mark  # (seq_len, seq_len)

    # 构建掩码
    def create_mask(self,inputs):
        # 获取为0的padding项
        seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        # 扩充维度以便用于attention矩阵
        encode_padding_mask = seq[:, np.newaxis, np.newaxis, :]  # (batch_size,1,1,seq_len)
        return encode_padding_mask


    def predict_(self,queries):


        # 数据格式化
        inputs = list(queries.strip().replace(' ', '').replace('\n', '').replace('\t', ''))

        # 序列截断
        if len(inputs) >= self.args.max_seq_length - 1:
            inputs = inputs[0:(self.args.max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志

        ntokens = []
        ntokens.append("[CLS]")  # 句子开始设置CLS 标志
        for i, token in enumerate(inputs):
            ntokens.append(token)

        ntokens.append("[SEP]")
        length = len(ntokens)
        tokenizer = tokenization.FullTokenizer(vocab_file=self.args.vocab_file)
        text = tokenizer.convert_tokens_to_ids(ntokens)
        while len(text) < self.args.max_seq_length:
            text.append(0)
        data = tf.expand_dims(text, axis=0)

        encode_padding_mask = self.create_mask(data)

        predictions,scores = self.transformer(data, True, encode_padding_mask)
        predictions_label = tf.argmax(scores, axis=1, name='predictions')
        pre_label = predictions_label.numpy().tolist()[0]



        return {self.id_2_label[pre_label]:scores.numpy().tolist()[0][pre_label]}





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
    model = 'server'
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    args = parameter.parser_opt(model)
    detector = Predictor(args)
    args.logger.info("基于bert_bigru模型：文本分类模型.....")
    @app.route('/api/v1/classification', methods=['POST'])
    def predict():

        try:
            # 参数获取
            infos = request.get_json()
            data_dict = {
                'text':''
            }
            for k, v in infos.items():
                data_dict[k] = v

            queries = data_dict['text']
            # 参数检查
            if queries is None or queries == '':
                return jsonify({
                    'code': 500,
                    'msg': '请给定参数text！！！'
                })
            # 直接调用预测的API
            predictions_dic = detector.predict_(queries)
            return jsonify({
                'code': 200,
                'msg': '成功',
                'data': str(predictions_dic)
            })
        except Exception as e:
            args.logger.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 500,
                'msg': '预测数据失败!!!'
            })
    # 启动
    app.run(host='0.0.0.0',port=5556)

