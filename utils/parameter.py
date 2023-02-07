# coding=utf-8
# =============================================
# @Time      : 2022-04-06 17:10
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config
import shutil



# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    args.mode = os.environ.get("mode")
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    # 清除模型以及可视化文件
    if model == 'train':
        args.network_name = os.environ.get('network_name')
        args.data_dir = os.environ.get('train_data_dir')
        args.output_dir = os.environ.get('output_dir')
        args.tensorboard_dir = os.environ.get('tensorboard_dir')
        args.model_ckpt_name = os.environ.get('model_ckpt_name')
        args.vocab_file = os.environ.get('vocab_file')
        args.label_2_id_dir = os.path.join(args.output_dir, 'label_2_id.json')
        args.gru_size = int(os.environ.get('gru_size'))
        args.num_layers = int(os.environ.get('num_layers'))
        args.embedding_size = int(os.environ.get('embedding_size'))
        args.num_heads = int(os.environ.get('num_heads'))
        args.feed_input_size = int(os.environ.get('feed_input_size'))
        args.input_vocab_size = int(os.environ.get('input_vocab_size'))
        args.num_calss = int(os.environ.get('num_calss'))
        args.max_seq_length = int(os.environ.get('max_seq_length'))
        args.dropout_rate = float(os.environ.get('dropout_rate'))
        args.ckpt_model_num = int(os.environ.get('ckpt_model_num'))
        args.steps_per_checkpoint = int(os.environ.get('steps_per_checkpoint'))
        # for path in [args.output_dir,args.tensorboard_dir,args.data_dir]:
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        args.batch_size = int(os.environ.get('batch_size'))
        args.num_epochs = int(os.environ.get('num_epochs'))
    elif model =='env':
        pass
    elif model == 'server':
        args.network_name = os.environ.get('network_name')
        args.data_dir = os.environ.get('train_data_dir')
        args.output_dir = os.environ.get('output_dir')
        args.tensorboard_dir = os.environ.get('tensorboard_dir')
        args.model_ckpt_name = os.environ.get('model_ckpt_name')
        args.vocab_file = os.environ.get('vocab_file')
        args.label_2_id_dir = os.path.join(args.output_dir, 'label_2_id.json')
        args.gru_size = int(os.environ.get('gru_size'))
        args.num_layers = int(os.environ.get('num_layers'))
        args.embedding_size = int(os.environ.get('embedding_size'))
        args.num_heads = int(os.environ.get('num_heads'))
        args.feed_input_size = int(os.environ.get('feed_input_size'))
        args.input_vocab_size = int(os.environ.get('input_vocab_size'))
        args.num_calss = int(os.environ.get('num_calss'))
        args.max_seq_length = int(os.environ.get('max_seq_length'))
        args.dropout_rate = float(os.environ.get('dropout_rate'))
        args.ckpt_model_num = int(os.environ.get('ckpt_model_num'))
        args.steps_per_checkpoint = int(os.environ.get('steps_per_checkpoint'))
        args.batch_size = int(os.environ.get('batch_size'))
        args.num_epochs = int(os.environ.get('num_epochs'))
    else:
        raise print('请给定model参数，可选【traian env test】')
    return args


if __name__ == '__main__':
    args = parser_opt('train')