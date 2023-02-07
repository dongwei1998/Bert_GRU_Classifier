# coding=utf-8
# =============================================
# @Time      : 2022-03-25 14:17
# @Author    : DongWei1998
# @FileName  : gpu_git.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def check_gpus(mode,logger):
    '''
     GPU available check
     reference : http://feisky.xyz/machine-learning/tensorflow/gpu_list.html
     '''
    local_device_protos = tf.config.get_visible_devices()
    all_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if not all_gpus:
        logger.info('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        logger.info('Using CPU 0:/cpu:0')
        return tf.device('/cpu:0')
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        logger.info("'nvidia-smi' tool not found.")
        logger.info('Using CPU 0:/cpu:0')
        return tf.device('/cpu:0')
    else:
        qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
        device_ = GPUManager(qargs,logger)
        return device_.auto_choice(mode)

def parse(line, qargs):
    '''
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''


    numberic_args = ['memory.free', 'memory.total', 'power.draw','power.limit']  # 可计数的参数
    power_manage_enable = lambda v: (not 'Not Support' in v)  # lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', '').replace('[N/A]', '1.0'))  # 带单位字符串去掉单位
    process = lambda k, v: ((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


def query_gpu(qargs=[]):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''

    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()

    return [parse(line, qargs) for line in results]


def by_power(d):
    '''

   helper function fo sorting gpus by power
-
   '''


    power_infos = (d['power.draw'], d['power.limit'])
    if any(v == 1 for v in power_infos):

        print('Power management unable for GPU {}'.format(d['index']))
        return 1
    return float(d['power.draw']) / d['power.limit']


class GPUManager(object):
    '''

    qargs:

        query arguments

    A manager which can list all available GPU devices and sort them and choice the most free one.Unspecified ones pref.

    GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，优先选择未指定的GPU。

    '''

    def __init__(self, qargs,logger):
        '''

        '''
        self.qargs = qargs
        self.logger = logger
        self.gpus = query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified'] = False
        self.gpu_num = len(self.gpus)

    def _sort_by_memory(self, gpus, by_size=False):
        if by_size:
            self.logger.info('Sorted by free memory size')
            return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
        else:
            self.logger.info('Sorted by free memory rate')
            return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)

    def _sort_by_power(self, gpus):
        return sorted(gpus, key=by_power)

    def _sort_by_custom(self, gpus, key, reverse=False, qargs=[]):
        if isinstance(key, str) and (key in qargs):
            return sorted(gpus, key=lambda d: d[key], reverse=reverse)
        if isinstance(key, type(lambda a: a)):
            return sorted(gpus, key=key, reverse=reverse)
        raise ValueError(
            "The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

    def auto_choice(self, mode):
        '''

        mode:

            0:(default)sorted by free memory size

        return:

            a TF device object

        Auto choice the freest GPU device,not specified

        ones

        自动选择最空闲GPU

        '''
        for old_infos, new_infos in zip(self.gpus, query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

        if mode == 0:
            self.logger.info('Using CPU 0:/cpu:0')
            return tf.device('/cpu:0')
        elif mode == 1:
            self.logger.info('Choosing the GPU device has largest free memory...')
            chosen_gpu = self._sort_by_memory(unspecified_gpus, True)[0]
        elif mode == 2:
            self.logger.info('Choosing the GPU device has highest free memory rate...')
            chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
        elif mode == 3:
            self.logger.info('Choosing the GPU device by power...')
            chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
        else:
            self.logger.info('Given an unaviliable mode,will be chosen by memory')
            chosen_gpu = self._sort_by_memory(unspecified_gpus)[0]
        chosen_gpu['specified'] = True
        index = chosen_gpu['index']
        self.logger.info('Using GPU {i}:{info}'.format(i=index,info=' | '.join([str(k) + ':' + str(v) for k, v in chosen_gpu.items()])))
        os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:{}'.format(index)

        return '/gpu:{}'.format(index)


