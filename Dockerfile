FROM dongwei2021/centos_py37_tf25_gpu


# time zone set
WORKDIR /usr/share
ADD ./zoneinfo ./zoneinfo
RUN  ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "Asia/Shanghai" > /etc/timezone

# 创建目录
RUN mkdir /ntt
RUN mkdir /ntt/ckpt_model
RUN mkdir /ntt/tensorboard
RUN mkdir /ntt/datasets

# 复制文件
WORKDIR /opt
ADD ./config ./config
ADD ./log ./log
ADD ./utils ./utils
ADD .env .
ADD flasktest.py .
ADD flasktest.txt .
ADD release.sh .
ADD server.py .
ADD server.sh .
ADD train.py .
ADD train.sh .




