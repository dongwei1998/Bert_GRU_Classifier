# coding=utf-8
# =============================================
# @Time      : 2022-03-23 17:55
# @Author    : DongWei1998
# @FileName  : flasktest.py
# @Software  : PyCharm
# =============================================
import time
import requests
import json


def pic_post():
    url = f"http://10.19.234.179:5556/api/v1/classification"
    demo_text ={
        'text':'''
7，您好抱歉打扰到，公司号码专属经理工号0嚎，请问1尾号3手机1直用，您好女士，公司回馈长期支持月额外2G2千多M全国流量，套餐改变套餐，代收费外是手机消费满20元月额外2G2千多M流量答谢优惠，放心来用好，刚才流量，噢当月看张卡消费都17才这样的话等于3元钱2G2千多M流量长期县，不用担心载有套餐外上网费，月都用到2G流量，噢，不，网，无线网络普及，，，看您们偶尔手机外面扫码各款，微信上孩子接个乐聊微信上语音聊天超出带来1M2毛9扣费1个际扣60，话费月很，消费平时都77，很少钱2G流量单独开流量包月划算，，不，网上，噢，那好女士抱歉，提醒服务到期自动取消？噢，不，好感谢您接听后期1天优惠活动提供11专属服务好，好行祝您生活愉快再见女士|，，办3月，预付款超出范围，，这样的话花流量，赠送3月天气预报短信，打扰，再见，再见 '''
    }

    headers = {
        'Content-Type': 'application/json'
    }
    start = time.time()
    result = requests.post(url=url, json=demo_text,headers=headers)
    end = time.time()
    if result.status_code == 200:
        obj = json.loads(result.text)
        print(obj)
    else:
        print(result)
    print('Running time: %s Seconds' % (end - start))


pic_post()