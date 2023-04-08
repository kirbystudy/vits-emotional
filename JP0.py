from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
from random import randint
import uuid
import commons
import sys
import os
import json
import re
from torch import no_grad, LongTensor, FloatTensor
import logging
import argparse
import time
import gc
from winsound import PlaySound
from threading import Thread

from urllib3 import encode_multipart_formdata

import requests

####################################
# CHATGPT INITIALIZE
import base64
import json

import socket
import threading
from concurrent.futures import ThreadPoolExecutor

logging.getLogger('numba').setLevel(logging.WARNING)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

bind_ip = "127.0.0.1"
bind_port = 9999

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
server.listen(8)


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


# def tts(txt,  arousal, dominance, valence):
#     stn_tst = get_text(txt, hps)
#     with torch.no_grad():
#         x_tst = stn_tst.unsqueeze(0)
#         x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
#         sid = torch.LongTensor([0])
#         emo = torch.FloatTensor([[ arousal, dominance, valence]])
#
#         audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1, emo=emo)[0][0,0].data.float().numpy()
#     #ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

def generateSound(inputString, fname, model, config, speaker_id, arousal, dominance, valence):
    # arousal 表示情感的激烈程度 0 ~ 1

    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

    # model = input('Path of a VITS model: ')
    # config = input('Path of a config file: ')

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    # emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False
    
    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotionType="logits",
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    if n_symbols != 0:
        text = inputString
        length_scale, text = get_label_value(
            text, 'LENGTH', 1, 'length scale')
        noise_scale, text = get_label_value(
            text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = get_label_value(
            text, 'NOISEW', 0.8, 'deviation of noise')
        stn_tst = get_text(text, hps_ms)
        # print_speakers(speakers, escape)
        # speaker_id = get_speaker_id('Speaker ID: ')
        # out_path = input('Path to save: ')
        out_path = "E:/vscode/vits-client/output/" + fname + ".wav"
        # stn_tst = get_text(txt, hps)
        
        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)])
            sid = LongTensor([speaker_id])
            emo = FloatTensor([[arousal, dominance, valence]])
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                   noise_scale_w=noise_scale_w, length_scale=length_scale, emo=emo)[0][
                0, 0].data.cpu().float().numpy()
        write(out_path, hps_ms.data.sampling_rate, audio)
        print('Successfully emo saved!')



# 这是客户处理线程，也是一个回调函数，创建一个新的进程对象，将客户端套接字对象作为一个句柄传递给它。
def handle_client(client_socket):
    try:
        # 不能确定正确执行的代码
        # 打印处客户端发送得到的内容
        request = client_socket.recv(10240)

        print("[*] Received: %s " % request)

        # 返还一个 ACK 在计算机网络中的意思是返回包确认，表示收到
        # client_socket.send("ACK!")
        # client_socket.close()

        stringdata = request.decode('ISO-8859-2')
        print(stringdata)
        # print(stringdata.split('~')[1])
        result = stringdata.split('#')[1]

        random_id = '-'.join(''.join('{:x}'.format(randint(0, 15)) for _ in range(y)) for y in [8, 4, 4, 4, 8])
        print(random_id)
        info_dict = json.loads(result)
        model = info_dict['model']
        config = info_dict['config']
        idx = int(info_dict['index'])
        text = info_dict['text']
        arousal = float(info_dict['arousal'])
        dominance = float(info_dict['dominance'])
        valence = float(info_dict['valence'])
        eCode = str(text).encode()
        print(eCode)
        dCode = str(base64.b64decode(eCode), 'utf-8')
        print(dCode)

        generateSound(dCode, random_id, model, config, idx, arousal, dominance, valence)
    except Exception as e:
        print(e)

    client_socket.send(str(random_id).encode())
    client_socket.close()
    # gc.collect()


pool = ThreadPoolExecutor(max_workers=4)

if __name__ == '__main__':
    while True:
        gc.collect()
        client, addr = server.accept()

        print("[*] Accepted connection from: %s:%d" % (addr[0], addr[1]))
        pool.submit(handle_client, client)
        queue = pool._work_queue  # 获取任务队列
        print(queue.qsize())
        if (queue.qsize() >= 12):
            print("测试")
            sys.exit(1)

