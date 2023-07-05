# -*- codeing = utf-8 -*-
# Time : 2023/7/5 13:14
# @Auther : zhouchao
# @File: main.py
# @Software:PyCharm
import socket
import time

import numpy as np
import logging
import os
import sys


from classfier import Astragalin
from utils import DualSock, try_connect, receive_sock, parse_protocol, ack_sock, done_sock, process_cmd
from root_dir import ROOT_DIR


def main(is_debug=False):
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'report.log'))
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler], level=logging.DEBUG)
    dual_sock = DualSock(connect_ip='127.0.0.1')

    while not dual_sock.statue:
        dual_sock.reconnect()
    detector = Astragalin(load_from=ROOT_DIR / 'models' / 'astragalin.p')
    # _ = detector.predict(np.ones((4096, 1200, 10), dtype=np.float32))
    while True:
        pack, next_pack = receive_sock(dual_sock)
        if pack == b"":  # 无数据
            time.sleep(5)
            dual_sock.reconnect()
            continue

        cmd, data = parse_protocol(pack)
        process_cmd(cmd=cmd, data=data, connected_sock=dual_sock, detector=detector)


if __name__ == '__main__':
    main()