# -*- codeing = utf-8 -*-
# Time : 2023/7/5 13:35
# @Auther : zhouchao
# @File: qt_test.py
# @Software:PyCharm
import numpy as np
import socket
import logging
import matplotlib.pyplot as plt

def rec_socket(recv_sock: socket.socket, cmd_type: str, ack: bool) -> bool:
    if ack:
        cmd = 'A' + cmd_type
    else:
        cmd = 'D' + cmd_type
    while True:
        try:
            temp = recv_sock.recv(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return False
        except TimeoutError as e:
            logging.error(f'超时了，错误代码: \n{e}')
            return False
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return False
        if temp == b'\xaa':
            break

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        try:
            temp += recv_sock.recv(1)
        except Exception as e:
            logging.error(f'接收报文长度失败, 错误代码: \n{e}')
            return False
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}, \n报文内容\n{temp}')
        return False

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        try:
            temp += recv_sock.recv(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}，\n报文内容\n{temp}')
            return False
    data = temp
    if cmd.strip().upper() != data[:4].decode('ascii').strip().upper():
        logging.error(f'客户端接收指令错误,\n指令内容\n{data}')
        return False
    else:
        if cmd == 'DIM':
            print(data)

        # 进行数据校验
        temp = b''
        while len(temp) < 3:
            try:
                temp += recv_sock.recv(1)
            except Exception as e:
                logging.error(f'接收报文校验失败, 错误代码: \n{e}')
                return False
        if temp == b'\xff\xff\xbb':
            return True
        else:
            logging.error(f"接收了一个完美的只错了校验位的报文，\n data: {data}")
            return False



def main():
    socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_receive.bind(('127.0.0.1', 21123))
    socket_receive.listen(5)
    socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_send.bind(('127.0.0.1', 21122))
    socket_send.listen(5)
    print('等待连接')
    socket_send_1, receive_addr_1 = socket_send.accept()
    print("连接成功：", receive_addr_1)
    # socket_send_2 = socket_send_1
    socket_send_2, receive_addr_2 = socket_receive.accept()
    print("连接成功：", receive_addr_2)
    while True:
        cmd = input('请输入指令：').strip().upper()
        if cmd == 'IM':
            with open('data/newrawfile_ref.raw', 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.float32).reshape(750, 288, 384)
            data = data[:, [91, 92, 93, 94, 95, 96, 97, 98, 99, 100], :]
            n_rows, n_bands, n_cols = data.shape[0], data.shape[1], data.shape[2]
            print(f'n_rows：{n_rows}, n_bands：{n_bands}, n_cols：{n_cols}')
            n_rows, n_cols, n_bands = [x.to_bytes(2, byteorder='big') for x in [n_rows, n_cols, n_bands]]
            data = data.tobytes()
            length = len(data) + 10
            print(f'length: {length}')
            length = length.to_bytes(4, byteorder='big')
            msg = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + n_rows + n_cols + n_bands + data + b'\xff\xff\xbb'
            socket_send_1.send(msg)
            print('发送成功')
            result = socket_send_2.recv(5)
            length = int.from_bytes(result[1:5], byteorder='big')
            result = socket_send_2.recv(length)
            n_rows, n_cols = int.from_bytes(result[4:6], byteorder='big'), int.from_bytes(result[6:8], byteorder='big')
            data = np.frombuffer(result[8:], dtype=np.uint8).reshape(n_rows, n_cols)
            plt.imshow(data)
            plt.show()

if __name__ == '__main__':
    main()