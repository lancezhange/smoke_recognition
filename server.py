#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-24 23:15:52
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-12-08 10:42:14


import sys
import socket  # socket模块


BUF_SIZE = 1024  # 设置缓冲区大小
server_addr = ('127.0.0.1', 8888)  # IP和端口构成表示地址
try:
    server = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM)  # 生成一个新的socket对象
except socket.error, msg:
    print "Creating Socket Failure. Error Code : " + str(msg[0])
    + " Message : " + msg[1]
    sys.exit()

print "Socket Created!"
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置地址复用
try:
    server.bind(server_addr)  # 绑定地址
except socket.error, msg:
    print "Binding Failure. Error Code : " + str(msg[0])
    + " Message : " + msg[1]
    sys.exit()
print "Socket Bind!"
server.listen(5)  # 监听, 最大监听数为5
print "Socket listening"
while True:
    client, client_addr = server.accept()  # 接收TCP连接, 并返回新的套接字和地址, 阻塞函数
    print 'Connected by', client_addr
    while True:
        data = client.recv(BUF_SIZE)  # 从客户端接收数据
        print data
        client.sendall(data)  # 发送数据到客户端
server.close()
