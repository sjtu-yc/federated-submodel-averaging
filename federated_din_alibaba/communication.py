# -*- coding: UTF-8 -*-

import sys
import socket
import time
import ssl
import hmac
import tensorflow as tf
import numpy as np
from config import SEND_RECEIVE_CONF as SRC # used for the shared global configuration

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Communication(object):

    def __init__(self, is_ps, private_ip, public_ip):
        """Constructs a Federated Communication object
            Args:
              private_ip (str): complete local ip in which the chief is going to
                    serve its socket. Example: 172.134.65.123:7777
              public_ip (str): ip to which the workers are going to connect.
         """
        self._is_ps = is_ps # mark if the current host is a PS
        self._private_ip = private_ip.split(':')[0]         # the private(local) IP address
        self._private_port = int(private_ip.split(':')[1])  # the private(local) IP port
        self._public_ip = public_ip.split(':')[0]           # the public(Internet) IP address
        self._public_port = int(public_ip.split(':')[1])    # the public(Internet) IP port
        if self._is_ps:
            self.ps_socket = self.start_socket_ps()

    def start_socket_ps(self):
        """Creates a socket that will act as server.
            Returns:
              sever_socket (socket): ssl secured socket that will act as server.
         """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # create a socket
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # set the socket to reuse the address for different connections
        server_socket.bind((self._private_ip, self._private_port))          # bind private IP to listen requests
        server_socket.listen(1000)                                          # 1000 is the maximum number of connections at the same time
        return server_socket

    def start_socket_client(self):
        """Creates a socket that will act as client.
           Returns:
              sever_socket (socket): ssl secured socket that will work as client.
         """
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # create a socket
        client_socket.connect((self._public_ip, self._public_port))         # connect to the server's public IP
        return client_socket

    def receiving_subroutine(self, connection_socket):
        """Subroutine inside get_np_array to recieve a list of numpy arrays.
        If the sending was not correctly recieved, it sends back an error message
        to the sender in order to try it again.
        Args:
          connection_socket (socket): a socket with a connection already
              established.
         """
        timeout = 1.0
        while True:
            ultimate_buffer = b''   # buffer for saving data
            connection_socket.settimeout(240)   # set timeout for the first round receiving operation
            first_round = True
            while True:
                try:
                    receiving_buffer = connection_socket.recv(SRC.buffer)   # try receiving data of SRC.buffer size and save it to receiving_buffer
                except Exception as e:
                    if str(e) != 'timed out':   # if the exception is not timed out, then error may have occured
                        print(e)
                        sys.stdout.flush()
                    break
                if first_round: # if we have just finished the first round with 240s timeout, we set a short timeout for the second round to check if all data has been received.
                    connection_socket.settimeout(timeout)
                    first_round = False
                if not receiving_buffer:    # if we have not received anything, then all data has been transferred
                    break
                ultimate_buffer += receiving_buffer # append the received data in this round to the ultimate_buffer

            pos_signature = SRC.hashsize    # set the end position of the hash signature
            signature = ultimate_buffer[:pos_signature] # this signature is used to check the data integrity
            message = ultimate_buffer[pos_signature:int(0-SRC.eofsize)] # this is the actual massage
            good_signature = hmac.new(SRC.key, message, SRC.hashfunction).digest()  # calculate the hash of the received message

            if signature != good_signature: # compare good_signature with the received signature to check the data integrity
                print("Received wrong message!")
                connection_socket.send(SRC.error)   # SRC.error signal causes data retransfer
                timeout += 0.5  # allow for a longer timeout
                continue
            else:
                connection_socket.send(SRC.recv)    # SRC.recv signal confirms that the message has been received successfully
                return message

    def get_np_array(self, connection_socket):
        """[Deprecated!] Routine to recieve a list of numpy arrays.
            Args:
              connection_socket (socket): a socket with a connection already
                  established.
         """
        message = self.receiving_subroutine(connection_socket)
        final_message = pickle.loads(message)
        return final_message

    def send_np_array(self, arrays_to_send, connection_socket):
        """[Deprecated!] Routine to send a list of numpy arrays. It sends it as many time as necessary
            Args:
              connection_socket (socket): a socket with a connection already
                  established.
         """
        serialized = pickle.dumps(arrays_to_send)
        signature = hmac.new(SRC.key, serialized, SRC.hashfunction).digest()
        assert len(signature) == SRC.hashsize
        message = signature + serialized + SRC.eof
        connection_socket.settimeout(240)
        connection_socket.sendall(message)
        while True:
            check = connection_socket.recv(10)
            if check == SRC.error:
                connection_socket.sendall(message)
            elif check == SRC.recv:
                break

    def get_message(self, connection_socket):
        """Routine to recieve binary data.
            Args:
              connection_socket (socket): a socket with a connection already established.
        """
        message = self.receiving_subroutine(connection_socket)
        return message

    def send_message(self, message_to_send, connection_socket):
        """Routine to send binary data. It sends it as many time as necessary.
            Args:
              message_to_send   (binary): binary data which is going to be sent.
              connection_socket (socket): a socket with a connection already established.
         """
        signature = hmac.new(SRC.key, message_to_send, SRC.hashfunction).digest()   # create the signature for the message
        assert len(signature) == SRC.hashsize   # check if the length of the signature is legal
        message = signature + message_to_send + SRC.eof # create the message with signature and EOF mark
        connection_socket.settimeout(240)       # set timeout for the sendall operation
        connection_socket.sendall(message)      # send the message using the socket connection
        while True:
            check = connection_socket.recv(10)  # receive status signal
            if check == SRC.error:              # if error signal is received, retry
                connection_socket.sendall(message)
            elif check == SRC.recv:             # message received successfully, end
                break
