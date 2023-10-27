import socket


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ip = '127.0.0.1'

    try:
        s.connect(("192.255.255.255", 1))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_port(port: int = 8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', port)) == 0:
            return find_port(port + 1)
        else:
            return port


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
