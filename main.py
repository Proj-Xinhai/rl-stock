from api import tasks, works, get_local_ip, find_port, get_version
from api.algorithms import list_algorithms
from api.data_locators import list_locators
from api.environments import list_environments
from backtest.backtest import backtest
from worker import worker
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, NonUniqueNameException, ServiceListener
import socket
import os
from time import sleep
from uuid import uuid4
from multiprocessing import Process
import socketio
import eventlet
from eventlet import wsgi


class ZFServiceListener(ServiceListener):
    def __init__(self):
        super().__init__()
        self.services = []

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        for item in self.services:
            if item['name'] == name:
                self.services.remove(item)

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info.properties[b'service'] == b'rl-stock':
            self.services.append({
                'name': name,
                'server': info.server.strip('.'),
                'port': info.port,
            })


def zeroconf_service(port: int):
    index_service_info = ServiceInfo(
        '_http._tcp.local.',
        'RL Stock Index._http._tcp.local.',
        addresses=[socket.inet_aton(get_local_ip())],
        port=port,
        properties={'service': 'rl-stock-index'},
        server='rl-stock.local.',
    )
    service = uuid4().hex[:7]
    service_info = ServiceInfo(
        '_http._tcp.local.',
        f'RL Stock {service}._http._tcp.local.',
        addresses=[socket.inet_aton(get_local_ip())],
        port=port,
        properties={'service': 'rl-stock'},
        server=f'rl-stock-{service}.local.',
    )

    zeroconf = Zeroconf()

    zeroconf.register_service(service_info)  # register service (this must success immediately)

    while True:
        # try to register service until success
        try:
            zeroconf.register_service(index_service_info)
            print('index service started')

            try:
                while True:
                    sleep(10)  # keep service alive
            except KeyboardInterrupt:
                print('index service stopped')

            break  # register success
        except NonUniqueNameException:
            sleep(10)  # register fail, wait for 10 seconds

    zeroconf.unregister_service(index_service_info)
    zeroconf.unregister_service(service_info)
    zeroconf.close()


def api(port: int):
    zeroconf = Zeroconf()
    listener = ZFServiceListener()
    browser = ServiceBrowser(zeroconf, '_http._tcp.local.', listener)

    sio = socketio.Server(async_mode='eventlet', cors_allowed_origins='*')

    @sio.event(namespace='/service')
    def connect(sid: str, environ: dict):
        print(f'connected: {sid}')

    @sio.event(namespace='/service')
    def disconnect(sid: str):
        print(f'disconnected: {sid}')

    @sio.event(namespace='/service')
    def get_services(sid):
        print(f'get services: {sid}')
        return listener.services

    @sio.event
    def connect(sid, environ: dict):
        sio.emit("git_version", get_version(), room=sid)
        sio.emit("update_tasks", tasks.list_tasks(), room=sid)
        sio.emit("update_works", works.list_works(), room=sid)
        sio.emit("update_algorithm", list_algorithms(), room=sid)
        sio.emit("update_data_locator", list_locators(), room=sid)
        sio.emit("update_environment", list_environments(), room=sid)
        print(f'connected: {sid}')

    @sio.event
    def update_all(sid):
        sio.emit("update_tasks", tasks.list_tasks(), room=None)
        sio.emit("update_works", works.list_works(), room=None)
        sio.emit("update_algorithm", list_algorithms(), room=None)
        sio.emit("update_data_locator", list_locators(), room=None)
        sio.emit("update_environment", list_environments(), room=None)

    @sio.event
    def disconnect(sid):
        print(f"disconnect {sid}")

    @sio.event
    def ping(sid):
        return sid

    @sio.event
    def create_task(sid, data):
        status, msg, detail = tasks.create_task(**data)
        sio.emit("update_tasks", tasks.list_tasks(), room=None)
        return status, msg, detail

    @sio.event
    def remove_task(sid, data):
        status, msg, detail = tasks.remove_task(data)
        sio.emit("update_tasks", tasks.list_tasks(), room=None)
        return status, msg, detail

    @sio.event
    def export_task(sid, data):
        return tasks.export_task(data)

    @sio.event
    def get_scalar(sid, data):
        return works.get_scalar(**data)

    @sio.event
    def create_work(sid, data):
        status, msg, detail = works.create_work(**data)
        sio.emit("update_works", works.list_works(), room=None)
        return status, msg, detail

    @sio.event
    def export_work(sid, data):
        return works.export_work(data)

    @sio.event
    def backtesting(sid, data):
        return backtest(**data)

    app = socketio.WSGIApp(sio)
    wsgi.server(eventlet.listen(('', port)), app)

    browser.cancel()
    zeroconf.close()


def run_worker():
    try:
        while True:
            for w in works.list_works():
                if w['status'] == 0:
                    works.set_status(w['id'], 1, 'start running')
                    status, _, detail = worker(w['id'])
                    if status:
                        works.set_status(w['id'], 2, 'finished')
                    else:
                        works.set_status(w['id'], -1, detail)

            print('waiting for 10 seconds')
            sleep(10)
    except KeyboardInterrupt:
        print('worker stopped')


def main():
    if not os.path.exists('tasks/works'):
        os.makedirs('tasks/works')

    port = find_port(8000)
    print(port)

    p_zeroconf_service = Process(target=zeroconf_service, args=(port,))
    p_zeroconf_service.start()

    p_api = Process(target=api, args=(port,))
    p_api.start()

    p_worker = Process(target=run_worker)
    p_worker.start()

    p_zeroconf_service.join()
    p_api.join()
    p_worker.join()

    print('bye!')


if __name__ == '__main__':
    main()
