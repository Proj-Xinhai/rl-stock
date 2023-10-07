import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange
import socket
import socketio
from typing import Any
from time import sleep

from multiprocessing import Process

import api.tasks as tasks
import api.works as works
from api.list_algorithm import list_algorithm
from api.list_helper import list_helper
from api.worker import run_work

SERVICES = []


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


def on_service_state_change(zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange):
    if state_change is ServiceStateChange.Added:
        info = zeroconf.get_service_info(service_type, name)
        if info:
            if info.properties[b'service'] == b'rl-stock':
                SERVICES.append(info.server.strip('.'))

    if state_change is ServiceStateChange.Removed:
        info = zeroconf.get_service_info(service_type, name)
        if info:
            if info.properties[b'service'] == b'rl-stock':
                SERVICES.remove(info.server.strip('.'))


app = fastapi.FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
sio_asgi_app: Any = socketio.ASGIApp(sio, app)

app.add_route("/socket.io/", route=sio_asgi_app, methods=["GET", "POST"])
app.add_websocket_route("/socket.io/", sio_asgi_app)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "services": SERVICES}


@sio.event
async def ping(sid):
    return "pong"


@sio.event
async def connect(sid, environ):
    await sio.emit("update_tasks", tasks.list_tasks(), room=sid)
    await sio.emit("update_works", works.list_works(), room=sid)
    await sio.emit("update_helper", list_helper(), room=sid)
    await sio.emit("update_algorithm", list_algorithm(), room=sid)
    print(f"connect {sid}")


@sio.event
async def disconnect(sid):
    print(f"disconnect {sid}")


@sio.event
async def create_task(sid, data):
    status, msg, detail = tasks.create_task(**data)
    await sio.emit("update_tasks", tasks.list_tasks(), room=None)
    return status, msg, detail


@sio.event
async def remove_task(sid, data):
    status, msg, detail = tasks.remove_task(data)
    await sio.emit("update_tasks", tasks.list_tasks(), room=None)
    return status, msg, detail


@sio.event
async def create_work(sid, data):
    status, msg, detail = works.create_work(**data)
    await sio.emit("update_works", works.list_works(), room=None)
    return status, msg, detail


@sio.event
async def update_all(sid):
    await sio.emit("update_tasks", tasks.list_tasks(), room=None)
    await sio.emit("update_works", works.list_works(), room=None)
    await sio.emit("update_helper", list_helper(), room=None)
    await sio.emit("update_algorithm", list_algorithm(), room=None)


@sio.event
async def get_scaler(sid, data):
    return works.get_scaler(data)


def main():
    info = ServiceInfo(
        "_http._tcp.local.",
        "RL Stock._http._tcp.local.",
        addresses=[socket.inet_aton(get_local_ip())],
        port=8000,
        properties={"service": "rl-stock"},
        server="rl-stock.local.",
    )

    zeroconf = Zeroconf()
    zeroconf.register_service(info)
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", handlers=[on_service_state_change])

    uvicorn.run(app, host="0.0.0.0", port=8000)

    browser.cancel()

    zeroconf.unregister_service(info)
    zeroconf.close()


def worker():
    w = None
    try:
        while True:
            for w in works.list_works():
                try:
                    if w['status'] == 0:
                        works.set_work_status(w['id'], 1, 'start running')
                        status, msg, detail = run_work(w['id'])
                        if status:
                            works.set_work_status(w['id'], 2, 'finished')
                        else:
                            works.set_work_status(w['id'], -1, detail)
                except Exception as e:
                    works.set_work_status(w['id'], -1, str(e))
                    print(e)

            print("waiting for 10 seconds")
            sleep(10)
    except KeyboardInterrupt:
        if w is not None:
            works.set_work_status(w['id'], -1, 'KeyboardInterrupt')
    # except Exception as e:
    #     if w is not None:
    #         task.set_work_status(w['id'], -1, str(e))
    #     print(e)


if __name__ == "__main__":
    p_main = Process(target=main)
    p_main.start()

    p_worker = Process(target=worker)
    p_worker.start()

    p_main.join()
    p_worker.join()
