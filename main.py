from api import get_local_ip
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange
import socket
import uvicorn
import os
from time import sleep


SERVICES = []


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


def api():
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
    browser = ServiceBrowser(zeroconf,
                             "_http._tcp.local.",
                             handlers=[on_service_state_change])

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    try:
        while True:
            sleep(10)
    except KeyboardInterrupt:
        print("worker stopped")

    browser.cancel()

    zeroconf.unregister_service(info)
    zeroconf.close()


def main():
    if not os.path.exists('tasks/works'):
        os.makedirs('tasks/works')


