#!/usr/bin/env python3
"""在 RoboCasa365 simulator client 与 X-WAM policy server 之间转发消息。"""

from __future__ import annotations

import argparse
import logging
from collections import deque


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="*", help="监听地址，默认所有本机接口。")
    parser.add_argument("--frontend-port", type=int, default=10086, help="Simulator client 端口。")
    parser.add_argument("--backend-port", type=int, default=10087, help="Policy server 端口。")
    args = parser.parse_args()
    for name, port in (("frontend", args.frontend_port), ("backend", args.backend_port)):
        if not 1 <= port <= 65535:
            parser.error(f"{name} port 必须位于 1..65535")
    if args.frontend_port == args.backend_port:
        parser.error("frontend/backend 不能使用同一端口")

    import zmq

    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.ROUTER)
    frontend.setsockopt(zmq.LINGER, 0)
    backend.setsockopt(zmq.LINGER, 0)
    frontend.bind(f"tcp://{args.host}:{args.frontend_port}")
    backend.bind(f"tcp://{args.host}:{args.backend_port}")
    logging.info(
        "RoboCasa365 broker ready: frontend=tcp://%s:%d backend=tcp://%s:%d",
        args.host,
        args.frontend_port,
        args.host,
        args.backend_port,
    )

    available_servers: deque[bytes] = deque()
    available_set: set[bytes] = set()
    pending_requests: deque[tuple[bytes, bytes]] = deque()
    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    poller.register(backend, zmq.POLLIN)

    try:
        while True:
            sockets = dict(poller.poll())
            if backend in sockets:
                frames = backend.recv_multipart()
                if len(frames) == 2 and frames[1] == b"READY":
                    server_id = frames[0]
                    if server_id not in available_set:
                        available_servers.append(server_id)
                        available_set.add(server_id)
                    logging.info(
                        "policy server READY id=%s available=%d",
                        server_id.hex()[:12],
                        len(available_servers),
                    )
                elif len(frames) == 4 and frames[1] == b"RESULT":
                    server_id, _, client_id, payload = frames
                    frontend.send_multipart([client_id, payload])
                    logging.info(
                        "result forwarded server=%s client=%s bytes=%d",
                        server_id.hex()[:12],
                        client_id.hex()[:12],
                        len(payload),
                    )
                else:
                    logging.error("discard malformed backend message frame_count=%d", len(frames))

            if frontend in sockets:
                frames = frontend.recv_multipart()
                if len(frames) != 2:
                    logging.error("discard malformed frontend message frame_count=%d", len(frames))
                else:
                    client_id, payload = frames
                    pending_requests.append((client_id, payload))
                    logging.info(
                        "request queued client=%s bytes=%d pending=%d",
                        client_id.hex()[:12],
                        len(payload),
                        len(pending_requests),
                    )

            while available_servers and pending_requests:
                server_id = available_servers.popleft()
                available_set.remove(server_id)
                client_id, payload = pending_requests.popleft()
                backend.send_multipart([server_id, b"WORK", client_id, payload])
                logging.info(
                    "request dispatched server=%s client=%s pending=%d",
                    server_id.hex()[:12],
                    client_id.hex()[:12],
                    len(pending_requests),
                )
    except KeyboardInterrupt:
        logging.info("broker stopped by user")
        return 0
    finally:
        frontend.close()
        backend.close()
        context.term()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
