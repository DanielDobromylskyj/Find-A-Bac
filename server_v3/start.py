from findabac import server

import signal
import atexit
import sys


instance = server.WebServer()

def cleanup():
    instance.shutdown()

def handle_signal(sig, frame):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_signal)  # Ctrl+C
atexit.register(cleanup)  # On Exit (Normal)

instance.run()