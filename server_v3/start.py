from findabac import server

# Chatgpt storage wisdom: https://chatgpt.com/c/68570890-4390-800b-b482-69d4c93982ec


instance = server.WebServer()

def cleanup():
    instance.shutdown()

def handle_signal(sig, frame):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_signal)  # Ctrl+C
atexit.register(cleanup)  # On Exit

instance.run()