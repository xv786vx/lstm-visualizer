import socket

try:
    print("Resolving fc.yahoo.com...")
    socket.gethostbyname("fc.yahoo.com")
    print("DNS Resolution succeeded.")
except socket.gaierror as e:
    print(f"DNS Resolution failed: {e}")