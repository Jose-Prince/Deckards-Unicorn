import socket

HOST = "127.0.0.1"
PORT = 65432

print("Connecting to AI server...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print("Connected. Type 'quit' to exit.\n")

while True:
    msg = input("You: ").strip()
    if not msg:
        continue

    client_socket.sendall(msg.encode())

    if msg.lower() in ["quit", "exit"]:
        print("Closing connection.")
        break

    data = client_socket.recv(2048)
    print("AI:", data.decode())

client_socket.close()
