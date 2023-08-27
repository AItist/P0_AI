def send_packet(packet, address='localhost', port=8080):
    import websocket
    # Create a connection to the WebSocket server
    ws = websocket.WebSocket()

    url = f'ws://{address}:{port}' # "ws://localhost:8080"

    # Connect to the server
    ws.connect(url)

    # Send the packet
    ws.send(packet)
    print(f'socket send complete. url: {url}')

    # Close the connection
    ws.close()