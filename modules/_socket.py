
runFlag = True

class SocketClient:
    def __init__(self):
        self.runFlag = True
        pass

async def client(recv_queue, send_queue, address, port, debug, socketInstance):
    import asyncio
    import websockets
    import threading
    import base64
    import cv2
    import numpy as np
    import json

    uri = f"ws://{address}:{port}" #"ws://localhost:8080"
    print(f'uri : {uri}')
    
    chunk_datas = []
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                data = await websocket.recv()

                chunk_data = json.loads(data)
                chunk_datas.extend(chunk_data['chunk']['data'])
                # chunk_datas.append(chunk_data['chunk']['data'])
                # print(type(chunk_data))
                # print(chunk_data['last'])
                
                if chunk_data['last']:
                    # bytes 배열을 문자열로 변환
                    data_str = ''.join(chr(b) for b in chunk_datas)
                    # # bytes를 문자열로 변환
                    # data_str = chunk_datas.decode('utf-8')

                    # 문자열을 dict로 변환
                    data_dict = json.loads(data_str)

                    decoded_bytes = base64.b64decode(data_dict['frame'])
                    nparr = np.frombuffer(decoded_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    data_dict['frame'] = frame

                    # large_data = ''.join(chunk_datas)
                    # data = ''.join(chunk_datas)
                    chunk_datas = []

                    # recv_queue.put(data_dict)
                    print("get data")


                # parsed_data = json.loads(data)
                # recv_queue.put(data)

                if send_queue.qsize() > 0:
                    # print('qsize : ', send_queue.qsize())
                    for i in range(send_queue.qsize()):
                        packet = send_queue.get()
                        await websocket.send(json.dumps(packet))
                    pass
            except Exception as e:
                print(e)
                chunk_datas = []
                pass
            # print(f"< {data}")

def run_client(recv_queue, send_queue, address, port, debug, socketInstance):
    asyncio.new_event_loop().run_until_complete(client(recv_queue, send_queue, address, port, debug, socketInstance))


if __name__ == '__main__':
    import queue
    recv_queue = queue.Queue()
    send_queue = queue.Queue()
    address = 'localhost'
    port = 8070
    debug = True

    socketInstance = SocketClient()
    thread = threading.Thread(target=run_client, args=(recv_queue, send_queue, address, port, debug))
    thread.start()
    thread.join()

