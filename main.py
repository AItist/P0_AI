from main_functions import step1_get_index_from_args
import common.inference as inf
import modules._websocket0 as _websocket0

from common.data_ import P1_data_package_process
from common.enum_ import ePoses, eSegs

import asyncio
import websockets
import threading
import base64
import cv2
import numpy as np
import time

RUN_POSE = True
RUN_SEG = True
poseFlag = ePoses.MEDIAPIPE # 포즈 검출 시행시 어떤 모듈을 사용할지 결정
segFlag = eSegs.MEDIAPIPE # 영역 검출 시행시 어떤 모듈을 사용할지 결정

runFlag = True


async def client(recv_queue, send_queue, address, port, debug):
    import json

    uri = f"ws://{address}:{port}" #"ws://localhost:8080"
    print(f'uri : {uri}')
    
    async with websockets.connect(uri) as websocket:
        while runFlag:
            try:
                data = await websocket.recv()
                chunk_data = json.loads(data)
                
                decoded_bytes = base64.b64decode(chunk_data['frame'])
                nparr = np.frombuffer(decoded_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                chunk_data['frame'] = frame

                recv_queue.put(chunk_data)

                if send_queue.qsize() > 0:
                    # print('qsize : ', send_queue.qsize())
                    for i in range(send_queue.qsize()):
                        packet = send_queue.get()
                        await websocket.send(packet)
                    # print('8070 send data complete')
                    pass
            except Exception as e:
                print(e)
                pass
            # print(f"< {data}")

def run_client(recv_queue, send_queue, address, port, debug):
    asyncio.new_event_loop().run_until_complete(client(recv_queue, send_queue, address, port, debug))

def run_inference(recv_queue, send_queue, debug, socket_address='localhost', socket_port=8070):
    while True:
        try:
            if recv_queue.empty() or recv_queue.qsize() == 0:
                continue
            
            data = None
            data = recv_queue.get()
            
            if data is None:
                continue

            # data['camIndex'] # 카메라 인덱스
            # data['frame'] # 이미지
            pose, seg, pose_center = inf.ai_model_inference(data['camIndex'], data['frame'], debug, RUN_POSE, RUN_SEG, poseFlag, segFlag)

            packet = P1_data_package_process(pose, seg, pose_center, data['camIndex'])

            send_queue.put(packet)

            pass
        except Exception as e:
            print(e)

    pass

if __name__ == '__main__':
    index, sAddr, sPort, debug = step1_get_index_from_args()
    sAddr = 'localhost'
    # sAddr = '192.168.50.50'
    sPort = '8070'
    debug = True

    # import modules._socket as _socket    
    import modules._socket as _socket
    import queue
    import threading
    recv_queue = queue.Queue()
    send_queue = queue.Queue()

    thread_socket = threading.Thread(target=run_client, args=(recv_queue, send_queue, sAddr, sPort, debug))
    thread_socket.start()
    
    thread_inference = threading.Thread(target=run_inference, args=(recv_queue, send_queue, debug, sAddr, sPort))
    thread_inference.start()

    thread_socket.join()
    thread_inference.join()
    pass