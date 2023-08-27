import argparse

debug = True

def get_args():

    # Create the parser
    parser = argparse.ArgumentParser(description='This is a sample program')

    # Add the arguments
    parser.add_argument('--index', type=int, help='This is the first argument')
    parser.add_argument('--sAddr', type=str, help='웹 소켓 서버 주소')
    parser.add_argument('--sPort', type=int, help='웹 소켓 서버 포트')
    parser.add_argument('--debug', type=int, help='디버그 모드')

    # Execute the parse_args() method
    args = parser.parse_args()

    argDebug = True if args.debug == 1 else False # 1이면 True, 0이면 False
    if debug:
        print(f'Argument 1 cam index: {args.index}')
        print(f'Argument 2 socket address: {args.sAddr}')
        print(f'Argument 3 socket port: {args.sPort}')
        
        print(f'Argument 4 debug: {argDebug}')

    return args.index, args.sAddr, args.sPort, argDebug

if __name__ == '__main__':
    get_args()