import socket
import netifaces

def get_all_internal_ips():
    ips = []
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            ips.extend([addr['addr'] for addr in addrs[netifaces.AF_INET]])
    return ips

if __name__ == '__main__':
    print(get_all_internal_ips())
