# Cybersecurity_LLM_for_Network_Security
LLM to achieve automated multi-class classification of IP surveillance network intrusions at scale

[Sparsh Camera] 
      |
   Ethernet cable
      |
[Laptop eno1]
      |
   Linux kernel (routing)
      |
[Alfa wlx00c0cab2afcc]
      |
   Wi-Fi (SparshCam)
      |
[Other laptop/phone]

What Each Component Does
Camera
Streams video over HTTPS (port 443) and RTSP (port 554)
Only connected via Ethernet — has no Wi-Fi
Doesn't know anything beyond your laptop

eno1
Your laptop's physical Ethernet port
Receives all camera traffic
Camera thinks your laptop is its only client

Linux Kernel (IP Forwarding)
echo 1 > /proc/sys/net/ipv4/ip_forward enables this
Without this, Linux drops packets not meant for itself
With this enabled, Linux acts as a router and forwards packets between interfaces

iptables MASQUERADE
When other laptop sends request to camera, source IP is 192.168.10.x
Camera only accepts 192.168.1.x connections
MASQUERADE rewrites source IP to 192.168.1.100 before sending to camera
Camera sees request coming from your laptop, responds normally
Your laptop then forwards response back to other laptop

Alfa
Broadcasts SparshCam Wi-Fi network
Acts as wireless access point
Any device connecting to SparshCam gets IP 192.168.10.x from dnsmasq

hostapd
Software that makes Alfa behave as a Wi-Fi access point
Handles WPA2 authentication for SparshCam
Without this, Alfa is just a Wi-Fi adapter, not an AP

dnsmasq
DHCP server — automatically assigns IPs to devices joining SparshCam
Gives within a range
Also handles DNS for connected devices

Not used in final setup
Was used during testing
