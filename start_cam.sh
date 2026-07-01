#!/bin/bash
echo "Loading Alfa driver..."
sudo modprobe 8814au

echo "Setting up Alfa interface..."
sudo ip link set wlx00c0cab2afcc up
sudo ip addr add <rpi ip addr> dev wlx00c0cab2afcc 2>/dev/null

echo "Setting up camera Ethernet..."
sudo ip link set eno1 up
sudo ip addr add <rpi ip addr> dev eno1 2>/dev/null

echo "Starting hostapd and dnsmasq..."
sudo systemctl start hostapd
sudo systemctl start dnsmasq

echo "Enabling routing..."
sudo sh -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
sudo iptables -F
sudo iptables -t nat -F
sudo iptables -P FORWARD ACCEPT
sudo iptables -t nat -A POSTROUTING -s <rpi ip addr> -o eno1 -j MASQUERADE
sudo systemctl stop docker 2>/dev/null

echo "Waiting for SparshCam to broadcast..."
sleep 10
nmcli dev wifi rescan ifname wlo1
sleep 3
nmcli dev wifi connect "SparshCam" password "s*****1***" ifname wlo1 2>/dev/null

echo "Done!"
echo "Camera URL: https://192.168.1.188"
echo "Login: admin / Admin@1234"
echo "Other devices: Connect to SparshCam (password: s*****1***) then open https://<ip addr>"
