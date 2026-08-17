#!/usr/bin/env bash
# usage: rails_netplan.sh <octet>  : persistent dual-rail static addressing via netplan (survives reboot)
set -u; O=$1
sudo -n cp /etc/netplan/99-qsfp-stack.yaml /etc/netplan/99-qsfp-stack.yaml.bak-$(date +%Y%m%d%H%M) 2>/dev/null
sudo -n tee /etc/netplan/99-qsfp-stack.yaml >/dev/null <<Y
# dual-rail RoCEv2 fabric, static, MTU 9000. rail A = 192.168.100.$O, rail B = 192.168.101.$O
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      dhcp6: false
      addresses: [192.168.100.$O/24]
      mtu: 9000
    enP2p1s0f0np0:
      dhcp4: false
      dhcp6: false
      addresses: [192.168.101.$O/24]
      mtu: 9000
Y
sudo -n chmod 600 /etc/netplan/99-qsfp-stack.yaml
sudo -n nmcli con delete railB >/dev/null 2>&1
sudo -n netplan generate 2>&1 | grep -v "^$" | tail -2
sudo -n netplan apply 2>&1 | grep -viE "warning|^$" | tail -2
sleep 4
echo "$(hostname): live=$(ip -4 -o addr show enp1s0f0np0 | awk '{print $4}' | tr '\n' ' ')| $(ip -4 -o addr show enP2p1s0f0np0 | awk '{print $4}' | tr '\n' ' ') mtu=$(cat /sys/class/net/enp1s0f0np0/mtu)/$(cat /sys/class/net/enP2p1s0f0np0/mtu) NM=$(nmcli -t -f NAME,DEVICE con show --active | grep -E 'enp1s0f0np0|enP2p1s0f0np0' | tr '\n' ' ')"
