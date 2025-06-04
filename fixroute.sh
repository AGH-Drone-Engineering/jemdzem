#!/bin/bash

SSID="AndroidAP_8740"
TARGET_IP="192.168.241.1"
SUBNET="192.168.241.0/24"
INTERFACE="wlp2s0"

# Get currently connected SSID
CURRENT_SSID=$(nmcli -t -f active,ssid dev wifi | grep '^yes' | cut -d: -f2)

if [ "$CURRENT_SSID" != "$SSID" ]; then
    echo "Currently connected to SSID: $CURRENT_SSID"
    echo "Attempting to connect to $SSID..."

    nmcli dev wifi connect "$SSID" ifname "$INTERFACE"
    sleep 5  # Give it a few seconds to connect
else
    echo "Already connected to $SSID"
fi

# Get default gateway for the interface
GATEWAY=$(ip route show dev $INTERFACE | awk '/default/ {print $3}')

# Check if we can ping the target IP
ping -c 1 -W 2 $TARGET_IP > /dev/null

if [ $? -ne 0 ]; then
    echo "$TARGET_IP unreachable. Fixing route..."

    if [ -z "$GATEWAY" ]; then
        echo "Could not find default gateway for $INTERFACE."
        exit 1
    fi

    echo "Using gateway: $GATEWAY"

    # Delete existing route
    sudo ip route del $SUBNET 2>/dev/null

    # Add new route
    sudo ip route add $SUBNET via $GATEWAY

    echo "Route fixed via $GATEWAY"
else
    echo "$TARGET_IP is reachable. No action needed. Gateway $GATEWAY"
fi
