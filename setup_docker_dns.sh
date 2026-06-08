#!/bin/bash
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'CONF'
{
  "dns": ["8.8.8.8", "8.8.4.4"],
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me",
    "https://docker.m.daocloud.io"
  ]
}
CONF
service docker restart
