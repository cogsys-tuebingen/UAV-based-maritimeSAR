#!/bin/sh
ffmpeg -analyzeduration 100M -probesize 100M -i "udp://127.0.0.1:3333?overrun_nonfatal=1&fifo_size=50000000" -map "0:0" -c:v copy -copyts -f mpegts udp://127.0.0.1:5000 -map "0:1" -c:v copy -copyts -f mpegts udp://127.0.0.1:5001

