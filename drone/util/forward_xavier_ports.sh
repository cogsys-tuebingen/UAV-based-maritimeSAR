#!/bin/sh

ssh -L 134.2.173.21:554:192.168.55.1:554  -L 134.2.173.21:8554:192.168.55.1:8554  -L 134.2.173.21:9999:192.168.55.1:9999  -L 134.2.173.21:5005:192.168.55.1:5005  nvidia@192.168.55.1
