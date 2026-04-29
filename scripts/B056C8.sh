#!/bin/bash

# Proper one
python capture.py --crop 175,115,320,280 730,240,400,200 1240,125,380,480 \
--width 1800 --height 1000 --wait 10 --bmp3 https://192.168.0.98:1881/ui $@

# python capture2.py --output out.bmp --capture "url=https://192.168.0.98:1881/ui;x=60;y=100;w=600;h=300;wait=10000;viewport=380x380" --capture "url=https://192.168.0.98:1881/ui;x=60;y=100;w=1200;h=1000;wait=1000;viewport=500x500