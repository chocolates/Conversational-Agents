#!/bin/bash

module load python/3.3.3
export PYTHONPATH=$HOME/python/lib64/python3.3/site-packages:$PYTHONPATH
bsub -W 3:00 -n 4 python chatbot_ubuntu.py