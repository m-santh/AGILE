#!/bin/bash

AGILE_PATH=/home/ub-12-3/san/bam/AGILE
sudo LD_LIBRARY_PATH=${AGILE_PATH}/driver/gdrcopy/src ${@}
