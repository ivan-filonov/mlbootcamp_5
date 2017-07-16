#!/bin/sh
set -x
for x in l1_*.py; do echo BEGIN $x;python3 $x 2>&1;echo END $x;done|grep -v LightGBM
for x in l2_*.py; do echo BEGIN $x;python3 $x 2>&1;echo END $x;done|grep -v LightGBM
