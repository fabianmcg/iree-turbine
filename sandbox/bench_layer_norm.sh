#!/bin/bash

configs=(
  "-t f32 -b 16 1 25 -n 192 -c"
  "-t f32 -b 16 1 23 -n 192 -c"
  "-t f32 -b 16 1 21 -n 192 -c"
  "-t f32 -b 16 256 1 -n 2048 -c"
  "-t f32 -b 16 256 -n 2048 -c"
  "-t bf16 -b 16 256 -n 2048 -c"
  "-t f32 -b 16 1536 -n 576 -c"
  "-t f32 -b 16 1536 -n 512 -c"
  "-t bf16 -b 16 257 -n 2048 -c"
)

for conf in "${configs[@]}";
do
  cmd="python layer_norm.py ${conf}"
  echo "********************************************************************************"
  echo "${cmd}"
  eval ${cmd}
  echo
done
