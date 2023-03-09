#!/bin/bash

num=100
iid=0


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./save_path" ]; then
    mkdir ./save_path
fi

rm -rf logs/*
rm -rf save_path/*

python multiprocess_train_ps.py >./logs/log_0.txt 2>&1 &

sleep 20s

for((i = 1; i <= num; i++))
do
    python train_client.py --index_num=$i >./logs/log_$i.txt 2>&1 &
done

echo "Runing $num workers."

exit 0
