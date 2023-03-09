#!/bin/bash

# 使用方法 bash run.sh -n int(worker数量)

num=100
iid=0

# 接收参数
while getopts ":n:d:c:" opt
do
    case $opt in
        n)
        echo "参数n的值$OPTARG"
        num=$OPTARG
        ;;
        d)
        echo "参数iid的值$OPTARG"
		iid=$OPTARG
        ;;
        c)
        echo "参数c的值$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./save_path" ]; then
    mkdir ./save_path
fi

rm -rf logs/*
# rm -rf save_path/*

# 开始执行python创建所有worker
python3.6 multiprocess_train_ps.py >./logs/log_0.txt 2>&1 &
# python multiprocess_train_ps.py >./logs/log_0.txt 2>&1 &

sleep 20s

for((i = 1; i <= num; i++))
do
    python3.6 train_client.py --index_num=$i >./logs/log_$i.txt 2>&1 &
    # python train_client.py --index_num=$i >./logs/log_$i.txt 2>&1 &
done
done

echo "Runing $num workers."

exit 0
