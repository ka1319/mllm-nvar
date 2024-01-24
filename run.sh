#!/bin/bash

model=$1
dataset=$2
mode=$3
max_gen_len=$4
extra_args=$5

echo "Running $model on $dataset in $mode mode:"

if [ $model == "blip2" ]; then
    python main.py --model blip2-opt-2.7b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model blip2-opt-6.7b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model blip2-flan-t5-xl --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model blip2-flan-t5-xxl --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
elif [ $model == "fuyu" ]; then
    python main.py --model fuyu-8b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
elif [[ "${model}" =~ ^"gemini" ]]; then
    python main.py --model $model --dataset $dataset --mode $mode --max_generation_length $max_gen_len --raw $extra_args
elif [[ "${model}" =~ ^"gpt-4" ]]; then
    python main.py --model $model --dataset $dataset --mode $mode --max_generation_length $max_gen_len --base64 $extra_args
elif [ $model == "idefics" ]; then
    python main.py --model idefics-9b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model idefics-80b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --bf16 $extra_args
elif [ $model == "idefics-instruct" ]; then
    python main.py --model idefics-9b-instruct --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model idefics-80b-instruct --dataset $dataset --mode $mode --max_generation_length $max_gen_len --bf16 $extra_args
elif [ $model == "instructblip" ]; then
    python main.py --model instructblip-vicuna-7b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model instructblip-vicuna-13b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu --bf16 $extra_args
    python main.py --model instructblip-flan-t5-xl --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model instructblip-flan-t5-xxl --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
elif [ $model == "llava" ]; then
    python main.py --model llava-1.5-7b-hf --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model llava-1.5-13b-hf --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu --fp16 $extra_args
    python main.py --model bakLlava-v1-hf --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
elif [ $model == "mmicl" ]; then
    python main.py --model MMICL-vicuna-7b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model MMICL-vicuna-13b --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu --bf16 $extra_args
    python main.py --model MMICL-Instructblip-T5-xl --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
    python main.py --model MMICL-Instructblip-T5-xxl --dataset $dataset --mode $mode --max_generation_length $max_gen_len --gpu $extra_args
elif [ $model == "qwen" ]; then
    python main.py --model Qwen-VL --dataset $dataset --mode $mode --max_generation_length $max_gen_len --raw --gpu $extra_args
elif [ $model == "qwen-chat" ]; then
    python main.py --model Qwen-VL-Chat --dataset $dataset --mode $mode --max_generation_length $max_gen_len --raw --gpu $extra_args
fi
