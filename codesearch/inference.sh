#!/usr/bin/env bash


#inference
lang=python #programming language
idx=0 #test batch idx
#for idx in 0 .. 21
#do
CUDA_VISIBLE_DEVICES=0 python  run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models/$lang/checkpoint-best/ \
--test_result_dir ./results/$lang/${idx}_batch_result.txt
#done

#Evaluation
#python mrr.py