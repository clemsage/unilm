#!/bin/bash

mkdir output/predictions -p
for seed in 42 43 44 45 46
do
	# Pre-trained
	for n in 8 16 32 64 128 256 600
	do
		run_name="SROIE_SeqLab_PreTrained_${n}docs_seed_${seed}"
		python run_seq_labeling.py --data_dir data/SROIE --model_type layoutlm --model_name_or_path ../../pre-trained-models/layoutlm-base-uncased/ --do_lower_case --output_dir output/models/${run_name} --labels data/SROIE/labels.txt --do_train --max_steps 1000 --nb_max_docs $n --seed $seed --per_gpu_train_batch_size 8 --evaluate_during_training --save_steps -1 --logging_steps 50 --overwrite_cache --fp16
		python run_seq_labeling.py --data_dir data/SROIE --model_type layoutlm --model_name_or_path ../../pre-trained-models/layoutlm-base-uncased/ --do_lower_case --output_dir output/models/${run_name} --labels data/SROIE/labels.txt --do_predict --per_gpu_eval_batch_size 8 --overwrite_cache --fp16
		python SROIE.py --do_post_processing --predictions output/models/${run_name}/test_predictions.txt
		zip output/predictions/${run_name}.zip output/models/${run_name}/post_processed_predictions/* -j -q
	done

	# Random initialization
	for n in 8 16 32 64 128 256 600
	do
		run_name="SROIE_SeqLab_Random_${n}docs_seed_${seed}"
		python run_seq_labeling.py --data_dir data/SROIE --model_type layoutlm --config_name ../../pre-trained-models/layoutlm-base-uncased/ --tokenizer_name ../../pre-trained-models/layoutlm-base-uncased/ --do_lower_case --output_dir output/models/${run_name} --labels data/SROIE/labels.txt --do_train --max_steps 2000 --nb_max_docs $n --seed $seed --per_gpu_train_batch_size 8 --evaluate_during_training --save_steps -1 --logging_steps 50 --overwrite_cache --fp16
		python run_seq_labeling.py --data_dir data/SROIE --model_type layoutlm --model_name_or_path ../../pre-trained-models/layoutlm-base-uncased/ --do_lower_case --output_dir output/models/${run_name} --labels data/SROIE/labels.txt --do_predict --per_gpu_eval_batch_size 8 --overwrite_cache --fp16
		python SROIE.py --do_post_processing --predictions output/models/${run_name}/test_predictions.txt
		zip output/predictions/${run_name}.zip output/models/${run_name}/post_processed_predictions/* -j -q
	done

	# Bidirectional LSTM
	for n in 8 16 32 64 128 256 600
	do
		run_name="SROIE_SeqLab_BLSTM_${n}docs_seed_${seed}"
		python run_seq_labeling.py --data_dir data/SROIE --model_type blstm --config_name ../../configs/BLSTM.json --tokenizer_name ../../pre-trained-models/layoutlm-base-uncased/ --do_lower_case --output_dir output/models/${run_name} --labels data/SROIE/labels.txt --do_train --max_steps 2000 --nb_max_docs $n --seed $seed --per_gpu_train_batch_size 8 --evaluate_during_training --save_steps -1 --logging_steps 50 --overwrite_cache --fp16 --learning_rate 0.005
		python run_seq_labeling.py --data_dir data/SROIE --model_type blstm --config_name ../../configs/BLSTM.json --tokenizer_name ../../pre-trained-models/layoutlm-base-uncased/  --do_lower_case --output_dir output/models/${run_name} --labels data/SROIE/labels.txt --do_predict --per_gpu_eval_batch_size 8 --overwrite_cache --fp16
		python SROIE.py --do_post_processing --predictions output/models/${run_name}/test_predictions.txt
		zip output/predictions/${run_name}.zip output/models/${run_name}/post_processed_predictions/* -j -q
	done
done
