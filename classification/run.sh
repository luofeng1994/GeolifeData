#!/bin/bash
set -e -x

python main.py --num_steps 200 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike --is_training train --train_data_dir ./Data_Geolife/train --record False
python main.py --num_steps 200 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike --is_training test --test_data_dir ./Data_measured/all_cl --record True --metrics_record_file measured_record.csv
python main.py --num_steps 200 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike --is_training test --test_data_dir ./Data_measured/all_init --record True --metrics_record_file measured_record.csv
