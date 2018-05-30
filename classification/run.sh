#!/bin/bash
set -e -x

#for measured data
#python main.py --num_steps 200 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike --is_training train --train_data_dir ./Data_Geolife/train --record False
#python main.py --num_steps 200 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike --is_training test --test_data_dir ./Data_measured/all_cl --record True --metrics_record_file measured_record.csv
#python main.py --num_steps 200 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike --is_training test --test_data_dir ./Data_measured/all_init --record True --metrics_record_file measured_record.csv


#different num_layers
python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_num_layers
python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_num_layers

python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=2  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_num_layers
python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=2  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_num_layers

python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=3  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_num_layers
python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=3  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_num_layers

python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=4  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_num_layers
python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=4  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_num_layers

#different epochs
python main.py --num_steps 300 --epochs 10 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_epochs
python main.py --num_steps 300 --epochs 10 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_epochs

python main.py --num_steps 300 --epochs 20 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_epochs
python main.py --num_steps 300 --epochs 20 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_epochs

python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_epochs
python main.py --num_steps 300 --epochs 30 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_epochs

python main.py --num_steps 300 --epochs 40 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_epochs
python main.py --num_steps 300 --epochs 40 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_epochs

python main.py --num_steps 300 --epochs 50 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_epochs
python main.py --num_steps 300 --epochs 50 --learning_rate 0.0005 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_epochs

#different learning rate
python main.py --num_steps 300 --epochs 50 --learning_rate 0.0002 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_lr
python main.py --num_steps 300 --epochs 50 --learning_rate 0.0002 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_lr

python main.py --num_steps 300 --epochs 50 --learning_rate 0.0004 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_lr
python main.py --num_steps 300 --epochs 50 --learning_rate 0.0004 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_lr

python main.py --num_steps 300 --epochs 50 --learning_rate 0.0006 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_lr
python main.py --num_steps 300 --epochs 50 --learning_rate 0.0006 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_lr

python main.py --num_steps 300 --epochs 50 --learning_rate 0.0008 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_lr
python main.py --num_steps 300 --epochs 50 --learning_rate 0.0008 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_lr

python main.py --num_steps 300 --epochs 50 --learning_rate 0.001 --num_layers=1  --classes walk,bike,bus,car --is_training train --train_data_dir ./Data_Geolife/train --record True --metrics_record_file record_2.csv --section DNN_lr
python main.py --num_steps 300 --epochs 50 --learning_rate 0.001 --num_layers=1  --classes walk,bike,bus,car --is_training test --train_data_dir ./Data_Geolife/test --record True --metrics_record_file record_2.csv --section DNN_lr

