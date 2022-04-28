#!/bin/sh
source ../deepcegar_venv/bin/activate

START=$1
TSTART=$2
TEND=$3

while [ $TSTART -lt $TEND ]
do
	TEMP=`expr $TSTART + 1`
	while [ $START -lt 100 ]
	do
		END=`expr $START + 1`
		echo $START
		echo $END
		python3 . --domain refinepoly --dataset mnist --x_input_dataset ../benchmark/mnist_challenge/x_y/x0.txt --y_input_dataset ../benchmark/mnist_challenge/x_y/y0.txt --output ../experiment/raw/refine_causal_test/ --use_abstract_attack --start $START --end $END --test_start $TSTART --test_end $TEMP
		START=$END
	done
	TSTART=TEMP
done