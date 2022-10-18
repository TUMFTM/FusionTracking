#!/bin/sh
TIMESTAMP_RUN2=$1
TIMESTAMP_RUN3=$2
TARGETDIR=$3

# prepare evaluation
python evaluation/prepare_eval.py --timestamp $TIMESTAMP_RUN2 --save_file_name RUN2_semi_final --results_save_path evaluation/results/$TARGETDIR
python evaluation/prepare_eval.py --timestamp $TIMESTAMP_RUN3 --save_file_name RUN3_final --results_save_path evaluation/results/$TARGETDIR

# # visualize your data
# python evaluation/evaluation.py --timestamp $TIMESTAMP_RUN2 --save_file_name RUN2_semi_final --show_filtered --results_save_path evaluation/results/$TARGETDIR
# python evaluation/evaluation.py --timestamp $TIMESTAMP_RUN3 --save_file_name RUN3_final --show_filtered --results_save_path evaluation/results/$TARGETDIR

# run overall evaluation
# add --show_plt to open to plots, they are stored as .pdf to the <load-path>/plots
python evaluation/evaluation.py --load_path evaluation/results/$TARGETDIR
