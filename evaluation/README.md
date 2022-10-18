# Evaluation of real-world data
In the following the procedure is described how the presented data are created and evaluated. All steps are fully reproducible and all data is available open source.

## Data set
The results are created from real-world data, recorded during the final event of the [AC@CES](https://www.indyautonomouschallenge.com/). The procedure how to process them is described in the next session.
The following rosbags are given:

| Name | Description |
| -------------- | ------------- |
| RUN2_semi_final | Semi final of the AC@CES with overtaking maneuvers up to 220 km/h
| RUN3_final | Final run of the AC@CES with overtaking maneuvers up to 270 km/h

## Replay data
The replay of the rosbags is described here: [Readme](bag_play/README.md).

Note: The logs of all replayed data is available via download as well ([Link](https://doi.org/10.5281/zenodo.7220695), folder `logs`). Paste the folder `logs/2022_10_01` into `tracking/tracking/logs` to directly use the replayed data for the following evaluation.

## Log overview:
Variation of the data is conducted as follows:
### Default (parameters as set in the repo):
| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_00 | - | RUN2_semi_final with default configuration |
| 2022_10_01-00_00_01 | - | RUN3_final with default configuration |

### Cycle Frequency Variation:

| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_06 | Cycle Frequency = 10.0 Hz | RUN2_semi_final, varied ROS2 Node Frequency |
| 2022_10_01-00_00_07 | Cycle Frequency = 10.0 Hz | RUN3_final, varied ROS2 Node Frequency|
| 2022_10_01-00_00_04 | Cycle Frequency = 20.0 Hz | RUN2_semi_final, varied ROS2 Node Frequency |
| 2022_10_01-00_00_05 | Cycle Frequency = 20.0 Hz | RUN3_final, varied ROS2 Node Frequency|
| 2022_10_01-00_00_02 | Cycle Frequency = 100.0 Hz | RUN2_semi_final, varied ROS2 Node Frequency|
| 2022_10_01-00_00_03 | Cycle Frequency = 100.0 Hz | RUN3_final, varied ROS2 Node Frequency|

### Detection pipelines:
| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_08 | active_sensors = ["lidar_cluster"] | RUN2_semi_final, single detection modality |
| 2022_10_01-00_00_09 | active_sensors = ["lidar_cluster"] | RUN3_final, single detection modality |
| 2022_10_01-00_00_10 | active_sensors = ["radar"] | RUN2_semi_final, single detection modality |
| 2022_10_01-00_00_11 | active_sensors = ["radar"] | RUN3_final, single detection modality |
### Out of track filter
| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_12 | bounds buffer = -0.75m | RUN2_semi_final, varied outer filter bounds |
| 2022_10_01-00_00_13 | bounds buffer = -0.75m | RUN3_final, varied outer filter bounds |
| 2022_10_01-00_00_14 | bounds buffer = 0.0m | RUN2_semi_final, varied outer filter bounds |
| 2022_10_01-00_00_15 | bounds buffer = 0.0m | RUN3_final, varied outer filter bounds |

### Matching Distance
| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_16 | max matching distance = 1.0 | RUN2_semi_final, varied matching distance |
| 2022_10_01-00_00_17 | max_matching_distance = 1.0 | RUN3_final, varied matching distance |
| 2022_10_01-00_00_18 | max matching distance = 2.0 | RUN2_semi_final, varied matching distance |
| 2022_10_01-00_00_19 | max_matching_distance = 2.0 | RUN3_final, varied matching distance |
| 2022_10_01-00_00_20 | max matching distance = 7.0 | RUN2_semi_final, varied matching distance |
| 2022_10_01-00_00_21 | max_matching_distance = 7.0 | RUN3_final, varied matching distance |

### Death Counter
| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_22 | counter = 6 | RUN2_semi_final, varied death threshold |
| 2022_10_01-00_00_23 | counter = 6 | RUN3_final, varied death counter threshold |
| 2022_10_01-00_00_24 | counter = 12 | RUN2_semi_final, varied death threshold |
| 2022_10_01-00_00_25 | counter = 12 | RUN3_final, varied death threshold |
| 2022_10_01-00_00_46 | counter = 37 | RUN2_semi_final, varied death threshold |
| 2022_10_01-00_00_47 | counter = 37 | RUN3_final, varied death threshold |

### EKF Frequency
| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_26 | filter frequency = 10.0 | RUN2_semi_final, varied filter update frequency |
| 2022_10_01-00_00_27 | filter frequency = 10.0 | RUN3_final, varied filter update frequency |
| 2022_10_01-00_00_28 | filter frequency = 50.0 | RUN2_semi_final, varied filter update frequency |
| 2022_10_01-00_00_29 | filter frequency = 50.0 | RUN3_final, varied filter update frequency |
| 2022_10_01-00_00_30 | filter frequency = 200.0 | RUN2_semi_final, varied filter update frequency |
| 2022_10_01-00_00_31 | filter frequency = 200.0 | RUN3_final, varied filter update frequency |

### Merge Distance
| Logs | Variation | Description | 
| -------------- | -------------- | ------------- |
| 2022_10_01-00_00_32 | overlapping detection distance = 1.7 | RUN2_semi_final, varied overlap distance in m (merges detections) |
| 2022_10_01-00_00_33 | overlapping detection distance = 1.7 | RUN3_final, varied overlap distance in m (merges detections) |
| 2022_10_01-00_00_34 | overlapping detection distance = 3.4 | RUN2_semi_final, varied overlap distance in m (merges detections) |
| 2022_10_01-00_00_35 | overlapping detection distance = 3.4 | RUN3_final, varied overlap distance in m (merges detections) |
| 2022_10_01-00_00_36 | overlapping detection distance = 7.2 | RUN2_semi_final, varied overlap distance in m (merges detections) |
| 2022_10_01-00_00_37 | overlapping detection distance = 7.2 | RUN3_final, varied overlap distance in m (merges detections) |


## Prepare evaluation

Note: Before you process all the data by hand, have a look at the section `Run all evaluation`.

To prepare the evaluation, run the following commands. It is recommended to run the data preparation pipeline for both runs (semi-final and final) of one parameter setting:
```
python evaluation/prepare_eval.py --timestamp <Y_%m_%d-%H_%M_%S> --save_file_name RUN2_semi_final --results_save_path evaluation/results/<target_dir>
python evaluation/prepare_eval.py --timestamp <Y_%m_%d-%H_%M_%S> --save_file_name RUN3_final --results_save_path evaluation/results/<target_dir>
```
e.g for the default data:
```
python evaluation/prepare_eval.py --timestamp 2022_10_01-00_00_00 --save_file_name RUN2_semi_final --results_save_path evaluation/results/__default
python evaluation/prepare_eval.py --timestamp 2022_10_01-00_00_01 --save_file_name RUN3_final --results_save_path evaluation/results/__default
```
## Visualize
Check the file `<save_file_name>_<timestamp>_stats.txt` in the `results_save_path` for a rough analysis.

In case you want to visualize the evaluation, run the following command:
```
python evaluation/evaluation.py --timestamp <Y_%m_%d-%H_%M_%S> --save_file_name RUN2_semi_final --show_filtered --results_save_path evaluation/results/<target_dir>
python evaluation/evaluation.py --timestamp <Y_%m_%d-%H_%M_%S> --save_file_name RUN3_final --show_filtered --results_save_path evaluation/results/<target_dir>
```
e.g. for the default data:
```
python evaluation/evaluation.py --timestamp 2022_10_01-00_00_00 --save_file_name RUN2_semi_final --show_filtered --results_save_path evaluation/results/__default
python evaluation/evaluation.py --timestamp 2022_10_01-00_00_01 --save_file_name RUN3_final --show_filtered --results_save_path evaluation/results/__default
```
The [logging tool](../tools/visualize_logfiles.py) will be opened and you can slide through the logs step by step.

## Run evaluation

Note: Before you process all the data by hand, have a look at the section `Run all evaluation`.

After the data is prepared we can run the overall evaluation for a given parameter configuration:
```
python evaluation/evaluation.py --load_path evaluation/results/<target_dir>
```
e.g. for the default data:
```
python evaluation/evaluation.py --load_path evaluation/results/__default
```
The storage location is `<load-path>/plots` (.pdf-format). Additionally a file with the overall stats is created (`<target_dir>_stats.txt`), check if out for the overall analysis.

## Run all evaluation
To process all of the above mentioned logs (see section `Log overview`) just run:
```
./evaluation/evaluation_all.sh
```
You find all processed data in the folder `evaluation/results`.

## Data sample

Note: You have to download the logs from ([here](https://doi.org/10.5281/zenodo.7220695), folder `logs`) and you have to paste `logs/2022_10_01/00_00_01` into `tracking/tracking/logs/2022_10_01/00_00_01` to visualize the sample.

Show case of a data sample can be visualized with the following script:
```
python evaluation/show_sample.py
```
A .mp4-video is created, which shows the raw and matched detections as well as the tracked objects. You can vary the logged data and desired object with the respective arguments (see arg-parser in [show_sample.py](./show_sample.py).


## Delay compensation

Note: You have to download the logs from ([here](https://doi.org/10.5281/zenodo.7220695), folder `logs`) and you have to paste `logs/2022_10_01/00_00_01` into `tracking/tracking/logs/2022_10_01/00_00_01` to visualize the delay compensation example.


Show case of the implemented delay compensation can be visualized with the following script:
```
python evaluation/show_delay_comp.py
```
A comprehensive plot is created, which visualizes the delay compensation. The plots are stored in `evaluation/results/__default/plots`. The default setting shows a sample during the fastest overtake maneuver of TUM with an ego-speed of 265 km/h.
