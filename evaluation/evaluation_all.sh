#!/bin/sh

# Set repository path
REPO_DIR=$HOME/source/FusionTracking

###############
## Template ##
###############
# TIMESTAMP_RUN2=$1
# TIMESTAMP_RUN3=$2
# TARGETDIR=$3
# $REPO_DIR/evaluation/evaluation_single.sh <TIMESTAMP_RUN2> <TIMESTAMP_RUN3> <TARGETDIR>

###############
## 50Hz Data ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_00
TIMESTAMP_RUN3=2022_10_01-00_00_01
TARGETDIR=__default
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## 100Hz Data ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_02
TIMESTAMP_RUN3=2022_10_01-00_00_03
TARGETDIR=100Hz
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## 20Hz Data ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_04
TIMESTAMP_RUN3=2022_10_01-00_00_05
TARGETDIR=20Hz
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## 10Hz Data ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_06
TIMESTAMP_RUN3=2022_10_01-00_00_07
TARGETDIR=10Hz
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## Clustering only ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_08
TIMESTAMP_RUN3=2022_10_01-00_00_09
TARGETDIR=clustering
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## RADAR only ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_10
TIMESTAMP_RUN3=2022_10_01-00_00_11
TARGETDIR=radar
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## bounds buffer -0.75 m ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_12
TIMESTAMP_RUN3=2022_10_01-00_00_13
TARGETDIR=boundbuffer075
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## bounds buffer 0.0 m ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_14
TIMESTAMP_RUN3=2022_10_01-00_00_15
TARGETDIR=boundbuffer0
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## max_match_dist_m": 1.0 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_16
TIMESTAMP_RUN3=2022_10_01-00_00_17
TARGETDIR=matchdist1
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## max_match_dist_m": 2.0 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_18
TIMESTAMP_RUN3=2022_10_01-00_00_19
TARGETDIR=matchdist2
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## max_match_dist_m": 7.0 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_20
TIMESTAMP_RUN3=2022_10_01-00_00_21
TARGETDIR=matchdist7
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## n_max_counter": 6 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_22
TIMESTAMP_RUN3=2022_10_01-00_00_23
TARGETDIR=maxctr6
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## n_max_counter": 12 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_24
TIMESTAMP_RUN3=2022_10_01-00_00_25
TARGETDIR=maxctr12
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## n_max_counter": 37 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_46
TIMESTAMP_RUN3=2022_10_01-00_00_47
TARGETDIR=maxctr37
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## filter_frequency: 10 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_26
TIMESTAMP_RUN3=2022_10_01-00_00_27
TARGETDIR=filterfreq10
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## filter_frequency: 50 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_28
TIMESTAMP_RUN3=2022_10_01-00_00_29
TARGETDIR=filterfreq50
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## filter_frequency: 200 ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_30
TIMESTAMP_RUN3=2022_10_01-00_00_31
TARGETDIR=filterfreq200
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## overlap_check = {"overlap_dist_m": 1.7} ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_32
TIMESTAMP_RUN3=2022_10_01-00_00_33
TARGETDIR=overlap17
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## overlap_check = {"overlap_dist_m": 3.4} ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_34
TIMESTAMP_RUN3=2022_10_01-00_00_35
TARGETDIR=overlap34
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## overlap_check = {"overlap_dist_m": 7.2} ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_36
TIMESTAMP_RUN3=2022_10_01-00_00_37
TARGETDIR=overlap72
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR


###############
## lidar_cluster "std_dev": [0.1, 0.1, 0.35]##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_38
TIMESTAMP_RUN3=2022_10_01-00_00_39
TARGETDIR=clustering_xy_01
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## lidar_cluster "std_dev": [0.5, 0.5, 0.35]##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_40
TIMESTAMP_RUN3=2022_10_01-00_00_41
TARGETDIR=clustering_xy_05
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## radar "std_dev": "std_dev": [3.0, 3.0, 0.35, 1.0] ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_42
TIMESTAMP_RUN3=2022_10_01-00_00_43
TARGETDIR=radar_v_1
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR

###############
## radar "std_dev": "std_dev": [3.0, 3.0, 0.35, 2.0] ##
###############
TIMESTAMP_RUN2=2022_10_01-00_00_44
TIMESTAMP_RUN3=2022_10_01-00_00_45
TARGETDIR=radar_v_2
$REPO_DIR/evaluation/evaluation_single.sh $TIMESTAMP_RUN2 $TIMESTAMP_RUN3 $TARGETDIR
