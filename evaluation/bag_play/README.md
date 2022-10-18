# ROSBAG replay
Description how to reproduce perception data from semifinal and final from AC@CES. All replayed logs are also available to download ([Link](https://doi.org/10.5281/zenodo.7220695), folder `logs`).

## Notes
- Used Image: `tumiac/base:0.0.0` (available on docker hub)
## Get source data
Download data from [here](https://doi.org/10.5281/zenodo.7220695) and paste the folder `tracking_input` into `data`.

## Run in replay mode
The procedure is shown with docker compose.
### 1. Setup parameters
Specify the following parameters in the [.env](.env)-file:
1. Tag of tracking node (build docker image before, see [README](../../README.md).
```
tracking=<tag>  # e.g. tracking=0.0.1
```
2. Time source of the node (ROS2 native parameter). Set true to listen to the published sim time from the `play`-nodes:
```
Simtime=<Boolean>  # e.g. Simtime=True
```
3. Cycle frequency of tracking node. We recommend 50Hz:
```
Frequency=<(float, int)>  # e.g. Frequency=50.0
```
4. We recommend to set a read ahead value, so the topics are published at the correct time:
```
Readahead=<int>  # e.g. Readahead=100000
```
5. Delay to state the play service. A delay of 2.0s is recommended for the play service, so the tracking node is fully initialized before the replay starts:
```
Delay=<float>  # e.g. Delay=2.0
```

### 2. Run the service
Start the tracking node and the ros2-replay with the following commands:
```
cd evaluation/bag_play  # relative path handling
docker compose -f replay.yml --env-file .env up mod_tracking2 play2 # run tracking with semifinal
docker compose -f replay.yml --env-file .env up mod_tracking3 play3 # run tracking with final
```
Notes:
- Don't start them all at the same time, otherwise, the created logs are stored in the same folder (same timestamp).
- The containers have to be stopped manually. Press `ctrl + c` when `bag_play_play2_1 exited with code 0` appears in the terminal.
- Duration: semifinal: 18min, final: 23min

### 3. Get log data
The logs of the module can be found in [logs-directory](../../tracking/tracking/logs/) sorted in subfolders by date and timestamp: `%Y_%m_%d/%H_%M_%S`. They can be analyzed by running the visualization (see [README](../README.md)).
