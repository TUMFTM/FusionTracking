"""Show Tracking Logs."""
import os
import sys
import argparse

import matplotlib.pyplot as plt

import cv2
import imageio.v2 as imageio
import tqdm
import datetime

import numpy as np

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(REPO_PATH)

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
EGO_COL = TUM_BLUE
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
OBJ_COL = TUM_ORAN
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2.0
FONTSIZE = 10
IMAGE_PATH = "out"

RUN3_FINAL = "RUN3_final_2022_10_01-00_00_01"
RESULTS_PATH = "evaluation/results/__default"
OBJ_ID = str(345)  # fastest overtake

from tools.visualize_logfiles import ViszTracking


class ViszSample(ViszTracking):
    """Class to visualize sample of tracked object with video as output.

    Args:
        ViszTracking: Inheritance
    """

    def __init__(self, args):
        """Initialize visualize sample class."""
        # add args that are not used (for parent class)
        args.n_obs = np.inf
        args.states = False
        args.ego_state = False
        args.filter = False
        args.mis_num_obj = False

        args.show_sample = True
        args.show_filtered = False
        args.is_main = False

        self.results_save_path = args.results_save_path

        self.args = args

        super().__init__(args=args)

        # set private members

        # previous trajectory
        self._ViszTracking__bool_prev_traj = True

        # future trajectory
        self._ViszTracking__bool_fut_traj = False

        # matched Detections
        self._ViszTracking__bool_sensor_detection = True

        # Filter possibilities
        self._ViszTracking__bool_filter = False

        # Filter-Prior
        self._ViszTracking__bool_filter_prior = False

        # Filter-Delay
        self._ViszTracking__bool_filter_delay = False

        # Filter-Posterior
        self._ViszTracking__bool_filter_post = False

        # Prediction
        self._ViszTracking__bool_predict = False

        # unmatched Detections
        self._ViszTracking__bool_unmatched = True

        self._ViszTracking__bool_raw_data = True

        if os.path.exists(IMAGE_PATH):
            _ = [
                os.remove(os.path.join(IMAGE_PATH, kk)) for kk in os.listdir(IMAGE_PATH)
            ]
        else:
            os.makedirs(IMAGE_PATH)

        self.obj_id = args.obj_id
        self.obj_idx = list(self.obj_items[self.obj_id]["object_states"].keys())

        self.video_save_name = (
            args.save_file_name
            + "_"
            + self.obj_id
            + "_"
            + datetime.datetime.now().__format__("%Y-%m-%d-%H-%M-%S")
            + ".mp4"
        )

    def iter_samples(self):
        """Iterate through samples of given object.

        Returns:
            freq (float): Frequency (mean) in Hz
            video_name (str): Name of video
        """
        vis_range = 10
        frame_values = list(range(self.obj_idx[0], self.obj_idx[-1]))

        freq = len(frame_values) / (
            (
                self.obj_items[self.obj_id]["t_dict"][frame_values[-1]]
                - self.obj_items[self.obj_id]["t_dict"][frame_values[0]]
            )
            / 1e9
        )

        self.ego_col = EGO_COL
        self.color_list = [BOUND_COL for _ in range(len(self.color_list))]
        self.color_list[int(int(self.obj_id) % len(self.color_list))] = OBJ_COL
        self.track_color = {"track_bounds": BOUND_COL, "pit_bounds": BOUND_COL}
        self.raw_data_color = "k"

        print("Creating plots {}...".format(len(frame_values)))

        for j, f_val in tqdm.tqdm(enumerate(frame_values)):

            self.__call__(frame_value=f_val, is_iterative=True)

            xx, yy = self.obj_items["ego"]["object_states"][f_val][:2]
            o_xx, o_yy = self.obj_items[self.obj_id]["object_states"][f_val][:2]

            self._ax.set_xlim(
                [
                    np.min([xx, o_xx]) - vis_range,
                    np.max([xx, o_xx]) + vis_range,
                ]
            )
            self._ax.set_ylim(
                [
                    np.min([yy, o_yy]) - vis_range,
                    np.max([yy, o_yy]) + vis_range,
                ]
            )

            self._ax.set_xlabel("$x_{\mathrm{glob}}$ in m")
            self._ax.set_ylabel("$y_{\mathrm{glob}}$ in m")
            self._ax.grid(True)

            _ = [txt.set_visible(False) for txt in self.fig.texts]
            # lg = self._ax.get_legend()
            # _ = lg.set_visible(False)

            plt.show(block=False)
            # plt.pause(1.0 / 10.0 / freq)

            extent = self._ax.get_window_extent().transformed(
                self.fig.dpi_scale_trans.inverted()
            )
            plt.savefig("out/" + str(j).zfill(4) + ".png", bbox_inches=extent)
            if j > 100 and self.args.debug:
                print("debug, aborting after 100 images")
                break

        return freq, self.video_save_name


def create_video(freq: float = 10.0, video_name: str = None):
    """Create .mp4-video from pngs.

    Args:
        freq (float, optional): Video frequency in Hz. Defaults to 10.0.
        video_name (str, optional): Name of video. Defaults to None.
    """
    if video_name is None:
        video_name = datetime.datetime.now().__format__("%Y-%m-%d-%H-%M-%S") + ".mp4"

    images = [img for img in os.listdir(IMAGE_PATH) if img.endswith(".png")]

    images_sort = [i.zfill(8) + i for i in images]
    images_sort.sort()
    images = [i.split("g")[1] + "g" for i in images_sort]
    frame = cv2.imread(os.path.join(IMAGE_PATH, images[1]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, freq, (width, height))

    i = 0
    print("Creating video ..")
    for image in tqdm.tqdm(images):
        i += 1
        img = cv2.imread(os.path.join(IMAGE_PATH, image))
        cv2.imshow("Creating video...", img)
        cv2.waitKey(1)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def create_gif(freq: float, gif_duration_s: float = 5.0, video_name: str = None):
    """Create gif from mp4 video.

    Args:
        freq (float): Frequency of the video in Hz.
        gif_duration_s (int, optional): Duration of gif in seconds. Defaults to 5.
        video_name (_type_, optional): name of output gif, has to end with '.gif'. Defaults to None.

    Raises:
        ValueError: Raised if gif name has wrong ending.
    """
    if video_name is None:
        video_name = datetime.datetime.now().__format__("%Y-%m-%d-%H-%M-%S") + ".gif"
    elif not video_name.endswith(".gif"):
        raise ValueError(
            "Invalide gif name {}, has to end with '.gif'".format(video_name)
        )

    filenames = [img for img in os.listdir(IMAGE_PATH) if img.endswith(".png")]
    _ = filenames.sort()

    max_images = int(gif_duration_s * freq)
    chunk_lists = [
        filenames[x : x + max_images] for x in range(0, len(filenames), max_images)
    ]

    kargs = {"duration": 1 / freq}
    for j, chunk_list in enumerate(chunk_lists):
        with imageio.get_writer(
            video_name.replace(".gif", "_{}.gif".format(j)), mode="I", **kargs
        ) as writer:
            for filename in chunk_list:
                image = imageio.imread(os.path.join(IMAGE_PATH, filename))
                writer.append_data(image)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--save_file_name", type=str, default=RUN3_FINAL)
    parser.add_argument("--results_save_path", type=str, default=RESULTS_PATH)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--obj_id", type=str, default=OBJ_ID)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gif", default=False, action="store_true")
    args = parser.parse_args()

    eval = ViszSample(args)
    freq, video_name = eval.iter_samples()

    if args.gif:
        create_gif(freq=freq, video_name=video_name.replace(".mp4", ".gif"))
    else:
        create_video(freq=freq, video_name=video_name)
