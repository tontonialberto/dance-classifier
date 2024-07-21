import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys


def draw_skeleton(img, points, confs, origin_x, origin_y, scale_x, scale_y):
    line_pairs = [
        [5, 6],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [5, 11],
        [6, 12],
        [11, 12],
        [11, 13],
        [13, 15],
        [14, 16],
        [12, 14],
    ]

    scaled_points = points * [scale_x, scale_y]
    scaled_points = scaled_points + [origin_x, origin_y]

    for point, scaled_point in zip(points, scaled_points):
        x, y = int(scaled_point[0]), int(scaled_point[1])
        cv2.circle(img, (x, y), 0, (255, 255, 0))
    for pair in line_pairs:
        conf1 = confs[pair[0]]
        conf2 = confs[pair[1]]
        if conf1 > 0.5 and conf2 > 0.5:
            point1 = scaled_points[pair[0]]
            p1_x, p1_y = int(point1[0]), int(point1[1])
            point2 = scaled_points[pair[1]]
            p2_x, p2_y = int(point2[0]), int(point2[1])
            cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0))


def transform_points(points, confs, box, include_face_poses, scale_to_bounding_box):
    if not include_face_poses:
        # include fictitious face keypoints
        points = np.concatenate([np.zeros((5, 2)), points])
        confs = np.concatenate([np.zeros(5), confs])

    if scale_to_bounding_box:
        box_width = box["x2"] - box["x1"]
        box_height = box["y2"] - box["y1"]
        points = (points - [box["x1"], box["y1"]]) / [box_width, box_height]

    return points, confs


def draw_frame(mode, img, track_ids, boxes, all_points, all_confs, origin):
    x = 0
    y = 0
    scale_width = 72
    scale_height = 128
    for id, box, points, confs in zip(track_ids, boxes, all_points, all_confs):
        if mode == "poses":
            draw_skeleton(img, points, confs, origin[0], origin[1], 1, 1)
            cv2.putText(
                img,
                str(id),
                (int(box["x1"] + origin[0] + 10), int(box["y1"] + origin[1] + 10)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 0),
            )
        elif mode == "boxes":
            draw_skeleton(
                img, points, confs, x + origin[0], y + origin[1], scale_width, scale_height
            )
            cv2.putText(
                img,
                str(id),
                (int(x + origin[0] + (scale_width / 2)), int(origin[1] + scale_height)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 0),
            )
            x += scale_width
    return img


def draw_video_frame(
    mode, image, frame_index, frame, origin, scale_to_bounding_box, include_face_poses
):
    track_ids = []
    boxes = []
    all_points = []
    all_confs = []
    for person in frame:
        if person.get("track_id"):
            box = person["box"]
            box_width = box["x2"] - box["x1"]
            box_height = box["y2"] - box["y1"]
            if box_width == 0 or box_height == 0:
                print(
                    f"WARNING: box width or height is 0. Frame {frame_index}, track_id {person.get('track_id')}"
                )
                continue

            xs = person["keypoints"]["x"]
            ys = person["keypoints"]["y"]
            confs = person["keypoints"]["visible"]

            points, confs = transform_points(
                np.array([xs, ys]).T, confs, box, include_face_poses, scale_to_bounding_box
            )
            all_points.append(points)
            all_confs.append(confs)
            boxes.append(box)
            track_ids.append(person.get("track_id"))

    draw_frame(mode, image, track_ids, boxes, all_points, all_confs, origin)


def draw_frame_of_multiple_videos(image, videos, frame_index, origins):
    for i, video in enumerate(videos):
        video_name = str(video["path"])
        cv2.putText(
            image,
            video_name,
            (origins[i][0], origins[i][1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            color=(255, 255, 255),
        )
        frame = video["frames"][frame_index]
        scale_to_bounding_box = video["scale"]
        include_face_poses = video["include_face"]
        display_poses = video["display_poses"]
        if display_poses:
            draw_video_frame(
                "poses",
                image,
                frame_index,
                frame,
                origins[i],
                scale_to_bounding_box=False,
                include_face_poses=include_face_poses,
            )
        elif video["display_boxes"]:
            draw_video_frame(
                "boxes",
                image,
                frame_index,
                frame,
                origins[i],
                scale_to_bounding_box=scale_to_bounding_box,
                include_face_poses=include_face_poses,
            )


def display_videos(videos):
    n_frames = len(videos[0]["frames"])
    all_videos_equal_length = all(n_frames == len(video["frames"]) for video in videos)

    if not all_videos_equal_length:
        raise ValueError("All videos must have the same number of frames")

    for i in range(n_frames):
        image = np.zeros((720, 1280, 3), np.uint8)
        cv2.line(image, (640, 0), (640, 1280), (255, 255, 255))
        cv2.line(image, (0, 360), (1280, 360), (255, 255, 255))
        cv2.putText(
            image,
            "Frame " + str(i),
            (0, 355),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            color=(255, 255, 255),
        )
        draw_frame_of_multiple_videos(image, videos, i, [(0, 0), (640, 0), (0, 360), (640, 360)])
        cv2.imshow("video", image)
        if (cv2.waitKey(25) & 0xFF) == ord("q"):
            print("frame", i)
            cv2.destroyAllWindows()
            return

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    for i, video in enumerate(config["videos"]):
        assert "path" in video, f"Video path not found in config - at index {i}"
        assert "scale" in video, f"Video scale not found in config - at index {i}"
        assert "include_face" in video, f"Video include_face not found in config - at index {i}"
        assert "display_poses" in video, f"Video display_poses not found in config - at index {i}"
        assert Path(
            video["path"]
        ).exists(), f"Video path {video['path']} does not exist - at index {i}"
    return config


def main(config_path):
    videos = load_config(config_path)["videos"]

    for video in videos:
        with open(video["path"], "r") as f:
            annotations = json.load(f)
            video["frames"] = annotations["results"]

    display_videos(videos)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <config_path>")
        sys.exit(1)
    main(sys.argv[1])
