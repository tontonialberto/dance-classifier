import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import (
    KEYPOINTS_RANGE,
    convert_video_to_dataset_format,
    filter_video,
    get_video_avg_variations,
    prepare_time_series,
    scale_video_points_to_bounding_box,
    split_persons_at_missing_frames,
)


def preprocess_videos(
    instances_paths: list[Path],
    include_face_keypoints: bool,
    fill_missing_ts_value,
    impute_and_filter: dict | None = {"moving_avg_window": 5},
):
    if impute_and_filter is not None and "moving_avg_window" not in impute_and_filter:
        raise ValueError("impute_and_filter must contain a 'moving_avg_window' key.")

    result = {}
    total_time = 0
    for i, instance_path in enumerate(instances_paths):
        instance_start_time = time.time()

        progress = f"{(i + 1)} / {len(instances_paths)}"
        print(f"({progress}) Loading TS from {instance_path}...")
        instance = load_json(instance_path)

        start_time = time.time()
        time_series = prepare_time_series(
            instance["results"], include_face_keypoints, fill_missing_ts_value
        )
        time_series = split_persons_at_missing_frames(time_series)
        print(f"({progress}) TS loaded in {time.time() - start_time:.1f}s. Filtering...")

        if impute_and_filter is None:
            time_series_filtered = time_series
            print(f"({progress}) No imputation or filtering applied.")
        else:
            moving_avg_window = impute_and_filter["moving_avg_window"]
            start_time = time.time()
            time_series_filtered = filter_video(time_series, moving_avg_window)
            print(f"({progress}) TS filtered in {time.time() - start_time:.1f}s. Scaling...")

        start_time = time.time()
        time_series_scaled = scale_video_points_to_bounding_box(time_series_filtered)
        print(f"({progress}) TS scaled in {time.time() - start_time:.1f}s. Computing stats...")

        start_time = time.time()
        time_series_stats = get_video_avg_variations(time_series_scaled)
        print(f"({progress}) Stats computed in {time.time() - start_time:.1f}s.")

        result[instance_path.name] = {
            "time_series": time_series_scaled,
            "stats": time_series_stats,
        }

        total_time += time.time() - instance_start_time

    print(f"Done. Total time: {total_time:.1f}s.")
    return result


def load_json(path):
    with open(path, "r") as f:
        video = json.load(f)
    return video


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def save_video_time_series_to_json(
    original_instance_path, output_path, video_time_series, include_face_keypoints
):
    Path(output_path).mkdir(exist_ok=True)

    keypoints_range = KEYPOINTS_RANGE(include_face_keypoints)

    original_instance_json = load_json(original_instance_path)
    n_frames = len(original_instance_json["results"])

    filtered_instance_json = {
        "name": original_instance_json["name"],
        "label": original_instance_json["label"],
        "results": [],
    }
    frames = {i + 1: [] for i in range(n_frames)}
    for track_id, ts in video_time_series.items():
        for frame in ts.to_dict(orient="records"):
            track_id = frame["track_id"]
            xs = [frame[f"kp{i+1}_x"] for i in range(*keypoints_range)]
            ys = [frame[f"kp{i+1}_y"] for i in range(*keypoints_range)]
            confs = [frame[f"kp{i+1}_conf"] for i in range(*keypoints_range)]
            person = {
                "name": "person",
                "class": 0,
                "confidence": frame["conf"],
                "box": {
                    "x1": frame["box_x1"],
                    "y1": frame["box_y1"],
                    "x2": frame["box_x2"],
                    "y2": frame["box_y2"],
                },
                "track_id": track_id,
                "keypoints": {
                    "x": xs,
                    "y": ys,
                    "visible": confs,
                },
            }
            frame_count = frame["frame_count"]
            frames[frame_count].append(person)
    filtered_instance_json["results"] = list(frames.values())

    output_file_path = Path(output_path) / Path(original_instance_path).name
    save_json(output_file_path, filtered_instance_json)
    print(f"Result saved to {output_file_path}")


def display_classification_metrics(y_true, y_pred):
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall: ", recall_score(y_true, y_pred))
    print("F1 Score: ", f1_score(y_true, y_pred))


def display_classifiers_performances(models, X, y, cv=10):
    for name, clf in models:
        start_time = time.time()
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        delta = time.time() - start_time
        print(f"### {name} ###")
        print(f"Time taken to complete CV {cv}fold: {delta:.1f} seconds.")
        print("----")
        display_classification_metrics(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d")
        plt.show()
        print()
        print()


def save_dataset(all_videos, output_path, include_face_keypoints, sliding_window):
    Path(output_path).mkdir(exist_ok=True)
    i = 0
    for name, time_series in all_videos.items():
        progress = f"{(i + 1)} / {len(all_videos)}"
        print(f"{progress} Processing {name}...")
        start_time = time.time()
        label = name.split("_")[0]
        name = "_".join(name.split("_")[1:])
        prepared_data = convert_video_to_dataset_format(
            name, label, time_series, include_face_keypoints, sliding_window
        )
        output_filepath = Path(output_path) / f"{label}_{name}.csv"
        prepared_data.to_csv(output_filepath, index=False)
        print(f"{progress} Done in {time.time() - start_time:.1f}s. Saved to {output_filepath}")
        i += 1
