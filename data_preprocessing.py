import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def KEYPOINTS_RANGE(include_face_keypoints):
    ALL_KEYPOINTS_RANGE = (0, 17)
    ALL_BUT_FACE_KEYPOINTS_RANGE = (5, 17)
    if include_face_keypoints:
        return ALL_KEYPOINTS_RANGE
    else:
        return ALL_BUT_FACE_KEYPOINTS_RANGE


def get_tracked_ids(frames):
    track_ids = set()
    for frame in frames:
        for person in frame:
            if person.get("track_id"):
                track_ids.add(person["track_id"])
    return track_ids


def prepare_time_series(frames, include_face_keypoints=True, fill_missing=np.nan):
    columns = [
        "track_id",
        "frame_count",
        "conf",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "box_width",
        "box_height",
    ]
    keypoints_range = KEYPOINTS_RANGE(include_face_keypoints)
    for i in range(*keypoints_range):
        columns.append(f"kp{i+1}_x")
        columns.append(f"kp{i+1}_y")
        columns.append(f"kp{i+1}_conf")

    track_ids = get_tracked_ids(frames)
    time_series = {}
    for person_id in track_ids:
        # Create a (likely sparse) time series for each tracked person.
        df = pd.DataFrame(
            index=np.arange(len(frames)), columns=np.arange(len(columns)), dtype="float"
        )

        for i, frame in enumerate(frames):
            matching_persons = [person for person in frame if person.get("track_id") == person_id]
            if len(matching_persons) == 1:
                person = matching_persons[0]

                box = person["box"]
                xs = np.array(person["keypoints"]["x"][keypoints_range[0] : keypoints_range[1]])
                ys = np.array(person["keypoints"]["y"][keypoints_range[0] : keypoints_range[1]])
                confidences = person["keypoints"]["visible"][
                    keypoints_range[0] : keypoints_range[1]
                ]

                box_width = box["x2"] - box["x1"]
                box_height = box["y2"] - box["y1"]

                points = np.array([xs, ys, confidences]).T
                points = points.flatten()

                df.iloc[i] = [
                    person_id,
                    i + 1,  # frame count
                    person["confidence"],
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                    box_width,
                    box_height,
                    *points,
                ]
        df.columns = columns
        df = df[
            df.first_valid_index() : df.last_valid_index() + 1
        ]  # Remove leading and trailing NaNs.
        df.replace(
            0, fill_missing, inplace=True
        )  # 0 values are missing values. Replace them with the given value.
        time_series[person_id] = df
    return time_series


def split_series_at_missing_frames(ts):
    """
    Given the input dataframe, this function splits it into multiple dataframes each containing a continuous sequence of not-nan rows.
    """
    columns_regex = ".*_(x|y).*"  # all x and y columns
    groups = ts.filter(regex=columns_regex).isna().all(axis=1).cumsum()

    splits = [df.dropna(how="all") for _, df in ts.groupby(groups)]
    splits = [
        df for df in splits if len(df) > 0 and all(df.isna().sum() < len(df))
    ]  # remove empty dataframes, and those with only NaNs for any column
    return splits


def fill_missing_values_linear(ts):
    return ts.interpolate(method="linear", limit_direction="both")


def filter_moving_average(ts, window):
    x_columns = ts.columns.str.endswith("_x")
    y_columns = ts.columns.str.endswith("_y")
    columns = np.concatenate(
        [
            ts.columns[x_columns + y_columns],
            ["conf", "box_x1", "box_y1", "box_x2", "box_y2", "box_width", "box_height"],
        ]
    )
    ts = ts.copy()
    ts[columns] = ts[columns].rolling(window=window).mean()
    ts.dropna(how="all", subset=columns, inplace=True)
    return ts


def generate_track_ids(splitted_time_series):
    flattened_time_series = {}
    last_track_id = max(splitted_time_series.keys()) + 1
    for track_id, splits in splitted_time_series.items():
        first_ts = splits[0]
        flattened_time_series[track_id] = first_ts
        rest_splits = splits[1:]
        for ts in rest_splits:
            ts["track_id"] = last_track_id
            flattened_time_series[last_track_id] = ts
            last_track_id += 1
    return flattened_time_series


def filter_video(video_time_series, moving_avg_window):
    filtered_time_series = {}
    for track_id, ts in video_time_series.items():
        # TODO: remove splitting from this function.
        ts_splitted = split_series_at_missing_frames(ts)
        ts_filtered = []
        for ts_split in ts_splitted:
            ts_split = fill_missing_values_linear(ts_split)
            ts_split = filter_moving_average(ts_split, window=moving_avg_window)
            ts_filtered.append(ts_split)
        if len(ts_filtered) > 0:  # remove people with no valid frames
            filtered_time_series[track_id] = ts_filtered
    filtered_time_series = generate_track_ids(filtered_time_series)
    return filtered_time_series


def scale_points_to_bounding_box(ts):
    ts_normalized = ts.copy()

    x_columns = ts.columns[ts.columns.str.endswith("_x")]
    ts_normalized[x_columns] = (
        (ts_normalized[x_columns].sub(ts_normalized["box_x1"], axis=0))
        .div(ts_normalized["box_width"], axis=0)
        .clip(0, 1)
    )

    y_columns = ts.columns[ts.columns.str.endswith("_y")]
    ts_normalized[y_columns] = (
        (ts_normalized[y_columns].sub(ts_normalized["box_y1"], axis=0))
        .div(ts_normalized["box_height"], axis=0)
        .clip(0, 1)
    )

    return ts_normalized


def scale_video_points_to_bounding_box(video_time_series):
    video_normalized = {}
    for track_id, ts in video_time_series.items():
        ts_normalized = scale_points_to_bounding_box(ts)
        video_normalized[track_id] = ts_normalized
    return video_normalized


def get_video_avg_variations(video_persons):
    video_stats = []

    for track_id, series in video_persons.items():
        # Input shape: (kp1_x, kp1_y, kp2_x, kp2_y, ..., kp17_x, kp17_y)
        # Output shape: (d1, d2, ..., d17) where d_i = sqrt(kpi_x**2 + kpi_y**2)
        diffs = series.diff().filter(regex="kp.*_(x|y)")
        diffs = diffs.dropna(how="all")
        diffs = diffs.values.astype("float")
        n_dims = int(diffs.shape[1] / 2)
        for i in range(0, n_dims, 2):
            diffs[:, i] = np.sqrt(diffs[:, i] ** 2 + diffs[:, i + 1] ** 2)
        diffs = diffs[:, 0::2]

        # Meaning: (conf_i + conf_{i+1}) / 2
        confs_rolling_avg = series.filter(regex=("kp.*_conf")).rolling(2).sum() / 2
        confs_rolling_avg = confs_rolling_avg.dropna(how="all")
        confs_rolling_avg = confs_rolling_avg.values.astype("float")

        if len(diffs) == 0:
            continue

        tot_variation = (diffs * confs_rolling_avg).sum() / len(diffs)
        video_stats.append([track_id, len(diffs), tot_variation])

    video_stats = pd.DataFrame(video_stats, columns=["track_id", "notna_count", "tot_variation"])
    return video_stats


def split_series_into_windows(series, window_size):
    """Sliding window approach with 50% overlap."""
    windows = [series[i : i + window_size] for i in range(0, len(series), int(window_size * 0.5))]
    windows = [window for window in windows if len(window) == window_size]
    return windows


def WINDOW_COLUMNS(window_size, include_face_keypoints):
    keypoints_range = KEYPOINTS_RANGE(include_face_keypoints)

    columns = ["track_id", "conf_avg", "conf_std", "first_frame", "last_frame", "window_size"]
    for i in range(*keypoints_range):
        columns.append(f"kp{i+1}_x_avg")
        columns.append(f"kp{i+1}_y_avg")
    for i in range(*keypoints_range):
        columns.append(f"kp{i+1}_x_std")
        columns.append(f"kp{i+1}_y_std")
    for i in range(window_size):
        columns.append(f"f{i+1}_box_conf")
        columns.append(f"f{i+1}_box_width")
        columns.append(f"f{i+1}_box_height")
        for j in range(*keypoints_range):
            columns.append(f"f{i+1}_kp{j+1}_x")
        for j in range(*keypoints_range):
            columns.append(f"f{i+1}_kp{j+1}_y")
        for j in range(*keypoints_range):
            columns.append(f"f{i+1}_kp{j+1}_conf")
    return columns


def extract_features_from_windows(windows, window_size, include_face_keypoints):
    n_windows = len(windows)

    columns = WINDOW_COLUMNS(window_size, include_face_keypoints)
    keypoints_range = KEYPOINTS_RANGE(include_face_keypoints)

    prepared_chunks = pd.DataFrame(index=np.arange(n_windows), columns=columns)
    for i, chunk in enumerate(windows):
        track_id = chunk["track_id"].iloc[0]
        conf_avg = chunk["conf"].mean()
        conf_std = chunk["conf"].std()
        first_frame = chunk["frame_count"].iloc[0]
        last_frame = chunk["frame_count"].iloc[-1]
        window_actual_size = chunk["frame_count"].max() - chunk["frame_count"].min()

        x_columns = [f"kp{i+1}_x" for i in range(*keypoints_range)]
        y_columns = [f"kp{i+1}_y" for i in range(*keypoints_range)]
        conf_columns = [f"kp{i+1}_conf" for i in range(*keypoints_range)]

        prepared_chunks.iloc[i] = np.concatenate(
            (
                [track_id, conf_avg, conf_std, first_frame, last_frame, window_actual_size],
                chunk.filter(regex="kp.*_(x|y)").mean().values,
                chunk.filter(regex="kp.*_(x|y)").std().values,
                chunk[
                    ["conf", "box_width", "box_height"] + x_columns + y_columns + conf_columns
                ].values.flatten(),
            )
        )
    return prepared_chunks


def split_persons_at_missing_frames(video_time_series):
    splitted_time_series = {}
    for track_id, ts in video_time_series.items():
        ts_splitted = split_series_at_missing_frames(ts)
        if len(ts_splitted) > 0:
            splitted_time_series[track_id] = ts_splitted
    splitted_time_series = generate_track_ids(splitted_time_series)
    return splitted_time_series


def filter_valid_persons(all_time_series, all_videos_stats, min_frames: int, min_variation: int):
    all_stats = []
    for video_name, stats in all_videos_stats.items():
        df = stats.copy()
        df["video_name"] = video_name
        all_stats.append(df)
    all_stats = pd.concat(all_stats)

    all_stats = all_stats.copy()[all_stats["notna_count"] >= min_frames]
    scaler = MinMaxScaler()
    all_stats["tot_variation_normalized"] = scaler.fit_transform(all_stats[["tot_variation"]])

    valid_stats = all_stats[all_stats["tot_variation_normalized"] >= min_variation]
    all_filtered_videos = {}
    for video_name, time_series in all_time_series.items():
        valid_ids = valid_stats[valid_stats["video_name"] == video_name]["track_id"].values
        all_filtered_videos[video_name] = {
            track_id: time_series[track_id] for track_id in valid_ids
        }
    return scaler, all_filtered_videos


def split_series_into_windows_size30_overlap50_step2(series):
    windows = [
        window for window in split_series_into_windows(series[0:-1:2], 30) if len(window) == 30
    ]
    windows.extend(
        [window for window in split_series_into_windows(series[1:-1:2], 30) if len(window) == 30]
    )
    return windows


def convert_video_to_dataset_format(
    name, label, time_series, include_face_keypoints, sliding_window
):
    if sliding_window not in ["size30_overlap50", "size30_overlap50_step2"]:
        raise ValueError(f"Unsupported sliding window: {sliding_window}")

    all_chunks = []
    for series in time_series.values():
        if sliding_window == "size30_overlap50":
            chunks = split_series_into_windows(series, 30)
        elif sliding_window == "size30_overlap50_step2":
            chunks = split_series_into_windows_size30_overlap50_step2(series)
        prepared_chunks = extract_features_from_windows(chunks, 30, include_face_keypoints)
        all_chunks.append(prepared_chunks)
    prepared_data = pd.concat(all_chunks)
    prepared_data["name"] = name
    prepared_data["label"] = label
    return prepared_data
