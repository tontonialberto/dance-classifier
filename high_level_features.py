import numpy as np
import pandas as pd
from data_preprocessing import KEYPOINTS_RANGE


def compute_frames_velocities_and_accelerations(X, window_size=30, include_face_keypoints=False):
    deltas = []

    keypoints_range = KEYPOINTS_RANGE(include_face_keypoints)
    for keypoint in range(*keypoints_range):
        x_colnames = [
            f"f{frame_count}_kp{keypoint+1}_x" for frame_count in range(1, window_size + 1)
        ]
        x_delta_colnames = [f"vel{i}_kp{keypoint+1}_x" for i in range(1, window_size)]
        x_acc_colnames = [f"acc{i}_kp{keypoint+1}_x" for i in range(1, window_size - 1)]

        y_colnames = [
            f"f{frame_count}_kp{keypoint+1}_y" for frame_count in range(1, window_size + 1)
        ]
        y_delta_colnames = [f"vel{i}_kp{keypoint+1}_y" for i in range(1, window_size)]
        y_acc_colnames = [f"acc{i}_kp{keypoint+1}_y" for i in range(1, window_size - 1)]

        x_deltas = (
            X[x_colnames].diff(axis=1).dropna(how="all", axis=1).set_axis(x_delta_colnames, axis=1)
        )
        y_deltas = (
            X[y_colnames].diff(axis=1).dropna(how="all", axis=1).set_axis(y_delta_colnames, axis=1)
        )

        deltas.extend(
            [
                x_deltas,
                x_deltas.diff(axis=1).dropna(how="all", axis=1).set_axis(x_acc_colnames, axis=1),
                y_deltas,
                y_deltas.diff(axis=1).dropna(how="all", axis=1).set_axis(y_acc_colnames, axis=1),
            ]
        )

    return pd.concat(deltas, axis=1)


def compute_points_distances(X, window_size=30):

    adjacencies = [
        [6, 7],
        [6, 8],
        [8, 10],
        [7, 9],
        [9, 11],
        [6, 12],
        [7, 13],
        [12, 13],
        [12, 14],
        [14, 16],
        [15, 17],
        [13, 15],
    ]

    colnames = []
    columns = []
    for frame_count in range(1, window_size + 1):
        for i, j in adjacencies:
            colnames.append(f"f{frame_count}_eucliddist_{i}_{j}")

            columns.append(
                np.sqrt(
                    (X[f"f{frame_count}_kp{i}_x"] - X[f"f{frame_count}_kp{j}_x"]) ** 2
                    + (X[f"f{frame_count}_kp{i}_y"] - X[f"f{frame_count}_kp{j}_y"]) ** 2
                ).values
            )
    return pd.DataFrame(columns).T.set_axis(colnames, axis=1)


def compute_angle_3points(X, points_indexes, window_size=30):
    angle_columns = []
    a, b, c = points_indexes
    angle_colnames = [f"f{i}_angle_{a}_{b}_{c}" for i in range(1, window_size + 1)]

    for i in range(1, window_size + 1):
        a_x = X[f"f{i}_kp{a}_x"]
        a_y = X[f"f{i}_kp{a}_y"]
        b_x = X[f"f{i}_kp{b}_x"]
        b_y = X[f"f{i}_kp{b}_y"]
        c_x = X[f"f{i}_kp{c}_x"]
        c_y = X[f"f{i}_kp{c}_y"]

        AB = np.array([a_x - b_x, a_y - b_y])
        CB = np.array([c_x - b_x, c_y - b_y])

        # Compute the dot product of AB and CB for each data point.
        dot_product = (AB * CB).sum(axis=0)

        magnitude_AB = np.linalg.norm(AB.T, axis=1)
        magnitude_CB = np.linalg.norm(CB.T, axis=1)
        normalization_factor = magnitude_AB * magnitude_CB

        # Compute the angle in radians.
        # Angle will be zero if a pair of points overlap (mostly due to wrongly estimated points).
        angle_rad = np.where(
            normalization_factor != 0.0,
            np.arccos((dot_product / normalization_factor).clip(-1, 1)),
            np.zeros_like(dot_product),
        )

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        angle_columns.append(angle_deg)
    return pd.DataFrame(angle_columns).T.set_axis(angle_colnames, axis=1)


def compute_angle_features(X, window_size=30):
    angles = [
        (6, 8, 10),
        (8, 6, 12),
        (7, 9, 11),
        (9, 7, 13),
        (12, 13, 15),
        (13, 12, 14),
        (12, 14, 16),
        (13, 15, 17),
    ]
    angle_features = []
    for angle in angles:
        angle_features.append(compute_angle_3points(X, angle, window_size))
    return pd.concat(angle_features, axis=1)


def get_high_level_features(X, window_size=30, include_face_keypoints=False):
    frames_velocities_accelerations = compute_frames_velocities_and_accelerations(
        X, window_size, include_face_keypoints
    )
    points_distances = compute_points_distances(X, window_size)
    angle_features = compute_angle_features(X, window_size)
    return pd.concat([X, frames_velocities_accelerations, points_distances, angle_features], axis=1)


def compute_window_stats(X):
    angles = [
        (6, 8, 10),
        (8, 6, 12),
        (7, 9, 11),
        (9, 7, 13),
        (12, 13, 15),
        (13, 12, 14),
        (12, 14, 16),
        (13, 15, 17),
    ]

    for angle in angles:
        a, b, c = angle
        X[f"angle_{a}_{b}_{c}_avg"] = X.filter(regex=f"f.*_angle_{a}_{b}_{c}").mean(axis=1)
        X[f"angle_{a}_{b}_{c}_std"] = X.filter(regex=f"f.*_angle_{a}_{b}_{c}").std(axis=1)
        X[f"angle_{a}_{b}_{c}_pct25"] = X.filter(regex=f"f.*_angle_{a}_{b}_{c}").quantile(
            0.25, axis=1
        )
        X[f"angle_{a}_{b}_{c}_pct50"] = X.filter(regex=f"f.*_angle_{a}_{b}_{c}").quantile(
            0.5, axis=1
        )
        X[f"angle_{a}_{b}_{c}_pct75"] = X.filter(regex=f"f.*_angle_{a}_{b}_{c}").quantile(
            0.75, axis=1
        )
        X[f"angle_{a}_{b}_{c}_max"] = X.filter(regex=f"f.*_angle_{a}_{b}_{c}").max(axis=1)
        X[f"angle_{a}_{b}_{c}_min"] = X.filter(regex=f"f.*_angle_{a}_{b}_{c}").min(axis=1)
        X[f"angle_{a}_{b}_{c}_range"] = X[f"angle_{a}_{b}_{c}_max"] - X[f"angle_{a}_{b}_{c}_min"]

    for i in range(6, 18):
        X[f"kp{i}_x_avg"] = X.filter(regex=f"f.*_kp{i}_x").mean(axis=1)
        X[f"kp{i}_x_std"] = X.filter(regex=f"f.*_kp{i}_x").std(axis=1)
        X[f"kp{i}_x_pct25"] = X.filter(regex=f"f.*_kp{i}_x").quantile(0.25, axis=1)
        X[f"kp{i}_x_pct50"] = X.filter(regex=f"f.*_kp{i}_x").quantile(0.5, axis=1)
        X[f"kp{i}_x_pct75"] = X.filter(regex=f"f.*_kp{i}_x").quantile(0.75, axis=1)
        X[f"kp{i}_x_max"] = X.filter(regex=f"f.*_kp{i}_x").max(axis=1)
        X[f"kp{i}_x_min"] = X.filter(regex=f"f.*_kp{i}_x").min(axis=1)
        X[f"kp{i}_x_range"] = X[f"kp{i}_x_max"] - X[f"kp{i}_x_min"]

        X[f"kp{i}_y_avg"] = X.filter(regex=f"f.*_kp{i}_y").mean(axis=1)
        X[f"kp{i}_y_std"] = X.filter(regex=f"f.*_kp{i}_y").std(axis=1)
        X[f"kp{i}_y_pct25"] = X.filter(regex=f"f.*_kp{i}_y").quantile(0.25, axis=1)
        X[f"kp{i}_y_pct50"] = X.filter(regex=f"f.*_kp{i}_y").quantile(0.5, axis=1)
        X[f"kp{i}_y_pct75"] = X.filter(regex=f"f.*_kp{i}_y").quantile(0.75, axis=1)
        X[f"kp{i}_y_max"] = X.filter(regex=f"f.*_kp{i}_y").max(axis=1)
        X[f"kp{i}_y_min"] = X.filter(regex=f"f.*_kp{i}_y").min(axis=1)
        X[f"kp{i}_y_range"] = X[f"kp{i}_y_max"] - X[f"kp{i}_y_min"]

        X[f"vel_kp{i}_x_avg"] = X.filter(regex=f"vel.*_kp{i}_x").mean(axis=1)
        X[f"vel_kp{i}_x_std"] = X.filter(regex=f"vel.*_kp{i}_x").std(axis=1)
        X[f"vel_kp{i}_x_pct25"] = X.filter(regex=f"vel.*_kp{i}_x").quantile(0.25, axis=1)
        X[f"vel_kp{i}_x_pct50"] = X.filter(regex=f"vel.*_kp{i}_x").quantile(0.5, axis=1)
        X[f"vel_kp{i}_x_pct75"] = X.filter(regex=f"vel.*_kp{i}_x").quantile(0.75, axis=1)
        X[f"vel_kp{i}_x_max"] = X.filter(regex=f"vel.*_kp{i}_x").max(axis=1)
        X[f"vel_kp{i}_x_min"] = X.filter(regex=f"vel.*_kp{i}_x").min(axis=1)
        X[f"vel_kp{i}_x_range"] = X[f"vel_kp{i}_x_max"] - X[f"vel_kp{i}_x_min"]

        X[f"vel_kp{i}_y_avg"] = X.filter(regex=f"vel.*_kp{i}_y").mean(axis=1)
        X[f"vel_kp{i}_y_std"] = X.filter(regex=f"vel.*_kp{i}_y").std(axis=1)
        X[f"vel_kp{i}_y_pct25"] = X.filter(regex=f"vel.*_kp{i}_y").quantile(0.25, axis=1)
        X[f"vel_kp{i}_y_pct50"] = X.filter(regex=f"vel.*_kp{i}_y").quantile(0.5, axis=1)
        X[f"vel_kp{i}_y_pct75"] = X.filter(regex=f"vel.*_kp{i}_y").quantile(0.75, axis=1)
        X[f"vel_kp{i}_y_max"] = X.filter(regex=f"vel.*_kp{i}_y").max(axis=1)
        X[f"vel_kp{i}_y_min"] = X.filter(regex=f"vel.*_kp{i}_y").min(axis=1)
        X[f"vel_kp{i}_y_range"] = X[f"vel_kp{i}_x_max"] - X[f"vel_kp{i}_x_min"]

        X[f"acc_kp{i}_x_avg"] = X.filter(regex=f"acc.*_kp{i}_x").mean(axis=1)
        X[f"acc_kp{i}_x_std"] = X.filter(regex=f"acc.*_kp{i}_x").std(axis=1)
        X[f"acc_kp{i}_x_pct25"] = X.filter(regex=f"acc.*_kp{i}_x").quantile(0.25, axis=1)
        X[f"acc_kp{i}_x_pct50"] = X.filter(regex=f"acc.*_kp{i}_x").quantile(0.5, axis=1)
        X[f"acc_kp{i}_x_pct75"] = X.filter(regex=f"acc.*_kp{i}_x").quantile(0.75, axis=1)
        X[f"acc_kp{i}_x_max"] = X.filter(regex=f"acc.*_kp{i}_x").max(axis=1)
        X[f"acc_kp{i}_x_min"] = X.filter(regex=f"acc.*_kp{i}_x").min(axis=1)
        X[f"acc_kp{i}_x_range"] = X[f"acc_kp{i}_x_max"] - X[f"acc_kp{i}_x_min"]

        X[f"acc_kp{i}_y_avg"] = X.filter(regex=f"acc.*_kp{i}_y").mean(axis=1)
        X[f"acc_kp{i}_y_std"] = X.filter(regex=f"acc.*_kp{i}_y").std(axis=1)
        X[f"acc_kp{i}_y_pct25"] = X.filter(regex=f"acc.*_kp{i}_y").quantile(0.25, axis=1)
        X[f"acc_kp{i}_y_pct50"] = X.filter(regex=f"acc.*_kp{i}_y").quantile(0.5, axis=1)
        X[f"acc_kp{i}_y_pct75"] = X.filter(regex=f"acc.*_kp{i}_y").quantile(0.75, axis=1)
        X[f"acc_kp{i}_y_max"] = X.filter(regex=f"acc.*_kp{i}_y").max(axis=1)
        X[f"acc_kp{i}_y_min"] = X.filter(regex=f"acc.*_kp{i}_y").min(axis=1)
        X[f"acc_kp{i}_y_range"] = X[f"acc_kp{i}_y_max"] - X[f"acc_kp{i}_y_min"]
    return X
