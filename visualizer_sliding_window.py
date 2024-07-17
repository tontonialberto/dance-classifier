import pandas as pd
import numpy as np
import cv2

def plot_skeleton(img, points, confs, origin_x, origin_y, include_face_keypoints=True, show_keypoint_numbers=False):
    points = np.array(points)
    confs = np.array(confs)
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
        [12, 14]
    ]
    
    points += [origin_x, origin_y]
    
    if not include_face_keypoints:
        points = np.concatenate([[[0,0]] * 5, points])
    
    # cv2.rectangle(img, (int(box["x1"]), int(box["y1"])), (int(box["x2"]), int(box["y2"])), (255, 0, 0))
    for i, point in enumerate(points):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 255, 0))
            if show_keypoint_numbers:
                cv2.putText(img, str(i+1), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    for pair in line_pairs:
        point1 = points[pair[0]]
        point2 = points[pair[1]]
        if point1[0] != 0 and point1[1] != 0 and point2[0] != 0 and point2[1] != 0:
            cv2.line(img, (int(points[pair[0]][0]), int(points[pair[0]][1])), (int(points[pair[1]][0]), int(points[pair[1]][1])), (0, 255, 0))
    
    # center_of_gravity_x = np.sum(points.T[0]) / sum(points.T[0] != 0)
    # center_of_gravity_y = np.sum(points.T[1]) / sum(points.T[1] != 0)
    # cg = [center_of_gravity_x, center_of_gravity_y]
    # cv2.circle(img, (int(cg[0]), int(cg[1])), 5, (0, 255, 255))
    # cv2.putText(img, "CG", (int(cg[0]), int(cg[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    

video_csv_path = "resources/medium_dataset_swstep2/dataset/bachata_Bailando Bachata- Chayanne ｜ Daniel y Tom Bachata Groove in Budapest [eN5CCONmJTU].mp4.json.csv"

df = pd.read_csv(video_csv_path)

columns = np.array([])
for i in range(1, 31):
    columns = np.concatenate([columns, np.array([(f"f{i}_kp{j}_x", f"f{i}_kp{j}_y") for j in range(6, 18)]).flatten()])

track_ids = df["track_id"].unique()
print("track_ids", track_ids)

persons = []
for track_id in track_ids:
    persons.append(df[df["track_id"] == track_id])
    
max_windows = max([len(person) for person in persons])

for i in range(max_windows):
    for k in range(30):
        img = np.zeros((500, 1300, 3), np.uint8)
        for j, person in enumerate(persons):
            if i < len(person):
                window = person.iloc[i]
                movement = window[columns].values.reshape((30, 12, 2))
                x_offset = (j * 100) % 1200
                y_offset = (j // 12) * 100
                plot_skeleton(img, movement[k] * 100, [1]*12, x_offset, y_offset, include_face_keypoints=False)
                description = f"{window['track_id']}, {i}/{len(person)}, {k}"
                cv2.putText(img, description, (x_offset, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.imshow("window", img)    
        if (cv2.waitKey(25) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()