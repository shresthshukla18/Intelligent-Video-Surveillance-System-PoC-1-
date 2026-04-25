#cell 2

code = r"""
import cv2
import os
import numpy as np
import torch
import time
import csv
from ultralytics import YOLO
import supervision as sv

BASE_PATH = "/content/drive/MyDrive/cv_project"

VIDEO_FOLDER = f"{BASE_PATH}/videos"
OUTPUT_FOLDER = f"{BASE_PATH}/output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= CONFIG =================
VERTICAL_LINE_PERCENT = 0.5
HORIZONTAL_LINE_PERCENT = 0.5

SLOW_THRESHOLD = 2
FAST_THRESHOLD = 8

MOVING_CLASSES = ["person","car","bus","truck","motorcycle","bicycle"]
# ==========================================


def run_pipeline(video_path):

    import time

    video_name = os.path.basename(video_path)

    # ===== UNIQUE RUN FOLDER =====
    run_id = f"run_{int(time.time())}"
    run_folder = os.path.join(OUTPUT_FOLDER, run_id)
    os.makedirs(run_folder, exist_ok=True)

    # ===== OUTPUT PATHS =====
    output_path = os.path.join(run_folder, "annotated.mp4")
    heatmap_video_path = os.path.join(run_folder, "heatmap.mp4")
    overlay_video_path = os.path.join(run_folder, "overlay.mp4")
    log_path = os.path.join(run_folder, "metrics.csv")

    print("\nRunning PoC-1 on:", video_path)

    # -------- GPU --------
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Device:", "GPU" if device==0 else "CPU")

    model = YOLO("yolov8m.pt")
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    LINE_X = int(width * VERTICAL_LINE_PERCENT)
    LINE_Y = int(height * HORIZONTAL_LINE_PERCENT)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path,fourcc,fps_video,(width,height))
    heatmap_writer = cv2.VideoWriter(heatmap_video_path,fourcc,fps_video,(width,height))
    overlay_writer = cv2.VideoWriter(overlay_video_path,fourcc,fps_video,(width,height))

    # ===== Tracking =====
    TRACK_CONFIRM_FRAMES = 5
    track_age = {}
    confirmed_tracks = set()
    track_history = {}

    total_tracks_created = 0
    total_tracks_lost = 0

    vertical_entry = 0
    vertical_exit = 0

    horizontal_entry = 0
    horizontal_exit = 0

    vertical_crossed_ids = set()
    horizontal_crossed_ids = set()

    peak_occupancy = 0
    peak_frame = 0

    direction_flow = {
        "left_to_right":0,
        "right_to_left":0,
        "top_to_bottom":0,
        "bottom_to_top":0
    }

    heatmap = np.zeros((height,width),dtype=np.float32)

    csv_file = open(log_path,"w",newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
    "Frame","Inference_ms","Frame_ms","FPS",
    "Occupancy","V_Entry","V_Exit","H_Entry","H_Exit",
    "Slow_Count","Moderate_Count","Fast_Count",
    "Tracks_Created","Tracks_Lost"
    ])

    frame_counter = 0
    fps_history = []
    all_fps = []

    # ================= MAIN LOOP =================
    while True:

        ret,frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame_start = time.time()

        heatmap *= 0.995

        # ----- Inference (CPU OPTIMIZED) -----
        t0 = time.time()
        results = model(frame, imgsz=640, conf=0.25, iou=0.5, device=device, verbose=False)[0]
        inference_time_ms = (time.time()-t0)*1000

        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update_with_detections(detections)

        slow_count = moderate_count = fast_count = 0
        current_ids = set()

        for box,track_id,cls in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id):

            x1,y1,x2,y2 = map(int,box)
            current_ids.add(track_id)

            if track_id not in track_age:
                track_age[track_id] = 1
            else:
                track_age[track_id] += 1

            # FIXED TRACK CREATION
            if track_age[track_id] == TRACK_CONFIRM_FRAMES:
                total_tracks_created += 1
                confirmed_tracks.add(track_id)

            if track_id not in confirmed_tracks:
                continue

            cx = int((x1+x2)/2)
            cy = int(y2)

            class_name = model.names[int(cls)]

            # Heatmap
            if class_name in MOVING_CLASSES and 0<=cx<width and 0<=cy<height:
                cv2.circle(heatmap,(cx,cy),8,1,-1)

            track_history.setdefault(track_id, []).append((cx,cy))
            if len(track_history[track_id])>15:
                track_history[track_id].pop(0)

            speed_label = ""

            if len(track_history[track_id]) >= 2:

                prev_cx,prev_cy = track_history[track_id][-2]
                vx,vy = cx-prev_cx, cy-prev_cy
                speed = np.sqrt(vx**2+vy**2)

                if speed < SLOW_THRESHOLD:
                    speed_label="Slow"; slow_count+=1
                elif speed < FAST_THRESHOLD:
                    speed_label="Moderate"; moderate_count+=1
                else:
                    speed_label="Fast"; fast_count+=1

                # Entry / Exit
                if track_id not in vertical_crossed_ids:
                    if prev_cy < LINE_Y and cy >= LINE_Y:
                        vertical_entry += 1
                        vertical_crossed_ids.add(track_id)
                    elif prev_cy >= LINE_Y and cy < LINE_Y:
                        vertical_exit += 1
                        vertical_crossed_ids.add(track_id)

                if track_id not in horizontal_crossed_ids:
                    if prev_cx < LINE_X and cx >= LINE_X:
                        horizontal_entry += 1
                        horizontal_crossed_ids.add(track_id)
                    elif prev_cx >= LINE_X and cx < LINE_X:
                        horizontal_exit += 1
                        horizontal_crossed_ids.add(track_id)

                # Direction
                if abs(vx)>abs(vy):
                    direction_flow["left_to_right" if vx>0 else "right_to_left"]+=1
                else:
                    direction_flow["top_to_bottom" if vy>0 else "bottom_to_top"]+=1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID {track_id} {class_name} {speed_label}",
                        (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # ===== CLEANUP =====
        lost_ids = set(track_age.keys()) - current_ids

        for lost_id in lost_ids:
            if track_age[lost_id] >= TRACK_CONFIRM_FRAMES:
                total_tracks_lost += 1

            track_age.pop(lost_id,None)
            confirmed_tracks.discard(lost_id)
            track_history.pop(lost_id,None)

        occupancy = len(confirmed_tracks)

        if occupancy > peak_occupancy:
            peak_occupancy = occupancy
            peak_frame = frame_counter

        # FPS
        frame_time_ms = (time.time()-frame_start)*1000
        fps = 1000/frame_time_ms if frame_time_ms>0 else 0
        fps_history.append(fps)
        all_fps.append(fps)

        if len(fps_history)>30:
            fps_history.pop(0)

        avg_fps = sum(fps_history)/len(fps_history)

        # Heatmap (OPTIMIZED)
        heatmap_blur = cv2.GaussianBlur(heatmap,(25,25),0)
        heatmap_color = cv2.applyColorMap(
            cv2.normalize(heatmap_blur,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        heatmap_writer.write(heatmap_color)
        overlay_writer.write(cv2.addWeighted(frame,0.7,heatmap_color,0.3,0))

        # Draw
        cv2.line(frame,(0,LINE_Y),(width,LINE_Y),(0,0,255),2)
        cv2.line(frame,(LINE_X,0),(LINE_X,height),(255,0,255),2)

        cv2.putText(frame,f"FPS:{avg_fps:.2f}",(10,40),0,0.7,(0,255,255),2)
        cv2.putText(frame,f"Occ:{occupancy}",(10,70),0,0.7,(255,255,0),2)
        cv2.putText(frame,f"Peak:{peak_occupancy}",(10,100),0,0.7,(255,100,0),2)

        out.write(frame)

        csv_writer.writerow([
            frame_counter, round(inference_time_ms,2), round(frame_time_ms,2),
            round(fps,2), occupancy,
            vertical_entry, vertical_exit,
            horizontal_entry, horizontal_exit,
            slow_count, moderate_count, fast_count,
            total_tracks_created, total_tracks_lost
        ])

    # ===== FINALIZE =====
    cap.release()
    out.release()
    heatmap_writer.release()
    overlay_writer.release()
    csv_file.close()

    final_heatmap = cv2.GaussianBlur(heatmap,(25,25),0)
    final_heatmap = cv2.applyColorMap(
        cv2.normalize(final_heatmap,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    heatmap_image_path = os.path.join(run_folder, "heatmap.png")
    cv2.imwrite(heatmap_image_path, final_heatmap)

    avg_fps_total = sum(all_fps)/len(all_fps) if all_fps else 0

    return {
        "video": output_path,
        "heatmap_video": heatmap_video_path,
        "overlay": overlay_video_path,
        "csv": log_path,
        "heatmap_img": heatmap_image_path,
        "summary": {
            "tracks_created": total_tracks_created,
            "tracks_lost": total_tracks_lost,
            "peak_occupancy": peak_occupancy,
            "direction_flow": direction_flow,
            "avg_fps": round(avg_fps_total, 2)
        }
    }
"""

with open("pipeline.py", "w") as f:
    f.write(code)

print("pipeline.py created successfully")
