import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time
from flask import Flask, Response, render_template, jsonify
from collections import defaultdict

app   = Flask(__name__)

path_45 = '/home/george/datasets/ikar_dataset_stratify_test_and_plus/yolo_runs/ablation_noise-13_best/weights/best.engine'
path_full = "/home/george/datasets/90_and_45_full/yolo_runs/ablation_noise-4/weights/best.engine"
path_full_new_augs = '/home/george/datasets/90_and_45_full/yolo_runs_new_augs/ablation_all_soft/weights/best.engine'

model = YOLO(path_full_new_augs, task='detect'
)

# model = YOLO('/models/best.engine', task='detect')

CAMERA_PORT  = '/dev/video1'
CONF         = 0.3
DEVICE       = 0
IMGSZ        = 1024
BBOX_PERSIST = 15  # показываем bbox ещё 15 кадров после исчезновения

COLORS = {
    'flange':    (0, 255, 0),
    'obloy':     (0, 0, 255),
    'underfill': (0, 165, 255),
}

frame_queue  = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)
output_frame = None
lock         = threading.Lock()

latest_status  = 'OK'
latest_defects = []
latest_conf    = 0.0
fps_value      = 0

# Буфер последних bbox
last_boxes   = []
last_seen    = 0  # номер кадра когда последний раз видели bbox

def capture_thread():
    cap = cv2.VideoCapture(CAMERA_PORT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print('Камера запущена')

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put_nowait(frame)

def inference_thread():
    global output_frame
    global last_boxes
    global last_seen

    global latest_status
    global latest_defects
    global latest_conf
    global fps_value
    
    print('Инференс запущен')
    frame_num = 0
    prev_time = time.time()

    while True:
        try:
            frame     = frame_queue.get(timeout=1)
            frame_num += 1
            results   = model(
                frame,
                imgsz=IMGSZ,
                conf=CONF,
                device=DEVICE,
                verbose=False
            )[0]

            # Собираем текущие bbox
            current_boxes = []
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                cls_name = results.names[cls_id]
                conf     = float(box.conf[0])
                coords   = list(map(int, box.xyxy[0]))
                current_boxes.append({
                    'cls':   cls_name,
                    'conf':  conf,
                    'coords': coords
                })

            # Обновляем буфер
            if current_boxes:
                last_boxes = current_boxes
                last_seen  = frame_num

            # Используем текущие bbox или буферизованные
            boxes_to_draw = current_boxes
            if not boxes_to_draw and (frame_num - last_seen) < BBOX_PERSIST:
                boxes_to_draw = last_boxes  # показываем старые

            # Отрисовка
            defects = []
            for det in boxes_to_draw:
                cls_name     = det['cls']
                conf         = det['conf']
                x1, y1, x2, y2 = det['coords']
                color        = COLORS.get(cls_name, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, f'{cls_name} {conf:.2f}',
                #            (x1, y1-10),
                #            cv2.FONT_HERSHEY_SIMPLEX,
                #            0.6, color, 2)

                if cls_name != 'flange':
                    defects.append(cls_name)

            status = f'Defect: {", ".join(set(defects))}' if defects else 'OK'
            color  = (0, 0, 255) if defects else (0, 255, 0)
            latest_status = status
            latest_defects = defects
            
            if boxes_to_draw:
                latest_conf = max([d['conf'] for d in boxes_to_draw])
            else:
                latest_conf = 0.0
            
            curr_time = time.time()
            
            fps_value = int(
                1 / (curr_time - prev_time + 1e-6)
            )
            
            prev_time = curr_time
            
            # cv2.putText(frame, status, (50, 80),
            #            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

            display = cv2.resize(frame, (1280, 720))
            with lock:
                output_frame = display.copy()

        except queue.Empty:
            continue
        except Exception as e:
            print(f'Ошибка: {e}')

def generate_stream():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes +
               b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():

    return jsonify({
        'status': latest_status,
        'defects': latest_defects,
        'confidence': round(latest_conf, 2),
        'fps': fps_value,
        'camera': 'ONLINE',
        'engine': 'ACTIVE'
    })

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

t_capture   = threading.Thread(target=capture_thread,   daemon=True)
t_inference = threading.Thread(target=inference_thread, daemon=True)
t_capture.start()
t_inference.start()

print('Открой в браузере: http://10.74.50.224:5000')
app.run(host='0.0.0.0', port=5000, debug=False)