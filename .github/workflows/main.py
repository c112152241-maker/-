# -*- coding: utf-8 -*-
import os
import cv2
import time
import serial
import threading
import numpy as np
import re
import csv
import bisect
from datetime import datetime
from YoloPose import Yolo

# ===============================
# 全域資料紀錄 (用於最後存檔)
# ===============================
ALL_Record_FSR = []   # (time_ms, voltage)
ALL_Record_Nano = []  # (time_ms, voltage)
ALL_Record_Yolo = []  # (time_ms, angle)

# 用來紀錄 t0 與 t1 發生的時間點
Event_Log = []        # (time_ms, label_string)

# ===============================
# 相機類別 (保持不變)
# ===============================
class CameraStream:
    def __init__(self, src=None, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(self.blank_frame, "SEARCHING CAMERA...", (50, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        self.stream = None
        self.frame = self.blank_frame.copy()
        self.started = False
        self.read_lock = threading.Lock()
        self.last_retry_time = 0
        self.retry_interval = 0.5
        if self.src is not None: self.connect(self.src)

    def connect(self, idx):
        try:
            if self.stream is not None: self.stream.release()
            print(f"🔄 嘗試連線相機 Index {idx}...")
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.stream = cap
                    self.src = idx
                    print(f"✅ 相機連線成功！(Index {idx})")
                    return True
            cap.release()
        except Exception as e:
            print(f"❌ 連線錯誤: {e}")
        return False

    def scan_and_connect(self):
        for i in range(5): 
            if not os.path.exists(f"/dev/video{i}"): continue
            try:
                temp = cv2.VideoCapture(i, cv2.CAP_V4L2)
                temp.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                if temp.isOpened():
                    ret, _ = temp.read()
                    temp.release()
                    if ret: return self.connect(i)
                else: temp.release()
            except: pass
        return False

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.stream is not None and self.stream.isOpened():
                try:
                    (grabbed, frame) = self.stream.read()
                    if grabbed:
                        with self.read_lock: self.frame = frame
                    else:
                        self.stream.release()
                        self.stream = None
                except: self.stream = None
            else:
                if time.time() - self.last_retry_time > self.retry_interval:
                    self.last_retry_time = time.time()
                    self.scan_and_connect()
                time.sleep(0.1)

    def read(self):
        with self.read_lock: frame = self.frame.copy()
        return frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'): self.thread.join()
        if self.stream is not None: self.stream.release()

# ===============================
# 參數設定
# ===============================
SERIAL_PORT = ["/dev/ttyACM0", "/dev/ttyACM1", "COM3", "COM4"]
BAUD = 9600

FSR_THRESHOLD = 2.5     
NANO_RISE_LEN = 10       

WINDOW_BEFORE = 300     
WINDOW_AFTER  = 2500    
COOLDOWN_MS   = 500     
KEEP_YOLO_MS  = 3000    

fsr_time, fsr_val = [], []
nano_time, nano_val = [], []
yolo_time, yolo_angle = [], []

is_serial_connected = False 

# ===============================
# 訊號邏輯
# ===============================
def find_center_t0(fsr_time, fsr_val, threshold):
    indices = [i for i, v in enumerate(fsr_val) if v > threshold]
    if not indices: return None
    middle_idx = indices[len(indices) // 2]
    return fsr_time[middle_idx]

def find_t1(t0, nano_time, nano_val, rise_len):
    start = next((i for i, t in enumerate(nano_time) if t >= t0), None)
    if start is None: return None
    cnt = 0
    for i in range(start + 1, len(nano_val)):
        if nano_val[i] > nano_val[i - 1]:
            cnt += 1
            if cnt >= rise_len:
                return nano_time[i - cnt]
        else:
            cnt = 0
    return None

# ===============================
# ✅ [修改] CSV 存檔 (Sensor 高頻率版)
# ===============================
def save_csv_log():
    if not ALL_Record_FSR and not ALL_Record_Nano:
        print("⚠️ 沒有 Sensor 數據，跳過存檔。")
        return

    SAVE_DIR = "/home/nvidia/yolo/Test-csv/"

    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
            print(f"📁 已建立資料夾: {SAVE_DIR}")
        except Exception as e:
            print(f"❌ 無法建立資料夾，將存於當前目錄。錯誤: {e}")
            SAVE_DIR = "."

    filename = f"Experiment_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    full_path = os.path.join(SAVE_DIR, filename)

    print(f"\n💾 正在儲存高頻率數據至 {full_path} ...")

    # 1. 建立一個包含所有 FSR 和 Nano 時間點的「主時間軸」
    #    這樣可以確保每一毫秒的變化都被記錄下來
    time_set = set()
    for t, v in ALL_Record_FSR: time_set.add(t)
    for t, v in ALL_Record_Nano: time_set.add(t)
    
    # 排序時間軸 (從小到大)
    master_timeline = sorted(list(time_set))

    # 轉成字典方便快速查找 (Time -> Value)
    fsr_dict = {t: v for t, v in ALL_Record_FSR}
    nano_dict = {t: v for t, v in ALL_Record_Nano}
    
    # 準備 YOLO 數據用於查找 (YOLO 頻率慢，我們用 bisect 找最近的「過去」數值)
    yolo_times = [r[0] for r in ALL_Record_Yolo]
    yolo_vals  = [r[1] for r in ALL_Record_Yolo]

    # 準備事件標記查找
    # 因為 t0/t1 是從 sensor 時間來的，理論上會精準匹配
    events_dict = {e[0]: e[1] for e in Event_Log}

    with open(full_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time (ms)", "Sensor Voltage (V)", "FSR Voltage (V)", "Knee Angle (deg)"])

        # 初始化變數 (Sample and Hold)
        current_fsr = 0.0
        current_nano = 0.0
        current_angle = 0.0
        
        for t in master_timeline:
            # --- 更新 Sensor 數值 ---
            # 如果這個時間點有 FSR 資料，就更新；否則維持上一次的值 (Hold)
            if t in fsr_dict:
                current_fsr = fsr_dict[t]
            
            # 如果這個時間點有 Nano 資料，就更新；否則維持上一次的值 (Hold)
            if t in nano_dict:
                current_nano = nano_dict[t]

            # --- 更新 YOLO 數值 (Sample and Hold) ---
            # 找出在這個時間點 t 之前(或剛好) 最新的 YOLO 數據索引
            # bisect_right 會回傳插入點，所以 idx-1 就是小於等於 t 的最後一個數據
            if yolo_times:
                idx = bisect.bisect_right(yolo_times, t)
                if idx > 0:
                    current_angle = yolo_vals[idx-1]
                # else: idx == 0 代表這個時間點比第一幀 YOLO 還早，維持 0.0 或等待

            # --- 處理標記 ---
            time_str = f"{t:.2f}"
            
            # 因為浮點數可能有微小誤差，我們檢查是否在事件表中
            # 但因為 master_timeline 就是由 sensor 數據組成的，理論上 t0/t1 會完全相等
            if t in events_dict:
                time_str += f" *{events_dict[t]}"

            writer.writerow([time_str, f"{current_nano:.4f}", f"{current_fsr:.4f}", f"{current_angle:.2f}"])
            
    print(f"✅ CSV 存檔完成！(高頻率模式)")

# ===============================
# Serial Reader
# ===============================
def serial_reader(stop_event, base_counter):
    global is_serial_connected
    print("🔍 啟動 Serial 監聽...")
    re_sensor = re.compile(r"Sensor Voltage:\s*([0-9]*\.?[0-9]+)\s*V", re.IGNORECASE)
    re_fsr    = re.compile(r"FSR Voltage:\s*([0-9]*\.?[0-9]+)\s*V", re.IGNORECASE)

    while not stop_event.is_set():
        ser = None
        port = next((p for p in SERIAL_PORT if os.path.exists(p)), None)
        
        if port:
            try:
                ser = serial.Serial(port, BAUD, timeout=1, write_timeout=1)
                print(f"✅ Serial 連線成功: {port}")
                is_serial_connected = True
            except Exception as e:
                print(f"⚠️ 無法開啟 Port {port}: {e}")
                time.sleep(1)
                continue 
        else:
            time.sleep(1)
            continue

        try:
            while not stop_event.is_set():
                try:
                    line_bytes = ser.readline()
                    if not line_bytes: continue 
                    line = line_bytes.decode(errors="ignore").strip()

                except (serial.SerialException, OSError) as e:
                    print(f"❌ Serial 斷線或重置偵測 ({e})，準備重連...")
                    is_serial_connected = False
                    break 

                except Exception as e:
                    print(f"⚠️ 未知 Serial 錯誤: {e}")
                    break

                if not line: continue
                
                now_ms = (time.perf_counter() - base_counter) * 1000.0

                if "Sensor Voltage" in line:
                    m = re_sensor.search(line)
                    if m:
                        val = float(m.group(1))
                        nano_time.append(now_ms)
                        nano_val.append(val)
                        ALL_Record_Nano.append((now_ms, val))

                if "FSR Voltage" in line:
                    m = re_fsr.search(line)
                    if m:
                        val = float(m.group(1))
                        fsr_time.append(now_ms)
                        fsr_val.append(val)
                        ALL_Record_FSR.append((now_ms, val))
        finally:
            if ser and ser.is_open:
                ser.close()
            is_serial_connected = False
            print("🔄 Serial 資源已釋放，等待重連...")

# ===============================
# 主程式
# ===============================
def main():
    base_counter = time.perf_counter()
    yolo = Yolo()
    cam = CameraStream(src=None, width=640, height=480, fps=30).start()
    
    stop_event = threading.Event()
    th = threading.Thread(target=serial_reader, args=(stop_event, base_counter), daemon=True)
    th.start()

    print("🚀 系統啟動！")
    
    last_event_t1 = -1e9
    prev_time = time.time()

    is_waiting_motion = False
    collect_deadline = 0 
    cached_t0 = 0 
    cached_t1 = 0 

    display_angle = 0.0

    try:
        while True:
            # --- 影像處理 ---
            frame = cam.read()
            img, kpts = yolo.inference(frame)
            out = yolo.draw(img)
            if out is not None: img = out

            fps = 1.0 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
            prev_time = time.time()
            
            # --- 取得當前數值 ---
            now_ms = (time.perf_counter() - base_counter) * 1000.0
            
            if is_serial_connected and fsr_time and (now_ms - fsr_time[-1] < 1000):
                cur_fsr = fsr_val[-1]
            else:
                cur_fsr = 0.0
            
            if is_serial_connected and nano_time and (now_ms - nano_time[-1] < 1000):
                cur_nano = nano_val[-1]
            else:
                cur_nano = 0.0

            # YOLO 角度計算
            if kpts and len(kpts) > 13:
                hip, knee, ankle = kpts[8], kpts[9], kpts[10]
                if hip != (0, 0) and knee != (0, 0) and ankle != (0, 0):
                    angle = yolo.get_angle(hip, knee, ankle)
                    if angle is not None:
                        yolo_time.append(now_ms)
                        yolo_angle.append(angle)
                        display_angle = angle
                        ALL_Record_Yolo.append((now_ms, angle))

            # ===============================
            # UI 顯示
            # ===============================
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            
            color_text = (150, 255, 150) if is_serial_connected else (0, 0, 255)
            status_text = "Connected" if is_serial_connected else "No Signal"
            cv2.putText(img, f"Serial: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
            cv2.putText(img, f"FSR:    {cur_fsr:.2f} V", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
            cv2.putText(img, f"Sensor: {cur_nano:.2f} V", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
            cv2.putText(img, f"Angle:  {display_angle:.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)

            # ===============================
            # 觸發與結束邏輯
            # ===============================
            if not is_waiting_motion:
                t0 = find_center_t0(fsr_time, fsr_val, FSR_THRESHOLD)
                
                if t0:
                    if t0 > last_event_t1 + COOLDOWN_MS:
                        t1 = find_t1(t0, nano_time, nano_val, NANO_RISE_LEN)

                        if t1 and (t1 > t0) and (t1 - t0 <= 200):
                            print(f"\n⚡ 訊號觸發!")
                            is_waiting_motion = True
                            cached_t0 = t0
                            cached_t1 = t1
                            collect_deadline = now_ms + WINDOW_AFTER
                            
                            Event_Log.append((t0, "t0"))
                            Event_Log.append((t1, "t1"))
                            
                        elif now_ms - t0 > 200:
                            cut_time = t0 + 100
                            cut = next((i for i, t in enumerate(fsr_time) if t > cut_time), 0)
                            del fsr_time[:cut], fsr_val[:cut]
                            cut = next((i for i, t in enumerate(nano_time) if t > cut_time), 0)
                            del nano_time[:cut], nano_val[:cut]
                    else:
                        cut = next((i for i, t in enumerate(fsr_time) if t > t0), 0)
                        del fsr_time[:cut], fsr_val[:cut]

            else:
                # -----------------------------------------------------------------
                # 動態穩定檢測 (Method 2)
                # -----------------------------------------------------------------
                cv2.putText(img, "Analyzing Motion...", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                STABLE_CHECK_DELAY = 600 
                
                if (now_ms - cached_t1) > STABLE_CHECK_DELAY:
                    CHECK_FRAMES = 15
                    if len(yolo_angle) >= CHECK_FRAMES:
                        recent_angles = yolo_angle[-CHECK_FRAMES:]
                        std_dev = np.std(recent_angles)
                        
                        if std_dev < 0.8:
                            print(f"🛑 檢測到動作穩定 (Std: {std_dev:.2f})，提早結束測量。")
                            collect_deadline = now_ms 

                # -----------------------------------------------------------------
                # 結算與報告 (計算方式更新)
                # -----------------------------------------------------------------
                if now_ms >= collect_deadline:
                    print("📊 計算數據...")
                    t0 = cached_t0
                    t1 = cached_t1
                    t_end = collect_deadline 
                    
                    mask = [(t >= t0 - WINDOW_BEFORE) and (t <= t_end) for t in yolo_time]
                    ang_seg = np.array([a for a, m in zip(yolo_angle, mask) if m])
                    tim_seg = np.array([t for t, m in zip(yolo_time, mask) if m])

                    if len(ang_seg) > 0:
                        idx_t0 = np.argmin(np.abs(tim_seg - t0))
                        start_avg_idx = max(0, idx_t0 - 4)
                        init_angles_subset = ang_seg[start_avg_idx : idx_t0 + 1]
                        
                        if len(init_angles_subset) > 0:
                            init_a = np.mean(init_angles_subset)
                        else:
                            init_a = ang_seg[idx_t0]

                        # 找出 t1 之後的索引
                        valid_indices = np.where(tim_seg > t1)[0]
                        
                        if len(valid_indices) > 0:
                            # 1. 找最大角度
                            rel_max_idx = np.argmax(ang_seg[valid_indices])
                            real_max_idx = valid_indices[rel_max_idx]
                            
                            max_a = ang_seg[real_max_idx]
                            t_peak = tim_seg[real_max_idx]

                            # 2. 找最小角度
                            min_a = max_a 
                            valid_indices_after_peak = np.where(tim_seg > t_peak)[0]
                            result_min_a = min_a

                            if len(valid_indices_after_peak) > 0:
                                rel_min_idx = np.argmin(ang_seg[valid_indices_after_peak])
                                real_min_idx = valid_indices_after_peak[rel_min_idx]
                                result_min_a = ang_seg[real_min_idx]
                            
                            # ====================================================
                            # 🆕 [更新] 逐幀角速度計算
                            # ====================================================
                            if max_a > init_a + 1.5:
                                total_reflex_time = t_end - t0
                                
                                # 準備計算容器
                                frame_velocities_all = [] # 存 t1 -> t_end 每一幀的速度
                                frame_velocities_ext = [] # 存 t1 -> t_peak (伸展期) 每一幀的速度

                                # 遍歷 t1 之後的每一幀進行微分
                                if len(valid_indices) > 1:
                                    for i in range(1, len(valid_indices)):
                                        curr_idx = valid_indices[i]
                                        prev_idx = valid_indices[i-1]
                                        
                                        dt = tim_seg[curr_idx] - tim_seg[prev_idx]
                                        d_ang = ang_seg[curr_idx] - ang_seg[prev_idx]
                                        
                                        # 避免除以零 (如果fps極高可能發生)
                                        if dt > 0:
                                            # 瞬時速度 (deg/s)
                                            v_inst = (d_ang / dt) * 1000.0
                                            frame_velocities_all.append(v_inst)
                                            
                                            # 如果當前時間還沒超過峰值時間，歸類為伸展期
                                            if tim_seg[curr_idx] <= t_peak:
                                                frame_velocities_ext.append(v_inst)

                                # 計算平均值
                                # A. 全程平均角速率 (Average Absolute Velocity) - 包含抬起與落下，取絕對值看"活動力"
                                if frame_velocities_all:
                                    avg_velocity_total_abs = np.mean(np.abs(frame_velocities_all))
                                else:
                                    avg_velocity_total_abs = 0.0

                                print(f"--- 結果報告 ---")
                                print(f"反射時間(RL): {t1 - t0:.1f} ms")
                                print(f"初始角度(RA): {init_a:.1f} deg") 
                                print(f"最大角度(REA): {max_a:.1f} deg") 
                                print(f"最大伸展角位移(PEAD): {max_a - init_a:.1f} deg")
                                print(f"到峰時間(TTE): {t_peak - t1:.1f} ms")
                                print(f"小腿抬起平均角速度：{(max_a - init_a) / (t_peak - t1) * 1000:.1f} deg/s")
                                print(f"最小角度(PFA): {result_min_a:.1f} deg")
                                print(f"最大屈曲角位移(PFAD): {max_a - result_min_a:.1f} deg")
                                print(f"總反射平均角速率(AV): {avg_velocity_total_abs:.1f} deg/s")
                                print(f"總反射時間(TPTR): {total_reflex_time:.1f} ms")
                                print(f"----------------")
                            else:
                                print("⚠️ 動作幅度過小")
                        else:
                            print("⚠️ T1 後無影像數據")
                    else:
                        print("⚠️ 無影像數據")

                    is_waiting_motion = False
                    last_event_t1 = t1
                    
                    cut = next((i for i, t in enumerate(fsr_time) if t > t1), 0)
                    del fsr_time[:cut], fsr_val[:cut]
                    cut = next((i for i, t in enumerate(nano_time) if t > t1), 0)
                    del nano_time[:cut], nano_val[:cut]

            cut = next((i for i, t in enumerate(yolo_time) if t >= now_ms - KEEP_YOLO_MS), 0)
            del yolo_time[:cut], yolo_angle[:cut]

            cv2.imshow("System", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        stop_event.set()
        cam.stop()
        cv2.destroyAllWindows()
        save_csv_log()

if __name__ == "__main__":
    main()