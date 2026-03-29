# -*- coding: utf-8 -*-
import sys
import cv2
import time
import serial
import serial.tools.list_ports
import re
import os
import numpy as np
import sqlite3
import logging
from datetime import datetime
import qdarktheme

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
                             QLineEdit, QMessageBox, QDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap

from YoloPose import Yolo

# ==========================================
# 🌟 系統日誌設定
# ==========================================
logging.basicConfig(
    filename='reflex_system.log', level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s', encoding='utf-8'
)

# ==========================================
# 🌟 背景資料庫儲存執行緒
# ==========================================
class DatabaseSaveThread(QThread):
    success_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, record_data):
        super().__init__()
        self.record_data = record_data

    def run(self):
        try:
            conn = sqlite3.connect("ReflexRecords.db")
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO patients_records 
                (patient_id, test_time, latency, init_angle, max_angle, 
                 max_ext_disp, time_to_peak, raise_vel, min_angle, 
                 max_flex_disp, avg_vel, total_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', self.record_data)
            conn.commit()
            conn.close()
            logging.info(f"成功儲存病歷紀錄: {self.record_data[0]}")
            self.success_signal.emit(self.record_data[0])
        except Exception as e:
            logging.error(f"資料庫寫入失敗: {e}")
            self.error_signal.emit(str(e))

# ==========================================
# 1. 歷史紀錄查詢視窗
# ==========================================
class HistoryDialog(QDialog):
    def __init__(self, default_pid=""):
        super().__init__()
        self.setWindowTitle("病患歷史紀錄查詢")
        self.resize(1200, 500)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0;")

        layout = QVBoxLayout(self)
        search_layout = QHBoxLayout()
        
        self.input_search = QLineEdit(default_pid)
        self.input_search.setStyleSheet("font-size: 16px; padding: 5px;")
        btn_search = QPushButton("🔍 搜尋")
        btn_search.setStyleSheet("font-size: 16px; background-color: #2980b9; color: white; padding: 5px 15px;")
        btn_search.clicked.connect(self.load_data)

        search_layout.addWidget(QLabel("病歷號查詢:").setStyleSheet("font-size: 16px; font-weight: bold;"))
        search_layout.addWidget(self.input_search)
        search_layout.addWidget(btn_search)
        layout.addLayout(search_layout)

        self.table = QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels([
            "測試時間", "病歷號", "反射時間(ms)", "初始角度(deg)", "最大角度(deg)", 
            "最大伸展位移(deg)", "到峰時間(ms)", "抬起角速度(deg/s)", "最小角度(deg)", 
            "最大屈曲位移(deg)", "平均角速度(deg/s)", "總反射時間(ms)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("QTableWidget { background-color: #1e1e1e; color: #fff; font-size: 14px; } QHeaderView::section { background-color: #333; color: #fff; padding: 5px; }")
        layout.addWidget(self.table)
        self.load_data()

    def load_data(self):
        pid = self.input_search.text().strip()
        self.table.setRowCount(0)
        try:
            conn = sqlite3.connect("ReflexRecords.db")
            cursor = conn.cursor()
            query = "SELECT test_time, patient_id, latency, init_angle, max_angle, max_ext_disp, time_to_peak, raise_vel, min_angle, max_flex_disp, avg_vel, total_time FROM patients_records"
            if pid:
                cursor.execute(query + " WHERE patient_id = ? ORDER BY test_time DESC", (pid,))
            else:
                cursor.execute(query + " ORDER BY test_time DESC")
            records = cursor.fetchall()
            conn.close()

            for row_idx, row_data in enumerate(records):
                self.table.insertRow(row_idx)
                for col_idx, data in enumerate(row_data):
                    item = QTableWidgetItem(f"{data:.2f}" if isinstance(data, float) else str(data))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row_idx, col_idx, item)
        except Exception as e:
            logging.error(f"歷史紀錄讀取失敗: {e}")

# ==========================================
# 2. 影像與 YOLO 執行緒 
# ==========================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_angle_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.yolo = Yolo()

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(0)

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                img, kpts = self.yolo.inference(frame)
                out_img = self.yolo.draw(img)
                if out_img is not None: img = out_img

                if kpts and len(kpts) >= 17:
                    hip, knee, ankle = kpts[12], kpts[14], kpts[16]
                    if hip != (0, 0) and knee != (0, 0) and ankle != (0, 0):
                        angle = self.yolo.get_angle(hip, knee, ankle)
                        if angle is not None:
                            self.update_angle_signal.emit(angle)
                self.change_pixmap_signal.emit(img)
            else: time.sleep(0.1)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==========================================
# 3. Serial 感測器執行緒 
# ==========================================
class SerialThread(QThread):
    update_sensor_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.baud = 9600

    def run(self):
        re_sensor = re.compile(r"Sensor Voltage:\s*([0-9]*\.?[0-9]+)\s*V", re.IGNORECASE)
        re_fsr    = re.compile(r"FSR Voltage:\s*([0-9]*\.?[0-9]+)\s*V", re.IGNORECASE)

        while self._run_flag:
            ser, connected_port = None, None
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if available_ports:
                for p in available_ports:
                    try:
                        ser = serial.Serial(p, self.baud, timeout=1)
                        self.update_sensor_signal.emit({"status": f"🟢 連線成功 ({p})"})
                        break
                    except: continue

            if not ser or not ser.is_open:
                self.update_sensor_signal.emit({"status": "🔴 尋找硬體設備中..."})
                time.sleep(1)
                continue

            try:
                while self._run_flag:
                    line_bytes = ser.readline()
                    if not line_bytes: continue 
                    line = line_bytes.decode(errors="ignore").strip()
                    data_dict = {}
                    if "Sensor Voltage" in line:
                        m = re_sensor.search(line)
                        if m: data_dict['nano'] = float(m.group(1))
                    if "FSR Voltage" in line:
                        m = re_fsr.search(line)
                        if m: data_dict['fsr'] = float(m.group(1))
                    if data_dict: self.update_sensor_signal.emit(data_dict)
            except Exception:
                self.update_sensor_signal.emit({"status": "🔴 連線意外中斷，重新掃描..."})
            finally:
                if ser and ser.is_open: ser.close()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==========================================
# 4. 主視窗 (🌟 Level 3: 演算法大腦實裝)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("下肢反射分析系統 - 臨床診斷介面")
        self.resize(1300, 800)
        self.init_database()

        # 🌟 即時硬體數值暫存區
        self.current_fsr = 0.0
        self.current_nano = 0.0
        self.current_angle = 0.0

        # 🌟 核心數據運算狀態機
        self.data_buffer = []       # 收集時序數據 [time, fsr, nano, angle]
        self.test_state = "IDLE"    # 狀態: IDLE -> WAITING -> RECORDING -> DONE
        self.record_start_time = 0  
        self.fsr_threshold = 0.5    # 🌟 觸發門檻 (FSR電壓大於 0.5V 視為敲擊)
        self.record_duration = 2.0  # 🌟 敲擊後紀錄 2 秒鐘的肌肉與骨架變化
        
        # 設定定時器 (每 20ms 取樣一次，相當於 50Hz)
        self.sample_timer = QTimer()
        self.sample_timer.timeout.connect(self.sample_data)

        self.current_results = {
            "latency": 0.0, "init_angle": 0.0, "max_angle": 0.0,
            "max_ext_disp": 0.0, "time_to_peak": 0.0, "raise_vel": 0.0,
            "min_angle": 0.0, "max_flex_disp": 0.0, "avg_vel": 0.0, "total_time": 0.0
        }

        self.initUI()
        self.video_thread = VideoThread()
        self.serial_thread = SerialThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_angle_signal.connect(self.update_angle)
        self.serial_thread.update_sensor_signal.connect(self.update_sensors)
        self.serial_thread.start()

    def init_database(self):
        try:
            conn = sqlite3.connect("ReflexRecords.db")
            conn.cursor().execute('''
                CREATE TABLE IF NOT EXISTS patients_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, test_time TEXT NOT NULL,
                    latency REAL, init_angle REAL, max_angle REAL, max_ext_disp REAL, time_to_peak REAL, 
                    raise_vel REAL, min_angle REAL, max_flex_disp REAL, avg_vel REAL, total_time REAL)
            ''')
            conn.commit()
            conn.close()
        except Exception as e: logging.error(f"資料庫初始化失敗: {e}")

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.image_label = QLabel("請點擊開始測試啟動相機")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #888; font-size: 20px;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label, stretch=5)

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=5)

        status_group = QGroupBox("即時監測 (Real-time)")
        status_layout = QVBoxLayout()
        self.lbl_serial_status = QLabel("感測器狀態: 🔴 搜尋中...")
        self.lbl_fsr = QLabel("FSR 電壓: 0.00 V")
        self.lbl_nano = QLabel("Nano 電壓: 0.00 V")
        self.lbl_angle = QLabel("當前角度: -- deg")
        self.lbl_angle.setStyleSheet("font-size: 20px; font-weight: bold; color: #3498db;")
        
        # 🌟 測試狀態指示燈
        self.lbl_test_state = QLabel("測試狀態: ⚪ 尚未開始")
        self.lbl_test_state.setStyleSheet("font-size: 20px; font-weight: bold; color: #f39c12; background-color: #333; padding: 5px;")
        
        for w in [self.lbl_serial_status, self.lbl_fsr, self.lbl_nano, self.lbl_angle, self.lbl_test_state]:
            status_layout.addWidget(w)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        # 測試結果面板
        result_group = QGroupBox("測試結果 (Test Results)")
        self.result_grid = QGridLayout()
        self.update_result_labels() # 初始化標籤
        result_group.setLayout(self.result_grid)
        right_layout.addWidget(result_group)

        # 資料庫面板
        save_group = QGroupBox("資料庫建檔 (Database)")
        save_layout = QVBoxLayout()
        pid_layout = QHBoxLayout()
        self.input_pid = QLineEdit()
        self.input_pid.setPlaceholderText("請輸入病患 ID...")
        self.input_pid.setStyleSheet("font-size: 16px; padding: 5px;")
        pid_layout.addWidget(QLabel("病歷號:"))
        pid_layout.addWidget(self.input_pid)
        save_layout.addLayout(pid_layout)

        self.btn_save = QPushButton("💾 儲存測試紀錄")
        self.btn_save.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #2980b9; color: white; padding: 10px;")
        self.btn_save.clicked.connect(self.save_to_database)
        save_layout.addWidget(self.btn_save)
        
        self.btn_history = QPushButton("📊 查看歷史紀錄")
        self.btn_history.setStyleSheet("font-size: 18px; background-color: #8e44ad; color: white; padding: 10px;")
        self.btn_history.clicked.connect(lambda: HistoryDialog(self.input_pid.text().strip()).exec())
        save_layout.addWidget(self.btn_history)
        
        save_group.setLayout(save_layout)
        right_layout.addWidget(save_group)
        right_layout.addStretch()

        self.btn_start = QPushButton("▶ 開始監測 (進入等待敲擊模式)")
        self.btn_start.setMinimumHeight(60)
        self.btn_start.setStyleSheet("font-size: 22px; font-weight: bold; background-color: #27ae60; color: white;")
        self.btn_start.clicked.connect(self.toggle_system)
        right_layout.addWidget(self.btn_start)

    def update_result_labels(self):
        # 刪除舊標籤再重畫，確保數值更新
        for i in reversed(range(self.result_grid.count())): 
            self.result_grid.itemAt(i).widget().setParent(None)
        
        font = "font-size: 15px; color: #e0e0e0;"
        labels = [
            QLabel(f"反射時間: {self.current_results['latency']:.1f} ms"),
            QLabel(f"初始角度: {self.current_results['init_angle']:.1f} deg"),
            QLabel(f"最大角度: {self.current_results['max_angle']:.1f} deg"),
            QLabel(f"最大伸展角位移: {self.current_results['max_ext_disp']:.1f} deg"),
            QLabel(f"到峰時間: {self.current_results['time_to_peak']:.1f} ms"),
            QLabel(f"小腿抬起角速度: {self.current_results['raise_vel']:.1f} deg/s"),
            QLabel(f"最小角度: {self.current_results['min_angle']:.1f} deg"),
            QLabel(f"最大屈曲角位移: {self.current_results['max_flex_disp']:.1f} deg"),
            QLabel(f"平均角速度: {self.current_results['avg_vel']:.1f} deg/s"),
            QLabel(f"總反射時間: {self.current_results['total_time']:.1f} ms")
        ]
        for l in labels: l.setStyleSheet(font)
        for i in range(5):
            self.result_grid.addWidget(labels[i], i, 0)
            self.result_grid.addWidget(labels[i+5], i, 1)

    # 🌟 背景取樣定時器 (串接感測器與相機)
    def sample_data(self):
        current_t = time.time()
        
        # 狀態：等待敲擊
        if self.test_state == "WAITING":
            # 偵測到槌子敲擊 FSR
            if self.current_fsr > self.fsr_threshold:
                self.test_state = "RECORDING"
                self.record_start_time = current_t
                self.data_buffer = [] # 清空並準備紀錄
                self.lbl_test_state.setText("測試狀態: 🔴 偵測到敲擊！紀錄中...")
                self.lbl_test_state.setStyleSheet("font-size: 20px; font-weight: bold; color: white; background-color: #c0392b; padding: 5px;")
                logging.info("偵測到敲擊，開始紀錄數據")

        # 狀態：紀錄 2 秒內的變化
        elif self.test_state == "RECORDING":
            self.data_buffer.append((current_t, self.current_fsr, self.current_nano, self.current_angle))
            
            # 兩秒時間到，啟動強大演算法！
            if current_t - self.record_start_time >= self.record_duration:
                self.sample_timer.stop()
                self.test_state = "DONE"
                self.lbl_test_state.setText("測試狀態: 🟢 分析完成！")
                self.lbl_test_state.setStyleSheet("font-size: 20px; font-weight: bold; color: white; background-color: #27ae60; padding: 5px;")
                self.calculate_reflex_metrics() # 呼叫大腦算數學

    # 🌟 系統大腦：10 項核心指標演算法
    def calculate_reflex_metrics(self):
        if len(self.data_buffer) < 10:
            QMessageBox.warning(self, "錯誤", "紀錄的數據太少，無法分析！")
            return

        # 將快取轉為 numpy 陣列方便數學運算
        data = np.array(self.data_buffer)
        times = data[:, 0]
        fsrs = data[:, 1]
        nanos = data[:, 2]
        angles = data[:, 3]

        try:
            # 1. 找敲擊點 (FSR 峰值)
            strike_idx = np.argmax(fsrs)
            t0 = times[strike_idx]

            # 2. 找肌肉反應點 (Nano 峰值)
            post_strike_nanos = nanos[strike_idx:]
            nano_peak_idx = strike_idx + np.argmax(post_strike_nanos)
            t1 = times[nano_peak_idx]

            # 計算 [反射時間]
            self.current_results["latency"] = abs(t1 - t0) * 1000.0

            # 3. 角度位移分析
            self.current_results["init_angle"] = angles[strike_idx]
            post_strike_angles = angles[strike_idx:]
            
            # 尋找腿踢到最高點 (最大角度)
            max_angle_idx = strike_idx + np.argmax(post_strike_angles)
            self.current_results["max_angle"] = angles[max_angle_idx]
            t2 = times[max_angle_idx]

            self.current_results["max_ext_disp"] = abs(self.current_results["max_angle"] - self.current_results["init_angle"])
            self.current_results["time_to_peak"] = abs(t2 - t1) * 1000.0

            # 避免除以零
            time_diff_sec = abs(t2 - t1)
            self.current_results["raise_vel"] = self.current_results["max_ext_disp"] / time_diff_sec if time_diff_sec > 0 else 0.0

            # 4. 腿部回擺 (屈曲)
            post_peak_angles = angles[max_angle_idx:]
            self.current_results["min_angle"] = np.min(post_peak_angles) if len(post_peak_angles) > 0 else self.current_results["max_angle"]
            self.current_results["max_flex_disp"] = abs(self.current_results["max_angle"] - self.current_results["min_angle"])

            # 總時間與總平均角速度
            self.current_results["total_time"] = self.record_duration * 1000.0
            self.current_results["avg_vel"] = (self.current_results["max_ext_disp"] + self.current_results["max_flex_disp"]) / self.record_duration

            # 算完後更新介面
            self.update_result_labels()
            logging.info("反射指標運算完成")

        except Exception as e:
            logging.error(f"運算演算法發生錯誤: {e}")
            QMessageBox.critical(self, "演算法錯誤", "運算過程發生錯誤，請重新敲擊測試。")

    # --- 接收執行緒訊號 ---
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)))

    def update_angle(self, angle):
        self.current_angle = angle # 更新暫存供定時器抓取
        self.lbl_angle.setText(f"當前角度: {angle:.1f} deg")

    def update_sensors(self, data):
        if 'status' in data: self.lbl_serial_status.setText(f"感測器狀態: {data['status']}")
        if 'fsr' in data: 
            self.current_fsr = data['fsr']
            self.lbl_fsr.setText(f"FSR 電壓: {data['fsr']:.2f} V")
        if 'nano' in data: 
            self.current_nano = data['nano']
            self.lbl_nano.setText(f"Nano 電壓: {data['nano']:.2f} V")

    # --- 資料庫儲存 ---
    def save_to_database(self):
        pid = self.input_pid.text().strip()
        if not pid: return QMessageBox.warning(self, "警告", "請輸入病歷號！")
        
        # 避免在還沒算完就存檔
        if self.test_state != "DONE":
            return QMessageBox.warning(self, "警告", "請先完成一次敲擊測試再儲存！")

        self.btn_save.setEnabled(False); self.btn_save.setText("儲存中...")
        record = (pid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *self.current_results.values())
        self.db_thread = DatabaseSaveThread(record)
        self.db_thread.success_signal.connect(lambda p: (self.btn_save.setEnabled(True), self.btn_save.setText("💾 儲存測試紀錄"), QMessageBox.information(self, "成功", f"紀錄已儲存！")))
        self.db_thread.error_signal.connect(lambda e: (self.btn_save.setEnabled(True), self.btn_save.setText("💾 儲存測試紀錄"), QMessageBox.critical(self, "錯誤", f"儲存失敗：\n{e}")))
        self.db_thread.start()

    def toggle_system(self):
        if not self.video_thread.isRunning():
            self.video_thread._run_flag = True
            self.video_thread.start()
            self.sample_timer.start(20) # 啟動定時器 (每 20ms 收一次資料)
            self.test_state = "WAITING"
            self.lbl_test_state.setText("測試狀態: 🟡 等待敲擊 FSR...")
            self.lbl_test_state.setStyleSheet("font-size: 20px; font-weight: bold; color: black; background-color: #f1c40f; padding: 5px;")
            self.btn_start.setText("⏹ 停止監測")
            self.btn_start.setStyleSheet("font-size: 22px; font-weight: bold; background-color: #c0392b; color: white;")
        else:
            self.video_thread.stop()
            self.sample_timer.stop()
            self.test_state = "IDLE"
            self.lbl_test_state.setText("測試狀態: ⚪ 尚未開始")
            self.lbl_test_state.setStyleSheet("font-size: 20px; font-weight: bold; color: #f39c12; background-color: #333; padding: 5px;")
            self.btn_start.setText("▶ 開始監測 (進入等待敲擊模式)")
            self.btn_start.setStyleSheet("font-size: 22px; font-weight: bold; background-color: #27ae60; color: white;")

    def closeEvent(self, event):
        self.video_thread.stop(); self.serial_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())