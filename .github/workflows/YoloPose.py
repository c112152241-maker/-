# -*- coding: utf-8 -*-
import cv2
import math
from ultralytics import YOLO

class Yolo:
    def __init__(self):
        print("🤖 系統提示: 正在初始化 YOLOv8 姿態辨識模型...")
        # 🌟 這裡會自動從網路下載最新版的模型，不需要手動處理檔案！
        self.model = YOLO('yolo26x-pose.pt') 

    def inference(self, frame):
        # 執行姿態辨識 (設定 verbose=False 讓終端機保持乾淨)
        results = self.model(frame, verbose=False)
        
        kpts_list = []
        # 安全檢查：確保畫面中有偵測到人，且有關節點資料
        if len(results) > 0 and results[0].keypoints is not None:
            xy = results[0].keypoints.xy
            if len(xy) > 0:
                # 取得畫面中第一個人的 17 個關鍵點座標
                keypoints = xy[0].cpu().numpy()
                for kp in keypoints:
                    kpts_list.append((int(kp[0]), int(kp[1])))
            
        return frame, kpts_list

    def draw(self, frame):
        # 使用 Ultralytics 內建的超強畫圖功能，直接畫出彩色骨架
        results = self.model(frame, verbose=False)
        if len(results) > 0:
            return results[0].plot()
        return frame

    def get_angle(self, p1, p2, p3):
        # 計算夾角 (p2為頂點，也就是膝蓋)
        if p1 == (0,0) or p2 == (0,0) or p3 == (0,0):
            return None
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle