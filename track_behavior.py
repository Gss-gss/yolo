import numpy as np
import cv2
from ultralytics import YOLO
import sys
import os
import json
from datetime import datetime

# =========================================================
# 🧠 ByteTrack：多目标跟踪核心
# =========================================================
sys.path.append(os.path.dirname(__file__))
from yolox.tracker.byte_tracker import BYTETracker


# =========================================================
# 📌 ByteTrack参数配置
# 👉 控制跟踪稳定性与灵敏度
# =========================================================
class BYTETrackerArgs:
    track_thresh: float = 0.25      # 低于该置信度不跟踪
    track_buffer: int = 30          # 丢失目标保留帧数（防遮挡丢ID）
    match_thresh: float = 0.8        # IOU匹配阈值
    aspect_ratio_thresh: float = 3.0 # 过滤异常框（防误检）
    min_box_area: float = 1.0        # 最小目标面积
    mot20: bool = False              # 是否开启密集场景优化


# =========================================================
# 🚧 禁区管理系统（支持多边形）
# 👉 用于“区域入侵检测”
# =========================================================
class ZoneManager:
    def __init__(self):
        self.zones = []      # 已完成的禁区
        self.current = []    # 正在绘制的禁区

    # 添加一个点（鼠标左键）
    #def add_point(self, x, y):
        #self.current.append([x, y])

    # 完成一个区域（鼠标右键）
    #def finish(self):
        #if len(self.current) >= 3:
            #self.zones.append(self.current.copy())
        #self.current = []

    # 绘制所有禁区
    def draw(self, frame):
        # 已完成禁区（黄色闭合多边形）
        for zone in self.zones:
            pts = np.array(zone, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        # 当前正在绘制（蓝色线）
        #if len(self.current) > 1:
            #pts = np.array(self.current, np.int32)
            #cv2.polylines(frame, [pts], False, (255, 255, 0), 2)

    # 判断点是否在禁区内
    def inside(self, point):
        for zone in self.zones:
            pts = np.array(zone, np.int32)
            if cv2.pointPolygonTest(pts, point, False) >= 0:
                return True
        return False

    # 保存禁区配置
    def save(self):
        with open("zones.json", "w") as f:
            json.dump(self.zones, f)

    # 加载禁区配置
    def load(self):
        if os.path.exists("zones.json"):
            with open("zones.json", "r") as f:
                self.zones = json.load(f)


# 初始化禁区管理器
zone_manager = ZoneManager()

# =========================================================
#  📌 新增：固定禁区初始化
# =========================================================
zone_manager.load()  # 可选：保留加载逻辑（若需要从文件读固定禁区）
# 重置并添加自定义固定禁区
zone_manager.zones = []
# 自定义640x480分辨率下的禁区（矩形）
fixed_zone = [[100, 100], [500, 100], [500, 400], [100, 400]]
zone_manager.zones.append(fixed_zone)

# =========================================================
# 🚨 风险评分系统（核心升级点）
# 👉 把多个行为统一成“风险分数”
# =========================================================
def compute_risk(intrusion, running, loitering):
    score = 0
    if intrusion:
        score += 50   # 入侵风险最高
    if running:
        score += 20   # 奔跑行为
    if loitering:
        score += 15   # 徘徊行为

    return min(score, 100)  # 最大100分


# =========================================================
# 📜 事件日志系统（用于比赛展示）
# 👉 记录所有异常行为
# =========================================================
def log_event(track_id, event, risk):
    log = {
        "id": int(track_id),
        "event": event,
        "risk": int(risk),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 写入日志文件（逐行JSON）
    with open("event_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")


# =========================================================
# 🖱 鼠标交互（绘制禁区）
# =========================================================
#def mouse_callback(event, x, y, flags, param):
    #if event == cv2.EVENT_LBUTTONDOWN:
        #zone_manager.add_point(x, y)
    #elif event == cv2.EVENT_RBUTTONDOWN:
        #zone_manager.finish()


# =========================================================
# 🧠 加载YOLO模型
# =========================================================
model = YOLO("yolov8n.pt")

 #打开摄像头
cap = cv2.VideoCapture(0)

#video_path = "person.mp4"

# cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 视频打开失败，请检查路径")
    exit()
else:
    print("✅ 视频加载成功")

# 设置分辨率（平衡性能）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =========================================================
# 🚀 ByteTrack初始化
# =========================================================
byte_tracker = BYTETracker(BYTETrackerArgs(), frame_rate=25)

# ========== 新增：初始化视频写入器 ==========
# 编码格式（mp4v对应mp4格式）
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 视频保存路径、编码、帧率（25帧）、分辨率（与视频源一致960x540）
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))

# =========================================================
# 📦 数据结构
# =========================================================
trajectories = {}   # 每个ID的轨迹
state = {}          # 每个ID的行为状态


# A=========================================================
# 📌 行为参数
# =========================================================
LOITER_TIME = 30      # 判断徘徊的时间窗口
LOITER_RADIUS = 60    # 活动范围阈值


# =========================================================
# 🪟 UI窗口
# =========================================================
#cv2.namedWindow("AI Surveillance V6")
#cv2.setMouseCallback("AI Surveillance V6", mouse_callback)

zone_manager.load()


# =========================================================
# 🔁 主循环（视频处理核心）
# =========================================================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # 镜像翻转（更符合人类视觉）
    frame = cv2.flip(frame, 1)

    # =====================================================
    # 🎯 YOLO目标检测
    # =====================================================
    results = model(frame, conf=0.25, imgsz=640)

    # 防止空检测崩溃
    if len(results[0].boxes) == 0:
        zone_manager.draw(frame)
        #cv2.imshow("AI Surveillance V6", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 转换检测结果
    outputs = results[0].boxes.data.cpu().numpy()

    # 只保留“人”
    outputs = outputs[outputs[:, 5] == 0]

    if len(outputs) == 0:
        zone_manager.draw(frame)
        #cv2.imshow("AI Surveillance V6", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    # =====================================================
    # 🚀 ByteTrack跟踪
    # =====================================================
    tracks = byte_tracker.update(outputs[:, :5], frame.shape, frame.shape)

    zone_manager.draw(frame)

    person_count = len(tracks)
    # ===================== 【新增1】异常统计初始化 =====================
    intrusion_cnt = 0
    running_cnt = 0
    loitering_cnt = 0
    abnormal_list = []
    # ==================================================================

    # =====================================================
    # 👤 遍历所有目标
    # =====================================================
    for track in tracks:

        try:
            box = track.tlbr
            track_id = track.track_id

            # 计算中心点
            center = (
                int((box[0] + box[2]) / 2),
                int((box[1] + box[3]) / 2)
            )

            # 初始化ID状态
            if track_id not in trajectories:
                trajectories[track_id] = []
                state[track_id] = {
                    "speed": [],
                    "run_frames": 0,
                    "loiter_frames": 0
                }

            # 保存轨迹
            trajectories[track_id].append(center)
            if len(trajectories[track_id]) > 50:
                trajectories[track_id].pop(0)


            # =================================================
            # 🚧 入侵检测（是否进入禁区）
            # =================================================
            intrusion = zone_manager.inside(center)


            # =================================================
            # 🏃 奔跑检测（速度变化）
            # =================================================
            running = False
            s = state[track_id]

            if len(trajectories[track_id]) >= 2:
                x1, y1 = trajectories[track_id][-2]
                x2, y2 = trajectories[track_id][-1]

                speed = np.hypot(x2 - x1, y2 - y1)

                s["speed"].append(speed)

                if len(s["speed"]) > 5:
                    s["speed"].pop(0)

                if np.mean(s["speed"]) > 15:
                    s["run_frames"] += 1
                else:
                    s["run_frames"] = 0

                running = s["run_frames"] > 5


            # =================================================
            # 🧍 徘徊检测（小范围停留）
            # =================================================
            loitering = False

            if len(trajectories[track_id]) > LOITER_TIME:

                pts = trajectories[track_id][-LOITER_TIME:]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]

                dx = max(xs) - min(xs)
                dy = max(ys) - min(ys)

                if dx < LOITER_RADIUS and dy < LOITER_RADIUS:
                    s["loiter_frames"] += 1
                else:
                    s["loiter_frames"] = 0

                loitering = s["loiter_frames"] > 10


            # =================================================
            # 🎯 风险评分（核心亮点）
            # =================================================
            risk = compute_risk(intrusion, running, loitering)

            # ===================== 【新增2】统计每个人异常 =====================
            behavior = []
            if intrusion:
                intrusion_cnt += 1
                behavior.append("禁区入侵")
            if running:
                running_cnt += 1
                behavior.append("奔跑")
            if loitering:
                loitering_cnt += 1
                behavior.append("徘徊")

            if behavior:
                abnormal_list.append(f"ID{track_id}：{'、'.join(behavior)}")
            # ==================================================================

            # =================================================
            # 🚨 事件输出 + 日志记录
            # =================================================
            if intrusion:
                cv2.putText(frame, "INTRUSION", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                log_event(track_id, "INTRUSION", risk)

            if running:
                cv2.putText(frame, "RUNNING", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                log_event(track_id, "RUNNING", risk)

            if loitering:
                cv2.putText(frame, "LOITERING", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                log_event(track_id, "LOITERING", risk)


            # =================================================
            # 📦 绘制检测框
            # =================================================
            cv2.rectangle(frame,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (143, 131, 226), 2)

            cv2.putText(frame,
                        f"ID:{track_id} Risk:{risk}",
                        (int(box[0]), int(box[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        except:
            # 防止单个目标异常导致整个系统崩溃
            continue

    # ===================== 【新增3】控制台打印异常汇总 =====================
    if abnormal_list:
        print("========================================================")
        print(f"📊 当前画面总人数：{person_count} 人")
        print(f"⚠️  异常统计：入侵{intrusion_cnt}人 | 奔跑{running_cnt}人 | 徘徊{loitering_cnt}人")
        for info in abnormal_list:
            print(f"   {info}")
        print("========================================================\n")
    # ======================================================================

    # =====================================================
    # 👥 人群检测
    # =====================================================
    if person_count > 5:
        cv2.putText(frame, "CROWD ALERT",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)


    # =====================================================
    # 🖥 显示画面
    # =====================================================
    #cv2.imshow("AI Surveillance V6", frame)
    out.write(frame)  # 新增：将处理后的帧写入视频文件


    # =====================================================
    # ⌨️ 控制
    # =====================================================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    #elif key == ord('s'):
    #    zone_manager.save()
    #    print("✅ 禁区已保存")

    #elif key == ord('c'):
    #    zone_manager.current = []


# =========================================================
# 🔚 释放资源
# =========================================================
cap.release()
#cv2.destroyAllWindows()
out.release()  # 新增：释放视频写入器
