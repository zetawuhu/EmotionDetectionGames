import cv2
import numpy as np
import random
import time
import math
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

class EmotionNinjaGame:
    def __init__(self):
        # 加载模型
        self.detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        self.emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        
        # 游戏设置
        self.current_level = 1
        self.max_levels = 5
        self.score = 0
        self.time_limit = 5  # 每个表情的时间限制（秒）
        self.success_threshold = 0.2  # 表情识别阈值
        
        # 游戏状态
        self.game_state = "intro"  # intro, playing, success, failed, complete
        self.target_emotion = None
        self.start_time = None
        self.remaining_time = 0
        
        # 加载忍者图片
        self.ninja_images = {
            "normal": cv2.imread("game_assets/game1/ninja_normal.png", cv2.IMREAD_UNCHANGED),
            "attack": cv2.imread("game_assets/game1/ninja_attack.png", cv2.IMREAD_UNCHANGED),
            "hit": cv2.imread("game_assets/game1/ninja_hit.png", cv2.IMREAD_UNCHANGED)
        }
        
        # 游戏场景设置
        self.WINDOW_WIDTH = 1000
        self.WINDOW_HEIGHT = 900
        
        # 关卡设置
        self.level_emotions = {
            1: ["happy"],  # 第一关：开心
            2: ["angry"],    # 第二关：生气
            3: ["scared", "sad"],   # 第三关：害怕和悲伤
            4: ["happy", "angry","scared"],  # 第四关：混合
            5: ["angry", "scared", "happy"]  # 第五关：终极挑战
        }
        
        # 添加奔跑动画相关的属性
        self.runner_x = 100  # 小人的初始X坐标
        self.runner_y = 700  # 小人的Y坐标
        self.runner_speed = 40  # 小人的移动速度
        self.target_x = 800  # 终点X坐标
        self.animation_frame = 0  # 动画帧计数器
        self.animation_speed = 0.2  # 动画速度
        self.is_running = False  # 是否在奔跑
        
        # 加载奔跑者图片
        self.runner_images = {
            "idle": cv2.imread("game_assets/game1/runner_idle.png", cv2.IMREAD_UNCHANGED),
            "run1": cv2.imread("game_assets/game1/runner_run1.png", cv2.IMREAD_UNCHANGED),
            "run2": cv2.imread("game_assets/game1/runner_run2.png", cv2.IMREAD_UNCHANGED),
            "success": cv2.imread("game_assets/game1/runner_success.png", cv2.IMREAD_UNCHANGED),
            "fail": cv2.imread("game_assets/game1/runner_fail.png", cv2.IMREAD_UNCHANGED)
        }
        
        # 加载背景元素
        self.background_elements = {
            "tree": cv2.imread("game_assets/game1/tree.png", cv2.IMREAD_UNCHANGED),
            "cloud": cv2.imread("game_assets/game1/cloud.png", cv2.IMREAD_UNCHANGED)
        }
        
        # 添加星星特效系统
        self.stars = []
        self.star_colors = [
            (255, 215, 0),  # 金色
            (255, 255, 0),  # 黄色
            (255, 200, 0),  # 橙色
        ]
        self.celebration_start_time = None
        self.celebration_duration = 2.0  # 总持续时间改为2秒
        self.flash_duration = 0.5  # 闪光持续0.5秒
        
    def create_game_background(self):
        """创建游戏背景"""
        background = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype="uint8")
        # 添加渐变背景
        for i in range(self.WINDOW_HEIGHT):
            color = (
                int(20 + (i / self.WINDOW_HEIGHT) * 40),
                int(10 + (i / self.WINDOW_HEIGHT) * 30),
                int(40 + (i / self.WINDOW_HEIGHT) * 20)
            )
            cv2.line(background, (0, i), (self.WINDOW_WIDTH, i), color, 1)
        return background
    
    def overlay_image(self, background, foreground, position):
        """在背景上叠加图片"""
        if foreground is None or len(foreground.shape) < 3:
            return background
            
        x, y = position
        if len(foreground.shape) == 3:
            h, w, _ = foreground.shape
        else:
            h, w = foreground.shape
            
        if x + w > background.shape[1] or y + h > background.shape[0]:
            return background
            
        if len(foreground.shape) == 3 and foreground.shape[2] == 4:
            # 处理透明通道
            alpha = foreground[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            rgb = foreground[:, :, :3]
            
            roi = background[y:y+h, x:x+w]
            background[y:y+h, x:x+w] = roi * (1 - alpha) + rgb * alpha
        else:
            background[y:y+h, x:x+w] = foreground
            
        return background
        
    def draw_text(self, image, text, position, font_scale=1.0, color=(255, 255, 255), thickness=2):
        """绘制带阴影的文本"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 绘制阴影
        cv2.putText(image, text, (position[0]+2, position[1]+2), font, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
        # 绘制主文本
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        
    def draw_progress_bar(self, image, progress, position, size=(400, 30), color=(0, 255, 0)):
        """绘制进度条"""
        x, y = position
        width, height = size
        
        # 绘制背景
        cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        # 绘制进度
        progress_width = int(width * progress)
        cv2.rectangle(image, (x, y), (x + progress_width, y + height), color, -1)
        
        # 绘制边框
        cv2.rectangle(image, (x, y), (x + width, y + height), (200, 200, 200), 2)
        
    def process_frame(self, frame):
        """处理视频帧并返回检测到的情绪"""
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            emotion = self.EMOTIONS[preds.argmax()]
            
            return frame, emotion, emotion_probability, (fX, fY, fW, fH)
        
        return frame, None, 0.0, None
        
    def update_runner_animation(self):
        """更新奔跑者的动画状态"""
        if self.game_state == "playing" and self.is_running:
            self.animation_frame += self.animation_speed
            if self.animation_frame >= 2:
                self.animation_frame = 0
            
            # 移动小人
            if self.runner_x < self.target_x:
                self.runner_x += self.runner_speed
                
    def get_current_runner_image(self):
        """获取当前的奔跑者图片"""
        if self.game_state == "success":
            return self.runner_images["success"]
        elif self.game_state == "failed":
            return self.runner_images["fail"]
        elif self.is_running:
            frame = int(self.animation_frame)
            return self.runner_images[f"run{frame+1}"]
        return self.runner_images["idle"]
        
    def draw_game_scene(self, display):
        """绘制游戏场景"""
        # 绘制背景元素
        for i in range(3):  # 绘制多棵树
            x = 200 + i * 300
            self.overlay_image(display, self.background_elements["tree"], (x, 650))
            
        # 绘制云
        for i in range(2):
            x = 100 + i * 600
            self.overlay_image(display, self.background_elements["cloud"], (x, 100))
            
        # 绘制跑道背景
        track_y = 800
        track_start_x = 50
        track_end_x = self.WINDOW_WIDTH - 50
        track_width = 30
        
        # 绘制跑道底色（灰色）
        cv2.rectangle(display, 
                     (track_start_x, track_y - track_width//2),
                     (track_end_x, track_y + track_width//2),
                     (100, 100, 100), -1)
        
        # 绘制跑道刻度
        num_segments = 10
        segment_width = (track_end_x - track_start_x) // num_segments
        for i in range(num_segments + 1):
            x = track_start_x + i * segment_width
            cv2.line(display, (x, track_y - track_width//2),
                    (x, track_y + track_width//2), (200, 200, 200), 2)
        
        # 计算已跑过的距离
        progress = (self.runner_x - 100) / (self.target_x - 100)
        progress_x = int(track_start_x + progress * (track_end_x - track_start_x))
        
        # 绘制已跑过的部分（绿色渐变）
        if progress > 0:
            for i in range(track_start_x, progress_x):
                # 创建从深绿到浅绿的渐变
                factor = (i - track_start_x) / (track_end_x - track_start_x)
                green_color = (
                    int(50 + factor * 50),    # B
                    int(200 - factor * 50),   # G
                    int(50 + factor * 50)     # R
                )
                cv2.line(display, 
                        (i, track_y - track_width//2),
                        (i, track_y + track_width//2),
                        green_color, 1)
        
        # 绘制跑道边框
        cv2.rectangle(display, 
                     (track_start_x, track_y - track_width//2),
                     (track_end_x, track_y + track_width//2),
                     (200, 200, 200), 2)
        
        # 绘制起点和终点标记
        cv2.putText(display, "START", (track_start_x - 20, track_y - track_width),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "FINISH", (track_end_x - 40, track_y - track_width),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 在跑道上方显示完成百分比
        percentage = int(progress * 100)
        cv2.putText(display, f"{percentage}%",
                   (progress_x - 20, track_y - track_width - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 绘制当前位置指示器（小三角形）
        triangle_pts = np.array([
            [progress_x, track_y - track_width//2 - 15],
            [progress_x - 10, track_y - track_width//2 - 25],
            [progress_x + 10, track_y - track_width//2 - 25]
        ], np.int32)
        cv2.fillPoly(display, [triangle_pts], (0, 255, 255))
        
        # 绘制奔跑者
        runner_img = self.get_current_runner_image()
        if runner_img is not None:
            self.overlay_image(display, runner_img, (int(self.runner_x), self.runner_y))
            
    def create_stars(self):
        """创建庆祝用的星星"""
        self.stars = []
        for _ in range(30):  # 创建30颗星星
            x = random.randint(0, self.WINDOW_WIDTH)
            y = random.randint(0, self.WINDOW_HEIGHT)
            size = random.randint(10, 30)
            color = random.choice(self.star_colors)
            speed = random.uniform(1, 3)
            self.stars.append({
                'x': x,
                'y': y,
                'size': size,
                'color': color,
                'speed': speed,
                'angle': random.uniform(0, 360)
            })
            
    def draw_star(self, image, x, y, size, color, angle=0):
        """绘制一个星星"""
        # 创建星星的点
        points = []
        for i in range(10):
            curr_angle = angle + i * 36  # 360/10 = 36度
            curr_size = size if i % 2 == 0 else size * 0.4
            rad = math.radians(curr_angle)
            px = x + curr_size * math.cos(rad)
            py = y + curr_size * math.sin(rad)
            points.append([int(px), int(py)])
            
        # 绘制填充的星星
        points = np.array(points, np.int32)
        cv2.fillPoly(image, [points], color)
        
    def update_stars(self):
        """更新星星的位置和角度"""
        for star in self.stars:
            star['angle'] += star['speed']
            if star['angle'] >= 360:
                star['angle'] -= 360
                
    def draw_celebration(self, display):
        """绘制庆祝效果"""
        if not self.celebration_start_time:
            return
            
        current_time = time.time()
        elapsed = current_time - self.celebration_start_time
        
        if elapsed > self.celebration_duration:
            return
            
        # 更新和绘制所有星星
        self.update_stars()
        for star in self.stars:
            self.draw_star(
                display,
                star['x'],
                star['y'],
                star['size'],
                star['color'],
                star['angle']
            )
            
        # 只在开始的flash_duration时间内显示闪光效果
        if elapsed < self.flash_duration:
            # 计算闪光强度，从1渐变到0
            flash_alpha = max(0, 1 - (elapsed / self.flash_duration))
            overlay = display.copy()
            cv2.addWeighted(
                overlay,
                flash_alpha * 0.1,  # 控制闪光强度
                np.full_like(overlay, 255),
                flash_alpha * 0.1,
                0,
                display
            )
        
    def update_game_state(self, emotion, probability):
        """更新游戏状态"""
        if self.game_state == "playing":
            current_time = time.time()
            self.remaining_time = max(0, self.time_limit - (current_time - self.start_time))
            
            # 更新奔跑状态
            self.is_running = (emotion == self.target_emotion and probability > self.success_threshold)
            
            # 更新动画
            self.update_runner_animation()
            
            # 检查是否到达终点或时间耗尽
            if self.runner_x >= self.target_x:
                self.score += int((self.remaining_time * 100))
                self.game_state = "success"
                self.celebration_start_time = time.time()
                self.create_stars()
            elif self.remaining_time <= 0:
                self.game_state = "failed"
                
    def get_next_emotion(self):
        return random.choice(self.level_emotions[self.current_level])
        
    def start_new_round(self):
        self.target_emotion = self.get_next_emotion()
        self.start_time = time.time()
        self.game_state = "playing"
        self.runner_x = 100  
        self.is_running = False
        self.animation_frame = 0
        
    def render_game(self, frame, emotion, probability, face_coords):
        """渲染游戏画面"""
        # 创建游戏画布
        display = self.create_game_background()
        
        # 添加视频帧
        frame_height = 300
        frame_width = int(frame.shape[1] * frame_height / frame.shape[0])
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        x_offset = (self.WINDOW_WIDTH - frame_width) // 2
        
        # 为视频框添加边框
        cv2.rectangle(display, 
                     (x_offset-10, 40),
                     (x_offset+frame_width+10, 50+frame_height+10),
                     (200, 200, 200), 2)
        
        display[50:50+frame_height, x_offset:x_offset+frame_width] = resized_frame
        
        # 绘制游戏场景
        self.draw_game_scene(display)
        
        # 绘制庆祝效果
        self.draw_celebration(display)
        
        if self.game_state == "intro":
            self.draw_text(display, "Emotion Ninja Challenge", (200, 400), 1.5, (255, 255, 0))
            self.draw_text(display, "Press SPACE to Start", (200, 450))
            self.draw_text(display, f"Current Level: {self.current_level}", (200,500))
            
        elif self.game_state == "playing":
            # 显示目标表情和剩余时间
            self.draw_text(display, f"Target Emotion: {self.target_emotion}", (50, 500), 1.2)
            self.draw_text(display, f"Time Left: {self.remaining_time:.1f}s", (50, 550), 1.2)
            self.draw_text(display, f"Score: {self.score}", (50, 600), 1.2)
            
        elif self.game_state == "success":
            self.draw_text(display, "Success!", (350,450), 2.0, (0, 255, 0))
            self.draw_text(display, f"Score: +{int(self.remaining_time * 100)}", (350,500))
            self.draw_text(display, "Press SPACE to Continue", (350,550))
            
        elif self.game_state == "failed":
            self.draw_text(display, "Failed!", (400,450), 2.0, (255, 0, 0))
            self.draw_text(display, "Press SPACE to Retry", (350,500))
            
        elif self.game_state == "complete":
            self.draw_text(display, "Congratulations!", (350,450), 2.0, (255, 255, 0))
            self.draw_text(display, f"Final Score: {self.score}", (350,500))
            self.draw_text(display, "Press ESC to Exit", (350,550))
            
        # 显示当前关卡
        self.draw_text(display, f"Level {self.current_level}/{self.max_levels}", (800, 50))
        
        return display
        
    def handle_key(self, key):
        """处理按键输入"""
        if key == ord(' '):  # 空格键
            if self.game_state == "intro":
                self.start_new_round()
            elif self.game_state == "success":
                if self.current_level < self.max_levels:
                    self.current_level += 1
                    self.game_state = "intro"
                else:
                    self.game_state = "complete"
            elif self.game_state == "failed":
                self.game_state = "intro"
                
        return key == 27  # ESC键退出
        
    def run(self):
        """运行游戏"""
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Emotion Ninja', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
                
            # 处理视频帧
            frame, emotion, probability, face_coords = self.process_frame(frame)
            
            # 更新游戏状态
            self.update_game_state(emotion, probability)
            
            # 渲染游戏画面
            display = self.render_game(frame, emotion, probability, face_coords)
            
            # 显示游戏画面
            cv2.imshow('Emotion Ninja', display)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if self.handle_key(key):
                break
                
        camera.release()
        cv2.destroyAllWindows() 