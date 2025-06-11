import cv2
import numpy as np
import random
import time
import math
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

class EmotionFlappyGame:
    def __init__(self):
        # 加载模型
        self.detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        self.emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        
        # 游戏窗口设置
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 800
        
        # 游戏状态
        self.game_state = "intro"  # intro, playing, game_over
        self.score = 0
        self.high_score = 0
        self.start_time = None
        
        # 小鸟设置
        self.bird_x = 300
        self.bird_y = self.WINDOW_HEIGHT // 2
        self.bird_size = 80
        self.velocity = 0
        self.move_speed = 10
        self.rotation = 0
        self.bird_frame = 0  # 当前小鸟动画帧
        self.animation_time = 0  # 动画计时器
        self.animation_speed = 0.1  # 加快动画速度
        self.idle_animation_frame = 0  # 静止时的动画帧
        
        # 管道设置
        self.pipes = []
        self.PIPE_WIDTH = 120
        self.PIPE_GAP = 300  # 上下管道之间的间隙
        self.pipe_spawn_timer = 0
        self.PIPE_SPAWN_INTERVAL = 2.0  # 每2秒生成一个新管道
        self.PIPE_SPEED = 10
        
        # 加载游戏资源
        self.load_game_assets()
        
    def load_game_assets(self):
        """加载游戏资源"""
        # 加载背景
        self.background = cv2.imread("game_assets/game2/bg.png")  # 移除 IMREAD_UNCHANGED
        if self.background is None:
            self.background = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        self.background = cv2.resize(self.background, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        
        # 加载开始和结束画面
        self.start_screen = cv2.imread("game_assets/game2/start.png")  # 移除 IMREAD_UNCHANGED
        self.gameover_screen = cv2.imread("game_assets/game2/gameover.png")  # 移除 IMREAD_UNCHANGED
        
        # 加载管道图片
        self.pipe_img = cv2.imread("game_assets/game2/pipe.png", cv2.IMREAD_UNCHANGED)
        if self.pipe_img is None:
            self.pipe_img = np.zeros((100, self.PIPE_WIDTH, 4), dtype=np.uint8)
        
        # 加载小鸟动画帧
        self.bird_up_frames = []
        self.bird_down_frames = []
        
        # 加载向上飞行的帧
        for i in range(5):  # 0-4
            img = cv2.imread(f"game_assets/game2/{i}.png", cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.bird_up_frames.append(img)
        
        # 加载向下飞行的帧
        for i in range(5, 8):  # 5-7
            img = cv2.imread(f"game_assets/game2/{i}.png", cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.bird_down_frames.append(img)
                
        # 确保至少有一帧
        if not self.bird_up_frames:
            default_bird = np.zeros((self.bird_size, self.bird_size, 4), dtype=np.uint8)
            self.bird_up_frames = [default_bird]
        if not self.bird_down_frames:
            self.bird_down_frames = [self.bird_up_frames[0]]
        
    def create_game_background(self):
        """创建游戏背景"""
        return self.background.copy()
        
    def spawn_pipe(self):
        """生成新的管道"""
        current_time = time.time()
        if current_time - self.pipe_spawn_timer >= self.PIPE_SPAWN_INTERVAL:
            # 确定缺口的位置
            gap_y = random.randint(200, self.WINDOW_HEIGHT - 300)
            
            self.pipes.append({
                'x': self.WINDOW_WIDTH,
                'gap_y': gap_y,
                'passed': False
            })
            
            self.pipe_spawn_timer = current_time
            
    def update_pipes(self):
        """更新管道位置"""
        for pipe in self.pipes[:]:
            pipe['x'] -= self.PIPE_SPEED
            
            # 检查是否通过管道
            if not pipe['passed'] and pipe['x'] + self.PIPE_WIDTH < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                
            # 移除超出屏幕的管道
            if pipe['x'] < -self.PIPE_WIDTH:
                self.pipes.remove(pipe)
                
    def check_collision(self):
        """检查碰撞"""
        # 获取小鸟的碰撞箱
        bird_rect = {
            'x1': self.bird_x - self.bird_size // 3,
            'y1': self.bird_y - self.bird_size // 3,
            'x2': self.bird_x + self.bird_size // 3,
            'y2': self.bird_y + self.bird_size // 3
        }
        
        # 检查地面碰撞
        if bird_rect['y2'] > self.WINDOW_HEIGHT - 100:
            return True
            
        # 检查管道碰撞
        for pipe in self.pipes:
            # 上管道
            if (bird_rect['x2'] > pipe['x'] and
                bird_rect['x1'] < pipe['x'] + self.PIPE_WIDTH and
                bird_rect['y1'] < pipe['gap_y'] - self.PIPE_GAP // 2):
                return True
                
            # 下管道
            if (bird_rect['x2'] > pipe['x'] and
                bird_rect['x1'] < pipe['x'] + self.PIPE_WIDTH and
                bird_rect['y2'] > pipe['gap_y'] + self.PIPE_GAP // 2):
                return True
                
        return False
        
    def draw_pipe(self, display, pipe):
        """绘制管道"""
        # 调整管道大小以适应屏幕高度
        pipe_height = self.WINDOW_HEIGHT - 100  # 减去地面高度
        pipe_img = cv2.resize(self.pipe_img, (self.PIPE_WIDTH, pipe_height))
        
        # 上管道（翻转）
        top_height = pipe['gap_y'] - self.PIPE_GAP // 2
        if top_height > 0:
            # 从完整管道图片中裁剪出所需高度
            top_pipe = pipe_img[pipe_height-top_height:pipe_height]
            top_pipe = cv2.flip(top_pipe, 0)  # 垂直翻转
            self.overlay_image(display, top_pipe, (int(pipe['x']), 0))
        
        # 下管道
        bottom_y = pipe['gap_y'] + self.PIPE_GAP // 2
        bottom_height = self.WINDOW_HEIGHT - bottom_y
        if bottom_height > 0:
            # 从完整管道图片中裁剪出所需高度
            bottom_pipe = pipe_img[:bottom_height]
            self.overlay_image(display, bottom_pipe, (int(pipe['x']), bottom_y))
        
    def draw_bird(self, display):
        """绘制小鸟"""
        current_time = time.time()
        if self.start_time:
            self.animation_time += current_time - self.start_time
        self.start_time = current_time
        
        # 更新动画帧
        if self.animation_time >= self.animation_speed:
            self.bird_frame = (self.bird_frame + 1) % len(self.bird_up_frames)
            self.idle_animation_frame = (self.idle_animation_frame + 1) % 3  # 在前3帧之间循环
            self.animation_time = 0
        
        # 选择合适的帧
        if self.velocity < 0:  # 向上飞
            bird_img = self.bird_up_frames[self.bird_frame]
        elif self.velocity > 0:  # 向下飞
            bird_img = self.bird_down_frames[min(self.bird_frame, len(self.bird_down_frames)-1)]
        else:  # 静止时使用上飞动画的前3帧循环
            bird_img = self.bird_up_frames[self.idle_animation_frame]
        
        # 调整大小
        bird_img = cv2.resize(bird_img, (self.bird_size, self.bird_size))
        
        # 计算绘制位置
        x = int(self.bird_x - self.bird_size // 2)
        y = int(self.bird_y - self.bird_size // 2)
        
        # 绘制小鸟
        self.overlay_image(display, bird_img, (x, y))
        
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
        
    def draw_text(self, image, text, position, size=1.0, color=None, thickness=2):
        """绘制文本"""
        if color is None:
            color = (255, 255, 255)
            
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, size, thickness)
            
        # 绘制文本背景
        padding = 10
        bg_pts = np.array([
            [position[0] - padding, position[1] - text_height - padding],
            [position[0] + text_width + padding, position[1] - text_height - padding],
            [position[0] + text_width + padding, position[1] + padding],
            [position[0] - padding, position[1] + padding]
        ], np.int32)
        
        cv2.fillPoly(image, [bg_pts], (0, 0, 0, 150))
        
        # 绘制文本阴影
        cv2.putText(image, text,
                    (position[0] + 2, position[1] + 2),
                    font, size, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                    
        # 绘制主文本
        cv2.putText(image, text,
                    position, font, size, color, thickness, cv2.LINE_AA)
                    
    def render_game(self, frame, emotion, probability, face_coords):
        """渲染游戏画面"""
        # 总是先绘制背景
        display = self.create_game_background()
        
        if self.game_state == "intro":
            # 在背景上叠加开始画面
            if self.start_screen is not None:
                start_img = cv2.resize(self.start_screen, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                if start_img.shape[2] == 3:  # 如果开始画面是3通道
                    start_img_rgba = np.zeros((start_img.shape[0], start_img.shape[1], 4), dtype=np.uint8)
                    start_img_rgba[:, :, :3] = start_img
                    start_img_rgba[:, :, 3] = 180  # 设置半透明
                    start_img = start_img_rgba
                self.overlay_image(display, start_img, (0, 0))
                
        elif self.game_state == "playing":
            # 绘制所有管道
            for pipe in self.pipes:
                self.draw_pipe(display, pipe)
            
            # 绘制小鸟
            self.draw_bird(display)
            
            # 显示分数
            self.draw_text(display, f"Score: {self.score}",
                          (20, 30), 1.0, (255, 255, 255))
            
        elif self.game_state == "game_over":
            # 继续显示最后的游戏画面
            for pipe in self.pipes:
                self.draw_pipe(display, pipe)
            self.draw_bird(display)
            
            # 在游戏画面上叠加结束画面
            if self.gameover_screen is not None:
                gameover_img = cv2.resize(self.gameover_screen, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                if gameover_img.shape[2] == 3:  # 如果结束画面是3通道
                    gameover_img_rgba = np.zeros((gameover_img.shape[0], gameover_img.shape[1], 4), dtype=np.uint8)
                    gameover_img_rgba[:, :, :3] = gameover_img
                    gameover_img_rgba[:, :, 3] = 180  # 设置半透明
                    gameover_img = gameover_img_rgba
                self.overlay_image(display, gameover_img, (0, 0))
            
            # 显示分数
            self.draw_text(display, f"Score: {self.score}",
                          (self.WINDOW_WIDTH//2 - 80, self.WINDOW_HEIGHT//2 + 50),
                          1.5, (255, 255, 255))
            self.draw_text(display, f"High Score: {self.high_score}",
                          (self.WINDOW_WIDTH//2 - 100, self.WINDOW_HEIGHT//2 + 90))
        
        # 最后添加摄像头画面
        frame_height = 200
        frame_width = int(frame.shape[1] * frame_height / frame.shape[0])
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        
        # 创建一个与背景相同通道数的区域来放置视频帧
        frame_region = np.zeros((frame_height, frame_width, display.shape[2]), dtype=np.uint8)
        if display.shape[2] == 4:  # 如果背景是4通道
            frame_region[:, :, :3] = resized_frame
            frame_region[:, :, 3] = 255  # 设置完全不透明
        else:  # 如果背景是3通道
            frame_region = resized_frame
            
        x_offset = self.WINDOW_WIDTH - frame_width - 20
        y_offset = 20
        
        # 为视频添加边框
        cv2.rectangle(display,
                     (x_offset - 10, y_offset - 10),
                     (x_offset + frame_width + 10, y_offset + frame_height + 10),
                     (255, 255, 255), 2)
        
        # 确保区域大小正确
        if y_offset + frame_height <= display.shape[0] and x_offset + frame_width <= display.shape[1]:
            display[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame_region
        
        return display
        
    def overlay_image(self, background, foreground, position):
        """在背景上叠加图片"""
        if foreground is None or len(foreground.shape) < 3:
            return background
            
        x, y = position
        h, w = foreground.shape[:2]
        
        # 确保x和w不会超出背景图片范围
        if x < 0:
            w += x
            foreground = foreground[:, -x:]
            x = 0
        if x + w > background.shape[1]:
            w = background.shape[1] - x
            foreground = foreground[:, :w]
        
        # 确保y和h不会超出背景图片范围
        if y < 0:
            h += y
            foreground = foreground[-y:, :]
            y = 0
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            foreground = foreground[:h, :]
        
        # 如果裁剪后的尺寸小于等于0，直接返回背景
        if w <= 0 or h <= 0:
            return background
        
        if len(foreground.shape) == 3 and foreground.shape[2] == 4:
            # 处理透明通道
            alpha = foreground[:h, :w, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            rgb = foreground[:h, :w, :3]
            
            roi = background[y:y+h, x:x+w]
            try:
                background[y:y+h, x:x+w] = roi * (1 - alpha) + rgb * alpha
            except ValueError as e:
                print(f"Error in overlay_image: roi shape={roi.shape}, alpha shape={alpha.shape}, rgb shape={rgb.shape}")
                return background
        else:
            background[y:y+h, x:x+w] = foreground[:h, :w]
            
        return background
        
    def update_game(self, emotion, probability):
        """更新游戏状态"""
        if self.game_state == "playing":
            # 根据表情控制移动
            if emotion == "happy":
                self.bird_y -= self.move_speed
                self.velocity = -self.move_speed
            elif emotion == "neutral":
                self.bird_y += self.move_speed
                self.velocity = self.move_speed
            else:
                self.velocity = 0  # 无表情时速度为0
                
            # 限制小鸟的移动范围
            self.bird_y = max(self.bird_size, min(self.bird_y,
                            self.WINDOW_HEIGHT - 100 - self.bird_size))
            
            # 生成和更新管道
            self.spawn_pipe()
            self.update_pipes()
            
            # 检查碰撞
            if self.check_collision():
                self.game_state = "game_over"
                if self.score > self.high_score:
                    self.high_score = self.score
                    
    def handle_key(self, key):
        """处理按键输入"""
        if key == ord(' '):  # 空格键
            if self.game_state == "intro":
                self.game_state = "playing"
                self.start_time = time.time()
                self.score = 0
                self.pipes.clear()
                self.bird_y = self.WINDOW_HEIGHT // 2
                self.velocity = 0
            elif self.game_state == "game_over":
                self.game_state = "intro"
                
        return key == 27  # ESC键退出
        
    def run(self):
        """运行游戏"""
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Emotion Flappy', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
                
            # 处理视频帧
            frame, emotion, probability, face_coords = self.process_frame(frame)
            
            # 更新游戏状态
            self.update_game(emotion, probability)
            
            # 渲染游戏画面
            display = self.render_game(frame, emotion, probability, face_coords)
            
            # 显示游戏画面
            cv2.imshow('Emotion Flappy', display)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if self.handle_key(key):
                break
                
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = EmotionFlappyGame()
    game.run() 