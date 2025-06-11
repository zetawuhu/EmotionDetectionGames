# 按 'R' 键生成统计报告
# 按 'S' 键保存会话数据
# 按 'Q' 键退出程序
import numpy as np
import cv2
from datetime import datetime
import json
import os
from collections import deque
import matplotlib.pyplot as plt

class EmotionStats:
    def __init__(self):
        self.emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        self.emotion_history = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_length = 100  # 保存最近100帧的情绪数据
        self.emotion_queue = deque(maxlen=self.history_length)
        
        # 创建数据存储目录
        self.data_dir = "emotion_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # 加载表情符号
        self.emojis = {}
        self.load_emojis()
    
    def load_emojis(self):
        """加载表情符号图片"""
        emoji_dir = "emojis"
        for emotion in self.emotions:
            emoji_path = os.path.join(emoji_dir, f"{emotion}.png")
            if os.path.exists(emoji_path):
                emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                if emoji is not None:
                    self.emojis[emotion] = emoji
    
    def update_stats(self, emotion_probs):
        """更新情绪统计数据"""
        timestamp = datetime.now()
        max_emotion = self.emotions[np.argmax(emotion_probs)]
        
        # 添加到历史记录
        record = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "emotions": {emotion: float(prob) for emotion, prob in zip(self.emotions, emotion_probs)},
            "dominant_emotion": max_emotion
        }
        self.emotion_history.append(record)
        self.emotion_queue.append(record)
    
    def save_session_data(self):
        """保存会话数据"""
        filename = os.path.join(self.data_dir, f"session_{self.current_session}.json")
        with open(filename, 'w') as f:
            json.dump(self.emotion_history, f, indent=2)
    
    def generate_report(self):
        """生成情绪统计报告"""
        if not self.emotion_history:
            return None
        
        # 创建报告图表
        fig = plt.figure(figsize=(15, 10))
        
        # 情绪分布饼图
        ax1 = fig.add_subplot(221)
        emotion_counts = {}
        for record in self.emotion_history:
            emotion = record["dominant_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        labels = list(emotion_counts.keys())
        sizes = list(emotion_counts.values())
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax1.set_title('Emotion Distribution')
        
        # 情绪变化趋势图
        ax2 = fig.add_subplot(222)
        timestamps = [i for i in range(len(self.emotion_queue))]
        for emotion in self.emotions:
            values = [record["emotions"][emotion] for record in self.emotion_queue]
            ax2.plot(timestamps, values, label=emotion)
        
        ax2.set_title('Emotion Trends')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend()
        
        # 保存报告
        report_path = os.path.join(self.data_dir, f"report_{self.current_session}.png")
        plt.savefig(report_path)
        plt.close()
        
        return report_path
    
    def overlay_emoji(self, frame, emotion, position):
        """在视频帧上叠加表情符号"""
        if emotion not in self.emojis:
            return frame
        
        emoji = self.emojis[emotion]
        
        # 调整表情符号大小
        emoji_size = 100
        emoji = cv2.resize(emoji, (emoji_size, emoji_size))
        
        # 获取表情符号的位置
        x, y = position
        
        # 确保位置在框架内
        if x + emoji_size > frame.shape[1]:
            x = frame.shape[1] - emoji_size
        if y + emoji_size > frame.shape[0]:
            y = frame.shape[0] - emoji_size
            
        # 如果表情符号有alpha通道
        if emoji.shape[2] == 4:
            alpha = emoji[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            rgb = emoji[:, :, :3]
            
            # 在原始帧上叠加表情符号
            roi = frame[y:y+emoji_size, x:x+emoji_size]
            frame[y:y+emoji_size, x:x+emoji_size] = roi * (1 - alpha) + rgb * alpha
            
        return frame 