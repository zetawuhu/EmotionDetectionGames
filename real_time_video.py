from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from emotion_utils import EmotionStats
import keyboard

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# 初始化模型
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# 定义情绪对应的颜色
EMOTION_COLORS = {
    "angry": (0, 0, 255),     # 红色
    "disgust": (0, 128, 0),   # 深绿色
    "scared": (128, 0, 128),  # 紫色
    "happy": (0, 255, 255),   # 黄色
    "sad": (139, 69, 19),     # 棕色
    "surprised": (255, 140, 0),# 橙色
    "neutral": (128, 128, 128) # 灰色
}

# 初始化情绪统计
emotion_stats = EmotionStats()


# 创建一个漂亮的标题栏
def create_title_bar(width, height, title):
    title_bar = np.zeros((height, width, 3), dtype="uint8")
    # 创建渐变背景
    for i in range(width):
        color = (40 + (i / width) * 40, 40 + (i / width) * 40, 40 + (i / width) * 40)
        cv2.line(title_bar, (i, 0), (i, height), color, 1)
    
    # 添加标题文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # 使用抗锯齿字体渲染
    cv2.putText(title_bar, title, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)  # 阴影
    cv2.putText(title_bar, title, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return title_bar


# 创建一个美化的概率条显示
def create_probability_bar(emotion, probability, width, height, y_offset):
    bar = np.zeros((height, width, 3), dtype="uint8")
    # 绘制背景
    cv2.rectangle(bar, (0, 0), (width, height), (40, 40, 40), -1)
    # 绘制概率条
    w = int(probability * (width - 20))
    color = EMOTION_COLORS[emotion]
    cv2.rectangle(bar, (10, 10), (10 + w, height - 10), color, -1)
    
    # 添加文字
    text = f"{emotion}: {probability * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_y = (height + text_size[1]) // 2
    
    # 添加文字阴影和抗锯齿效果
    cv2.putText(bar, text, (16, text_y), font, font_scale, (0, 0, 0), thickness + 1)  # 阴影
    cv2.putText(bar, text, (15, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return bar

# 添加按键说明
def add_key_instructions(display, height):
    instructions = [
        "Press 'R' to generate report",
        "Press 'S' to save session",
        "Press 'Q' to quit"
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 10
    for i, text in enumerate(instructions):
            cv2.putText(display, text, 
                   (padding, height - (len(instructions) - i) * 25), 
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


    return display

# 启动视频捕捉
cv2.namedWindow('Emotion Recognition', cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(0)

# 设置窗口大小
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 900
TITLE_HEIGHT = 60
PROB_BAR_HEIGHT = 45
SPACING = 6

# 视频帧尺寸
FRAME_HEIGHT = 380

while True:
    frame = camera.read()[1]
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # 创建主显示画布
    display = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype="uint8")
    
    # 添加标题栏
    title_bar = create_title_bar(WINDOW_WIDTH, TITLE_HEIGHT, "Emotion Recognition System")
    display[0:TITLE_HEIGHT] = title_bar

    # 处理视频帧
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        
        # 更新情绪统计
        emotion_stats.update_stats(preds)
        
        # 绘制带颜色的人脸框和标签
        color = EMOTION_COLORS[label]
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), color, 2)
        
        # 添加带背景的标签，使用抗锯齿文字
        label_background = np.zeros((40, 150, 3), dtype="uint8")
        cv2.rectangle(label_background, (0, 0), (150, 40), color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (150 - text_size[0]) // 2
        text_y = (40 + text_size[1]) // 2
        cv2.putText(label_background, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(label_background, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        try:
            frame[max(0, fY-50):max(0, fY-10), max(0, fX):max(0, fX+150)] = label_background
        except:
            pass
            
        # 添加表情符号
        frame = emotion_stats.overlay_emoji(frame, label, (frame.shape[1] - 120, 20))
    else:
        preds = np.zeros(len(EMOTIONS))

    # 将视频帧放入显示画布
    frame_height = FRAME_HEIGHT
    frame_width = int(frame.shape[1] * frame_height / frame.shape[0])
    resized_frame = cv2.resize(frame, (frame_width, frame_height))
    x_offset = (WINDOW_WIDTH - frame_width) // 2
    display[TITLE_HEIGHT + SPACING:TITLE_HEIGHT + SPACING + frame_height, 
            x_offset:x_offset + frame_width] = resized_frame

    # 添加概率条
    y_offset = TITLE_HEIGHT + SPACING + frame_height + SPACING
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
        prob_bar = create_probability_bar(emotion, prob, WINDOW_WIDTH - 40, PROB_BAR_HEIGHT, y_offset)
        current_y = y_offset + i * (PROB_BAR_HEIGHT + SPACING)
        if current_y + PROB_BAR_HEIGHT <= WINDOW_HEIGHT - SPACING:
            display[current_y:current_y + PROB_BAR_HEIGHT, 20:WINDOW_WIDTH - 20] = prob_bar

    # 添加按键说明
    add_key_instructions(display, WINDOW_HEIGHT)


    # 显示结果
    cv2.imshow('Emotion Recognition', display)
    
    # 检查按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        report_path = emotion_stats.generate_report()
        if report_path:
            print(f"Report generated: {report_path}")
    elif key == ord('s'):
        emotion_stats.save_session_data()
        print("Session data saved")

camera.release()
cv2.destroyAllWindows()
