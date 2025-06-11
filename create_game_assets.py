from PIL import Image, ImageDraw
import os

def create_directory():
    if not os.path.exists('game_assets'):
        os.makedirs('game_assets')

def create_runner_idle():
    # 创建100x100的图片，带透明背景
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 角色配色
    body_color = (65, 105, 225)  # 蓝色
    head_color = (255, 218, 185)  # 肤色
    detail_color = (25, 25, 112)  # 深蓝色
    
    # 绘制身体
    draw.rectangle([(40, 40), (60, 70)], fill=body_color)  # 躯干
    draw.rectangle([(35, 70), (45, 90)], fill=body_color)  # 左腿
    draw.rectangle([(55, 70), (65, 90)], fill=body_color)  # 右腿
    draw.rectangle([(30, 45), (40, 65)], fill=body_color)  # 左臂
    draw.rectangle([(60, 45), (70, 65)], fill=body_color)  # 右臂
    
    # 绘制头部
    draw.ellipse([(35, 20), (65, 50)], fill=head_color)  # 头
    
    # 绘制细节
    draw.ellipse([(42, 30), (48, 36)], fill=detail_color)  # 左眼
    draw.ellipse([(52, 30), (58, 36)], fill=detail_color)  # 右眼
    draw.arc([(45, 35), (55, 45)], 0, 180, fill=detail_color, width=2)  # 微笑
    
    img.save('game_assets/runner_idle.png')

def create_runner_run1():
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    body_color = (65, 105, 225)
    head_color = (255, 218, 185)
    detail_color = (25, 25, 112)
    
    # 跑步姿势1
    draw.rectangle([(40, 40), (60, 70)], fill=body_color)  # 躯干
    draw.rectangle([(35, 60), (45, 85)], fill=body_color)  # 左腿（抬起）
    draw.rectangle([(55, 75), (65, 95)], fill=body_color)  # 右腿（后伸）
    draw.rectangle([(25, 45), (35, 65)], fill=body_color)  # 左臂（前伸）
    draw.rectangle([(65, 45), (75, 65)], fill=body_color)  # 右臂（后伸）
    
    draw.ellipse([(35, 20), (65, 50)], fill=head_color)  # 头
    draw.ellipse([(42, 30), (48, 36)], fill=detail_color)  # 左眼
    draw.ellipse([(52, 30), (58, 36)], fill=detail_color)  # 右眼
    draw.arc([(45, 35), (55, 45)], 0, 180, fill=detail_color, width=2)  # 微笑
    
    img.save('game_assets/runner_run1.png')

def create_runner_run2():
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    body_color = (65, 105, 225)
    head_color = (255, 218, 185)
    detail_color = (25, 25, 112)
    
    # 跑步姿势2
    draw.rectangle([(40, 40), (60, 70)], fill=body_color)  # 躯干
    draw.rectangle([(35, 75), (45, 95)], fill=body_color)  # 左腿（后伸）
    draw.rectangle([(55, 60), (65, 85)], fill=body_color)  # 右腿（抬起）
    draw.rectangle([(65, 45), (75, 65)], fill=body_color)  # 左臂（后伸）
    draw.rectangle([(25, 45), (35, 65)], fill=body_color)  # 右臂（前伸）
    
    draw.ellipse([(35, 20), (65, 50)], fill=head_color)  # 头
    draw.ellipse([(42, 30), (48, 36)], fill=detail_color)  # 左眼
    draw.ellipse([(52, 30), (58, 36)], fill=detail_color)  # 右眼
    draw.arc([(45, 35), (55, 45)], 0, 180, fill=detail_color, width=2)  # 微笑
    
    img.save('game_assets/runner_run2.png')

def create_runner_success():
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    body_color = (65, 105, 225)
    head_color = (255, 218, 185)
    detail_color = (25, 25, 112)
    
    # 胜利姿势
    draw.rectangle([(40, 40), (60, 70)], fill=body_color)  # 躯干
    draw.rectangle([(35, 70), (45, 90)], fill=body_color)  # 左腿
    draw.rectangle([(55, 70), (65, 90)], fill=body_color)  # 右腿
    draw.rectangle([(25, 35), (35, 55)], fill=body_color)  # 左臂（举起）
    draw.rectangle([(65, 35), (75, 55)], fill=body_color)  # 右臂（举起）
    
    draw.ellipse([(35, 20), (65, 50)], fill=head_color)  # 头
    draw.ellipse([(42, 30), (48, 36)], fill=detail_color)  # 左眼
    draw.ellipse([(52, 30), (58, 36)], fill=detail_color)  # 右眼
    draw.arc([(45, 32), (55, 45)], 0, 180, fill=detail_color, width=3)  # 大笑
    
    # 添加星星
    def draw_star(x, y, size):
        points = [(x, y-size), (x+size/3, y-size/3), (x+size, y),
                 (x+size/3, y+size/3), (x, y+size), (x-size/3, y+size/3),
                 (x-size, y), (x-size/3, y-size/3)]
        draw.polygon(points, fill=(255, 215, 0))
    
    draw_star(20, 20, 10)
    draw_star(80, 20, 10)
    draw_star(50, 10, 10)
    
    img.save('game_assets/runner_success.png')

def create_runner_fail():
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    body_color = (65, 105, 225)
    head_color = (255, 218, 185)
    detail_color = (25, 25, 112)
    
    # 失败姿势（坐在地上）
    draw.rectangle([(40, 50), (60, 80)], fill=body_color)  # 躯干
    draw.rectangle([(35, 80), (45, 85)], fill=body_color)  # 左腿
    draw.rectangle([(55, 80), (65, 85)], fill=body_color)  # 右腿
    draw.rectangle([(30, 55), (40, 75)], fill=body_color)  # 左臂
    draw.rectangle([(60, 55), (70, 75)], fill=body_color)  # 右臂
    
    draw.ellipse([(35, 30), (65, 60)], fill=head_color)  # 头
    draw.line([(42, 45), (48, 39)], fill=detail_color, width=2)  # 左眼（X）
    draw.line([(42, 39), (48, 45)], fill=detail_color, width=2)
    draw.line([(52, 45), (58, 39)], fill=detail_color, width=2)  # 右眼（X）
    draw.line([(52, 39), (58, 45)], fill=detail_color, width=2)
    draw.arc([(45, 45), (55, 55)], 180, 0, fill=detail_color, width=2)  # 沮丧的嘴
    
    img.save('game_assets/runner_fail.png')

def create_tree():
    img = Image.new('RGBA', (150, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    trunk_color = (139, 69, 19)  # 树干颜色
    leaf_colors = [(34, 139, 34), (0, 100, 0), (0, 128, 0)]  # 不同深浅的绿色
    
    # 绘制树干
    draw.rectangle([(65, 80), (85, 130)], fill=trunk_color)
    
    # 绘制树叶（多层）
    for i, color in enumerate(leaf_colors):
        size = 60 - i * 10
        x = 75 - size//2
        y = 80 - i * 20
        draw.ellipse([(x, y), (x+size, y+size)], fill=color)
    
    img.save('game_assets/tree.png')

def create_cloud():
    img = Image.new('RGBA', (150, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    cloud_color = (255, 255, 255, 220)  # 半透明的白色
    
    # 绘制多个重叠的圆形来创建云朵
    draw.ellipse([(20, 30), (80, 70)], fill=cloud_color)
    draw.ellipse([(40, 20), (90, 60)], fill=cloud_color)
    draw.ellipse([(60, 30), (120, 70)], fill=cloud_color)
    draw.ellipse([(30, 40), (100, 80)], fill=cloud_color)
    
    img.save('game_assets/cloud.png')

def main():
    create_directory()
    create_runner_idle()
    create_runner_run1()
    create_runner_run2()
    create_runner_success()
    create_runner_fail()
    create_tree()
    create_cloud()
    print("所有游戏资源已创建完成！")

if __name__ == "__main__":
    main() 