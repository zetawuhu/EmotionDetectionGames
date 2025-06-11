import pygame
import sys
import math
import random
from emotion_ninja import EmotionNinjaGame
from emotion_flappy import EmotionFlappyGame

class ParticleEffect:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(2, 4)
        self.color = random.choice([
            (255, 223, 186),  # 温暖的橙色
            (255, 218, 185),  # 桃色
            (255, 228, 196),  # 米色
        ])
        self.speed = random.uniform(1, 3)
        self.angle = random.uniform(0, 2 * math.pi)
        self.life = 255

    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.life -= 3
        return self.life > 0

class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.is_hovered = False
        self.hover_progress = 0
        self.particles = []
        
    def draw(self, screen):
        # 绘制按钮阴影
        shadow_rect = self.rect.copy()
        shadow_rect.x += 4
        shadow_rect.y += 4
        pygame.draw.rect(screen, (30, 30, 30, 100), shadow_rect, border_radius=15)
        
        # 计算按钮颜色
        base_color = (240, 240, 240)
        hover_color = (255, 223, 186)
        current_color = [
            int(base_color[i] + (hover_color[i] - base_color[i]) * self.hover_progress)
            for i in range(3)
        ]
        
        # 绘制按钮主体
        pygame.draw.rect(screen, current_color, self.rect, border_radius=15)
        pygame.draw.rect(screen, (70, 70, 70), self.rect, 2, border_radius=15)
        
        # 绘制文本
        text_surf = self.font.render(self.text, True, (50, 50, 50))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
        # 更新并绘制粒子效果
        if self.is_hovered:
            if random.random() < 0.3:
                self.particles.append(
                    ParticleEffect(
                        random.randint(self.rect.left, self.rect.right),
                        random.randint(self.rect.top, self.rect.bottom)
                    )
                )
        
        self.particles = [p for p in self.particles if p.update()]
        for particle in self.particles:
            pygame.draw.circle(
                screen,
                (*particle.color, particle.life),
                (int(particle.x), int(particle.y)),
                particle.size
            )
    
    def update(self):
        target = 1.0 if self.is_hovered else 0.0
        self.hover_progress += (target - self.hover_progress) * 0.1

class GameLauncher:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("表情游戏合集")
        
        # 加载字体
        try:
            self.title_font = pygame.font.Font("fonts/SimHei.ttf", 64)
            self.font = pygame.font.Font("fonts/SimHei.ttf", 32)
        except:
            self.title_font = pygame.font.Font(None, 64)
            self.font = pygame.font.Font(None, 32)
        
        # 创建按钮
        self.buttons = [
            Button(200, 250, 400, 80, "逃跑的忍者", self.font),
            Button(200, 350, 400, 80, "表情像素鸟", self.font),
            Button(200, 450, 400, 80, "退出游戏", self.font)
        ]
        
        # 背景动画
        self.bg_particles = []
        self.time = 0
        
    def create_background_particle(self):
        x = random.randint(0, self.screen_width)
        y = random.randint(0, self.screen_height)
        return ParticleEffect(x, y)
        
    def draw_background(self):
        # 绘制渐变背景
        for y in range(self.screen_height):
            progress = y / self.screen_height
            color = [
                int(240 - progress * 40),  # R
                int(240 - progress * 40),  # G
                int(255 - progress * 40)   # B
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
            
        # 更新并绘制背景粒子
        if random.random() < 0.1:
            self.bg_particles.append(self.create_background_particle())
            
        self.bg_particles = [p for p in self.bg_particles if p.update()]
        for particle in self.bg_particles:
            pygame.draw.circle(
                self.screen,
                (*particle.color, particle.life),
                (int(particle.x), int(particle.y)),
                particle.size
            )
            
    def draw_title(self):
        self.time += 0.05
        offset = math.sin(self.time) * 5
        
        title = self.title_font.render("表情游戏合集", True, (50, 50, 50))
        title_rect = title.get_rect(center=(self.screen_width // 2, 100 + offset))
        
        # 绘制标题阴影
        shadow = self.title_font.render("表情游戏合集", True, (30, 30, 30))
        shadow_rect = shadow.get_rect(center=(self.screen_width // 2 + 4, 100 + offset + 4))
        self.screen.blit(shadow, shadow_rect)
        
        # 绘制主标题
        self.screen.blit(title, title_rect)

    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, button in enumerate(self.buttons):
                        if button.rect.collidepoint(event.pos):
                            if i == 0:  # 忍者
                                pygame.quit()
                                ninja_game = EmotionNinjaGame()
                                ninja_game.run()
                                pygame.init()
                                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                            elif i == 1:  # 像素鸟
                                pygame.quit()
                                flappy_game = EmotionFlappyGame()
                                flappy_game.run()
                                pygame.init()
                                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                            elif i == 2:  # 退出游戏
                                pygame.quit()
                                sys.exit()
            
            # 更新按钮状态
            for button in self.buttons:
                button.is_hovered = button.rect.collidepoint(mouse_pos)
                button.update()
            
            # 绘制界面
            self.draw_background()
            self.draw_title()
            
            for button in self.buttons:
                button.draw(self.screen)
            
            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    launcher = GameLauncher()
    launcher.run() 