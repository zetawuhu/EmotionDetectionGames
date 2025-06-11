# buaa 大三《人机交互》课程大作业——面部表情识别游戏
## 简介
本项目基于开源项目 https://github.com/otaha178/Emotion-recognition 进行开发，包括“逃跑的忍者”和“表情像素鸟”两个游戏。

其中“逃跑的忍者”为一款表情匹配游戏，做出指定的表情并维持一段时间即可通关；

“表情像素鸟”是基于Flappy Brid开发的躲避游戏，当玩家做出“Neutral(中立)”表情时会向下飞；作出“Happy(开心)”表情时会向上飞；其余表情维持高度不变。

## 安装
本项目需要 python 版本为 3.6，同时以 tensorflow 为基础，运行
```
pip install -r requirement.txt
```
以安装需要的依赖。
## 运行
环境配置完毕后，运行 `python real_time_video.py` 可以启动正常的表情检测

运行 `python game_launcher.py` 可以启动游戏，点击可以选择想玩的游戏。
