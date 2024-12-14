"""
12/10 白天 log: 
- 增加了金币的检测, 金币的值分为10, 50, 100三种, 分类对应2, 3, 4; 
- 优化僵尸攻击逻辑: 放弃通过分类来区分攻击速度，统一攻击三次
- 增加老板键，防止退不出程序
- 优化信息打印逻辑
计划更新内容：
- 更换库为一个能够后台点击的库
- 添加简单的图形化界面

12/14 晚上 log:
- 优化部分逻辑，添加更多注释
"""

from ultralytics import YOLO
import pyautogui as auto
import pygetwindow
import numpy as np
import time
import os
from PIL import ImageGrab
import cv2
import sys
from utils import bossKeyboard

ROOTPATH = os.path.dirname(os.path.abspath(__file__)) # main.py所在目录
MODEL = os.path.join(ROOTPATH, "models","1zombie_yolo11s.pt") # 模型路径
WINDOWS_TITLE = "植物大战僵尸中文版" # 窗口标题
ZOMBIE_SIZE = (0.06, 0.1)  # 僵尸的尺寸，宽度和高度占屏幕的比例
MONEY = [10, 50, 100]  # 金币的值
PROGRAM_RUNNING_FLAG = bossKeyboard.bossKeyboard(["q"])  # 全局变量，控制主程序的状态
IS_DRAW = True # 是否绘画矩形框和展示图像


def dropFakeZombie(_x, _y, _wx, _wy) -> bool:
    """判断是否是识别错误的僵尸"""
    return _x < _wx * ZOMBIE_SIZE[0] or _y < _wy * ZOMBIE_SIZE[1]


def clickIt(locations, _window, img, clickTimes=1, clickNum=2):
    """点击指定位置"""
    for i in range(len(locations) if len(locations) < clickNum else clickNum):
        x, y, w, h = locations[i]
        for i in range(clickTimes):
            auto.click(x + _window.left, y + _window.top)
        if IS_DRAW:
            # 中心坐标转换为左上角坐标
            x, y, w, h = x - w // 2, y - h // 2, w, h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if IS_DRAW:
        return img
    return


def main():
    model = YOLO(MODEL, task="detect", verbose=False)
    classes = model.names
    try:
        window = pygetwindow.getWindowsWithTitle(WINDOWS_TITLE)[0]
    except IndexError:
        print("未找到窗口，请打开植物大战僵尸游戏")
        return

    # hwnd = win32gui.FindWindow(None, WINDOWS_TITLE)
    # if not hwnd:
    #     print("未找到窗口，请打开植物大战僵尸游戏")
    #     return

    print(
        f"Root path: {ROOTPATH}, Model path: {MODEL}, Classes: {classes}, Windows title: {WINDOWS_TITLE}"
    )
    coinCounter = 0
    diamandCounter = 0
    goldCounter = 0
    silverCounter = 0

    print("主程序正在运行，按下 'q' 键退出...")
    while PROGRAM_RUNNING_FLAG.program_running:
        bg = time.time()
        if window:
            x, y, w, h = window.left, window.top, window.width, window.height
            shot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            shot = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
            wx, wy = shot.shape[1], shot.shape[0]
            # print(f"wx: {wx}, wy: {wy}")

            # 遮挡不需要的区域，在指定区域绘画矩形
            shot = cv2.rectangle(
                shot, (0, 0), (int(wx * 0.7), int(wy * 0.17)), (0, 0, 0), -1
            )
            shot = cv2.rectangle(shot, (0, int(wy * 0.95)), (wx, wy), (0, 0, 0), -1)
            shot = cv2.rectangle(
                shot, (0, int(wy * 0.85)), (int(wx * 0.1), wy), (0, 0, 0), -1
            )

            shotDet = cv2.resize(shot, (640, 640))

            results = model(shotDet, task="detect", conf=0.8, verbose=False)[0]

            # resultsXY格式，二维数组：[[x,y,w,h],[x,y,w,h],...]
            # resultsCLS格式，一维数组：[0,1,0,1,...]
            # 每个标签的坐标和种类索引一一对应
            resultsXYandCLS = results.boxes.xywh.cpu().numpy()  # 所有标签的坐标
            resultsCLS = results.boxes.cls.cpu().numpy()  # 所有标签的种类索引
            # 更新数组
            resultsXYandCLS = [
                [*[int(j) for j in resultsXYandCLS[i]], int(resultsCLS[i])]
                for i in range(len(resultsXYandCLS))
            ]
            # print(resultsXYandCLS)
            zombies = []
            coins = []
            suns = []

            for i in range(len(resultsXYandCLS)):
                x, y, w, h = resultsXYandCLS[i][:4]

                # 分辨率转换
                x, y, w, h = x * wx // 640, y * wy // 640, w * wx // 640, h * wy // 640
                if resultsXYandCLS[i][4] == 0:
                    # print(f"Zombie: {w/wx} and {h/wy}")
                    if not dropFakeZombie(w, h, wx, wy):
                        zombies.append([x, y, w, h])

                if resultsXYandCLS[i][4] == 1:
                    suns.append([x, y, w, h])

                if resultsXYandCLS[i][4] in [2, 3, 4]:
                    coins.append([x, y, w, h])
                    coinCounter += MONEY[resultsXYandCLS[i][4] - 2]
                    if resultsXYandCLS[i][4] == 2:
                        diamandCounter += 1
                    if resultsXYandCLS[i][4] == 3:
                        goldCounter += 1
                    if resultsXYandCLS[i][4] == 4:
                        silverCounter += 1

            # 从左到右逻辑排序
            zombies.sort(key=lambda x: x[0] - x[2] // 2)

            if IS_DRAW:
                shot = clickIt(suns, window, shot, clickTimes=2)
                shot = clickIt(coins, window, shot, clickTimes=2)
                shot = clickIt(zombies, window, shot, clickTimes=3)
                auto.click(window.left + 200, window.top + 200)
                cv2.imshow("Screen", shot)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    pass
            else:
                clickIt(suns, window, shot, clickTimes=2)
                clickIt(coins, window, shot, clickTimes=2)
                clickIt(zombies, window, shot, clickTimes=3)
                auto.click(window.left + 200, window.top + 200)
                
            ed = time.time()
            sys.stdout.write(
                f"\rdetect speed: {ed-bg} ms, moneyFound: {coinCounter}, diamand_count: {diamandCounter}, gold_count: {goldCounter}, silver_count: {silverCounter}"
                + " " * 10
            )
            sys.stdout.flush()

            pass

    cv2.destroyAllWindows()


# 启动键盘监听器线程
PROGRAM_RUNNING_FLAG.startListen()

main()
