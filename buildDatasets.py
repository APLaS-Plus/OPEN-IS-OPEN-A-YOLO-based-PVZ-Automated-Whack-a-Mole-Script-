# 计时器
import time

bgt = time.time()

import os
import shutil
from PIL import Image, ImageDraw
import numpy as np


# 创建输出文件夹
ROOTPATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOTPATH, "datasets", "ZVP")
SOURCEIMG = os.path.join(ROOTPATH, "ZVPImgs")
CLASSES = ["nomal_zombie", "sun", "diamond", "coin_gold_dollar", "coin_silver_dollar"]
OVERLAPTHRESHOLD = 0.4
ISSHOW = False
NUM_OF_DATASET = 10
SEED = 42
np.random.seed(SEED)

if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
    print("Removed old datasets")
os.makedirs(OUTPUT)

sourceImgs = os.listdir(SOURCEIMG)

# 读取图片
zombies = [img for img in sourceImgs if "zombie" in img]
# print(zombies)
for i in range(len(zombies)):
    gif = Image.open(os.path.join(SOURCEIMG, zombies[i]))
    frames = []
    # 读取所有帧
    while True:
        try:
            gif.seek(gif.tell() + 1)
            frames.append(gif.copy())
        except EOFError:
            break

    zombies[i] = frames

backgrounds = [img for img in sourceImgs if "Background" in img]

gif = Image.open(os.path.join(SOURCEIMG, "Sun.gif"))
sun = []
# 读取所有帧
while True:
    try:
        gif.seek(gif.tell() + 1)
        sun.append(gif.copy())
    except EOFError:
        break

diamond = Image.open(os.path.join(SOURCEIMG, "Diamond.png"))
goldCoin = Image.open(os.path.join(SOURCEIMG, "coin_gold_dollar.png"))
silverCoin = Image.open(os.path.join(SOURCEIMG, "coin_silver_dollar.png"))


def detectOverlap(allBoxes, box):
    # 计算重叠比例是否超过一方的面积的OVERLAPTHRESHOLD
    # print(allBoxes)
    # print(box)
    boxArea = box[2] * box[3]
    for i in allBoxes:
        # print("i:", i)
        x1 = max(i[0], box[0])
        y1 = max(i[1], box[1])
        x2 = min(i[0] + i[2], box[0] + box[2])
        y2 = min(i[1] + i[3], box[1] + box[3])
        if x1 >= x2 or y1 >= y2:
            continue
        overlapArea = (x2 - x1) * (y2 - y1)
        if (
            overlapArea > boxArea * OVERLAPTHRESHOLD
            or overlapArea > i[2] * i[3] * OVERLAPTHRESHOLD
        ):
            return True
    return False


def drawRectangle(img, boxes: list):
    # 检查boxes是否越界
    if boxes[0] + boxes[2] > img.size[0]:
        boxes[2] = img.size[0] - boxes[0]
    if boxes[1] + boxes[3] > img.size[1]:
        boxes[3] = img.size[1] - boxes[1]
    # 更改boxes为左上角坐标和右下角坐标
    boxes = [boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]]
    # 画框
    draw = ImageDraw.Draw(img)
    draw.rectangle(boxes, outline="red")

    return img


def addThings(img, imgloop, thing, isScale=False, isScaleX=False):
    Bx, By = img.size # 背景的宽和高
    W, H = thing.size # 物体的宽和高

    # 正比例缩放
    if isScale:
        scale = np.random.uniform(0.5, 1.2)
        W, H = int(W * scale), int(H * scale)
        thing = thing.resize((W, H))

    # 只缩放宽度
    if isScaleX:
        scale = np.random.uniform(0.3, 1.0)
        W = int(W * scale)
        thing = thing.resize((W, H))

    # 检测是否重叠并且添加
    # print(f"Bx: {Bx}, By: {By}, W: {W}, H: {H}")
    while True:
        x, y = np.random.randint(Bx - W), np.random.randint(By - H)
        if not detectOverlap(imgloop, (x, y, W, H)):
            img.paste(thing, (x, y), thing)
            imgloop.append([x, y, W, H])
            break

    if ISSHOW:
        img = drawRectangle(img, [x, y, W, H])
        # 画出x,y位置
        draw = ImageDraw.Draw(img)
        draw.point((x + W // 2, y + H // 2), fill="red")

    return img, [(x + W // 2) / Bx, (y + H // 2) / By, W / Bx, H / By]


def buildDataset(idx):
    # 导入背景
    img = Image.open(os.path.join(SOURCEIMG, np.random.choice(backgrounds)))

    img = img.resize((1400, 640))

    # 将图片随机剪切成640*640
    tmp = np.random.randint(0, 760)
    img = img.crop((tmp, 0, tmp + 640, 640))
    # img.show()
    # input()

    imgLabel = ""

    boxloops = []

    # 添加普通僵尸，仿照yolo的数据集生成
    for i in range(np.random.randint(5, 10)):
        zombieAdd = zombies[np.random.randint(len(zombies))]
        zombieAdd = zombieAdd[np.random.randint(len(zombieAdd))]
        img, addLabel = addThings(img, boxloops, zombieAdd)
        imgLabel += "0 " + " ".join([str(i) for i in addLabel]) + "\n"

    # 添加阳光，并且适当缩放
    for i in range(np.random.randint(1, 5)):
        sunAdd = sun[np.random.randint(len(sun))]
        img, addLabel = addThings(img, boxloops, sunAdd, True)
        imgLabel += "1 " + " ".join([str(i) for i in addLabel]) + "\n"

    # 添加钻石
    for i in range(np.random.randint(1, 5)):
        img, addLabel = addThings(img, boxloops, diamond)
        imgLabel += "2 " + " ".join([str(i) for i in addLabel]) + "\n"

    # 添加金币
    for i in range(np.random.randint(1, 5)):
        img, addLabel = addThings(img, boxloops, goldCoin, isScaleX=True)
        imgLabel += "3 " + " ".join([str(i) for i in addLabel]) + "\n"

    # 添加银币
    for i in range(np.random.randint(1, 5)):
        img, addLabel = addThings(img, boxloops, silverCoin, isScaleX=True)
        imgLabel += "4 " + " ".join([str(i) for i in addLabel]) + "\n"

    ## 展示图片
    # img.show()
    # print(imgLabel)
    # input()

    # 保存图片和标签
    img.save(os.path.join(OUTPUT, f"{idx}.jpg"))
    with open(os.path.join(OUTPUT, f"{idx}.txt"), "w") as f:
        f.write(imgLabel)
    print(f"Saved {idx}/{NUM_OF_DATASET}")
    print(f"Zombies: {imgLabel.count('0')}, Suns: {imgLabel.count('1')}")
    print(imgLabel)


for i in range(1, NUM_OF_DATASET + 1):
    buildDataset(i)

# 划分训练集和验证集3：1
os.makedirs(os.path.join(OUTPUT, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "labels", "val"), exist_ok=True)

imgList = os.listdir(OUTPUT)
imgList = [img for img in imgList if img.endswith(".jpg")]
np.random.shuffle(imgList)
trainList = imgList[: int(len(imgList) * 0.75)]
valList = imgList[int(len(imgList) * 0.75) :]

for i in trainList:
    shutil.move(
        os.path.join(OUTPUT, f"{i}"), os.path.join(OUTPUT, "images", "train", f"{i}")
    )
    shutil.move(
        os.path.join(OUTPUT, f"{i[:-4]}.txt"),
        os.path.join(OUTPUT, "labels", "train", f"{i[:-4]}.txt")
    )
for i in valList:
    shutil.move(
        os.path.join(OUTPUT, f"{i}"), os.path.join(OUTPUT, "images", "val", f"{i}")
    )
    shutil.move(
        os.path.join(OUTPUT, f"{i[:-4]}.txt"),
        os.path.join(OUTPUT, "labels", "val", f"{i[:-4]}.txt")
    )

edt = time.time()
print("Time used:", edt - bgt)
