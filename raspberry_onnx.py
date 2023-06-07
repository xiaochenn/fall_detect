import os
import cv2
import numpy as np
import onnxruntime
import time
import threading
import PCF8591 as ADC
import RPi.GPIO as GPIO
import math
import argparse
import imutils
from collections import deque

CLASSES = ['people', 'fall']
cap = cv2.VideoCapture(0)
is_running = True
onnx_path = '/home//pi//PycharmProjects//pc8591_main//best.onnx'
makerobo_R = 11  # 声控灯红输出管脚
makerobo_G = 12  # 声控灯绿输出管脚
makerobo_B = 13  # 声控灯蓝输出管脚
makerobo_sound_pin = 15  # 声音传感器输入管脚
makerobo_fire_pin = 16  # 火灾灯输出管脚
makerobo_Buzzer_pin = 18  # 火灾报警器输出管脚
time_counter = 5  # 时间计数器
fps_counter = 0  # 帧率计数器
start_time = time.time()  # 开始时间
buzzer_flag = False
GPIO.setmode(GPIO.BOARD)  # 采用实际的物理管脚给GPIO口


class YOLOV5():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    # -------------------------------------------------------
    #   获取输入输出的名字
    # -------------------------------------------------------
    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    # -------------------------------------------------------
    #   输入图像
    # -------------------------------------------------------
    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    # -------------------------------------------------------
    #   1.cv2读取图像并resize
    #	2.图像转BGR2RGB和HWC2CHW
    #	3.图像归一化
    #	4.图像增加维度
    #	5.onnx_session 推理
    # -------------------------------------------------------
    def inference(self, img):
        or_img = cv2.resize(img, (640, 640))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img


model = YOLOV5(onnx_path)


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #   计算框的面积
    #	置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    # -------------------------------------------------------
    #   删除为1的维度
    #	删除置信度小于conf_thres的BOX
    # -------------------------------------------------------
    org_box = np.squeeze(org_box)
    print(max(org_box[..., 4]))
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]
    # -------------------------------------------------------
    #	通过argmax获取置信度最大的类别
    # -------------------------------------------------------
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))
    # -------------------------------------------------------
    #   分别对每个类别进行过滤
    #	1.将第6列元素替换为类别下标
    #	2.xywh2xyxy 坐标转换
    #	3.经过非极大抑制后输出的BOX下标
    #	4.利用下标取出非极大抑制后的BOX
    # -------------------------------------------------------

    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        # curr_cls_box_old = np.copy(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box, iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output


def draw(image, box_data,fps):
    # -------------------------------------------------------
    #	取整，方便画框
    # -------------------------------------------------------
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)
    cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def cap_check():
    global cap
    global model
    global buzzer_flag
    global is_running
    global fps_counter
    while is_running:
        fps_counter += 1
        if fps_counter >= 15:
            fps = fps_counter / (time.time() - start_time)
            start_time = time.time()
            fps_counter = 0
        ret, frame = cap.read()
        if ret:
            output, or_img = model.inference(frame)
            outbox = filter_box(output, 0.5, 0.5)
            if (len(outbox) != 0):
                draw(or_img, outbox,fps)
                buzzer_flag = True
            else:
                buzzer_flag = False
            cv2.imshow('res.jpg', or_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            print('cap error')
            break


def makerobo_setup():  # 初始化函数
    ADC.setup(0x48)  # 设置8591地址
    global pins
    global p_R, p_G, p_B
    pins = {'pin_R': makerobo_R, 'pin_G': makerobo_G, 'pin_B': makerobo_B}
    # 设置GPIO口输入输出模式
    for i in pins:
        GPIO.setup(pins[i], GPIO.OUT)
        GPIO.output(pins[i], GPIO.LOW)
    GPIO.setwarnings(False)
    GPIO.setup(makerobo_sound_pin, GPIO.IN)
    GPIO.setup(makerobo_fire_pin, GPIO.OUT)
    GPIO.setup(makerobo_Buzzer_pin, GPIO.OUT)
    GPIO.output(makerobo_fire_pin, GPIO.LOW)
    GPIO.output(makerobo_Buzzer_pin, GPIO.HIGH)
    p_R = GPIO.PWM(pins['pin_R'], 2000)
    p_G = GPIO.PWM(pins['pin_G'], 1999)
    p_B = GPIO.PWM(pins['pin_B'], 5000)
    p_R.start(0)
    p_G.start(0)
    p_B.start(0)


def makerobo_rgb_off():
    for i in pins:
        GPIO.output(pins[i], GPIO.LOW)


def makerobo_rgb_on():
    # for i in pins:
    #     GPIO.output(pins[i],GPIO.HIGH)
    p_R.ChangeDutyCycle(0)
    p_G.ChangeDutyCycle(0)
    p_B.ChangeDutyCycle(0)


def makerobo_loop():  # 循环函数
    light = ADC.read(0)  # 读入8591的光敏电阻的值
    tempareture = change(ADC.read(1))  # 读入模拟温度传感器的值并转化为摄氏度
    global buzzer_flag
    # 打印相关信息
    print('light = ', light)
    print('temperature = ', tempareture, 'C')
    print('sound = ', GPIO.input(makerobo_sound_pin))

    global time_counter  # time_counter是为了能够时刻进行火灾预警而设的全局变量（避免使用time.sleep）
    time_counter += 0.2

    if light > 100 and tempareture < 50 and time_counter >= 5:  # 当光线足够暗且并非火灾时，控制声控灯的开关，time_counter使声控灯打开至少5s再进行下一次判断
        if GPIO.input(makerobo_sound_pin) == GPIO.LOW:  # 判断有无声音
            makerobo_rgb_on()
            time_counter = 0  # 重置time_counter
        else:
            makerobo_rgb_off()
    elif light < 100 and tempareture < 50:
        makerobo_rgb_off()

    if tempareture > 50 and light > 150:  # 如果温度和光线（烟雾浓度）超过预警，警报灯亮红灯，声控灯常亮，蜂鸣器工作
        makerobo_rgb_on()
        GPIO.output(makerobo_fire_pin, GPIO.HIGH)
        makerobo_beep(0.5)
    else:
        GPIO.output(makerobo_fire_pin, GPIO.LOW)
        GPIO.output(makerobo_Buzzer_pin, GPIO.HIGH)
    if buzzer_flag:
        makerobo_beep(0.5)
    else:
        GPIO.output(makerobo_Buzzer_pin,GPIO.HIGH)

def change(makerobo_analogVal):  # 将温度传感器的值转化为摄氏度
    makerobo_Vr = 5 * float(makerobo_analogVal) / 255
    makerobo_Rt = 10000 * makerobo_Vr / (5 - makerobo_Vr)
    makerobo_temp = 1 / (((math.log(makerobo_Rt / 10000)) / 3950) + (1 / (273.15 + 25)))
    makerobo_temp = makerobo_temp - 273.15
    return makerobo_temp


def makerobo_buzzer_on():
    GPIO.output(makerobo_Buzzer_pin, GPIO.LOW)


def makerobo_buzzer_off():
    GPIO.output(makerobo_Buzzer_pin, GPIO.HIGH)


def makerobo_beep(x):  # 蜂鸣器工作函数
    makerobo_buzzer_on()
    time.sleep(x)
    makerobo_buzzer_off()
    time.sleep(x)


def destroy():
    p_R.stop()
    p_G.stop()
    p_B.stop()
    makerobo_rgb_off()
    GPIO.cleanup()
    ADC.write(0)


if __name__ == "__main__":
    try:
        check = threading.Thread(target=cap_check)
        check.start()
        makerobo_setup()
        while True:
            makerobo_loop()
            time.sleep(0.5)
    except KeyboardInterrupt:
        destroy()
        is_running = False
        check.join()
        cap.release()
        cv2.destroyAllWindows()




