import sys
if sys.platform == 'win32':
    print('=>>>>load data from window platform')
    yolov5_src = r"D:\workspace\pro\source\yolov5"
    sys.path.append(yolov5_src)
import os, argparse

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# sys.path.append('/jiangtao2/code/deep-head-pose-master/')
print(sys.path)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import random

if sys.platform == "win32":
    sys.path.append("C:/code/yolov5")
else:
    sys.path.append("/mnt/ssd_disk_1/huhong/lzy/yolov5")

# from FaceDetection import *

from FaceDetection import FaceDetect

import face_pose, utils

def getlist(dir,extension,Random=True):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if extension == ext:
                list.append(os.path.join(root,name))
                list[-1]  = list[-1].replace('\\','/')
    if Random:
        random.shuffle(list)
    return list

def parse_args():

    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)

    parser.add_argument('--snapshot', dest='snapshot', help='Path of model.',
                        default='model/aver_error_2.2484_epoch_53_multi.pkl', type=str)

    parser.add_argument('--video', dest='video_path', help='Path of video',
                        default=r'E:\workspace\pro\facialExpression\data\kehu',type=str)

    parser.add_argument('--bboxes', dest='bboxes', help='Bounding box annotations of frames',
                        default='',type=str)

    parser.add_argument('--output_string', dest='output_string', help='String appended to output file',
                        default='_result',type=str)

    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames',
                        default=20,type=int)

    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', 
                        default=8.,type=float)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
 
    snapshot_path = args.snapshot
    out_dir = 'output'

    videoList = getlist(args.video_path,'.avi')
    videoList += getlist(args.video_path,'.mp4')


    class Obj(object):
        def __init__(self):
            super().__init__()


    yoloargs = Obj()
    yoloargs.cpu = False
    yoloargs.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
    yoloargs.imgsz = 640
    Detector = FaceDetect(args=yoloargs)
    # Detector = FaceDetect(device='cuda:{}'.format(gpu))
    # Detector.set_device('cuda:1')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    # model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    #model = hopenet.WHENet()
    # model = hopenet.Pose_lzy_new()
    model = face_pose.pose()
    # model = hopenet.HopeNet_JT()
    # model = hopenet.Hopenet_OPTION(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cuda:{}'.format(gpu)))
    model.load_state_dict(saved_state_dict,strict=True)

    print('Loading data.')

    # transformations = transforms.Compose([transforms.Resize(224),
    # transforms.CenterCrop(224), transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transformations = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    # transformations = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    idx_tensor_yaw = [idx for idx in range(120)]
    idx_tensor_yaw = torch.FloatTensor(idx_tensor_yaw).cuda(gpu)

    for video_path in tqdm(videoList):

        #video = cv2.VideoCapture(video_path)
        video = cv2.VideoCapture(0)

        # New cv2
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output/big_angles_multi/{}.avi'.format(os.path.splitext(os.path.basename(video_path))[0]), fourcc, args.fps, (width, height))


        # txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

        while True:

            ret,frame = video.read()
            if ret == False:
                break

            cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            rects = Detector.detect(frame)

            x_min, y_min, x_max, y_max = int(float(rects[0])), int(float(rects[1])), int(float(rects[2])), int(float(rects[3]))

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
                                                                
            # x_min -= 3 * bbox_width / 4
            # x_max += 3 * bbox_width / 4
            # y_min -= 3 * bbox_height / 4
            # y_max += bbox_height / 4
            x_min -= 50
            x_max += 50
            y_min -= 50
            y_max += 30
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            # Crop face loosely
            img = cv2_frame[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)
            with torch.no_grad():
                yaw, pitch, roll = model(img)

            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)
            # Get continuous predictions in degrees.
            # yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor_yaw) * 3 - 180
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            font = cv2.FONT_HERSHEY_COMPLEX
            # Print new frame with cube and axis
            cv2.putText(frame,'yaw:{:.2f} pitch:{:.2f} roll:{:.2f} \n'.format(yaw_predicted, pitch_predicted, roll_predicted),
                        (50,100), font, 1, (0, 0, 255), 1)
            # cv2.putText(frame,'yaw:{:.2f} \n'.format(yaw_predicted),
            #             (200,200), font, 2, (255, 255, 255), 1)
            # txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
            # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
            # util.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
            # Plot expanded bounding box
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)\

            cv2.namedWindow('video', 0)
            cv2.imshow('video', frame)
            key = cv2.waitKey(100)
            if key == ord("q"):
                break
            elif key == 27:
                sys.exit()

            # out.write(frame)

        # out.release()
        video.release()

