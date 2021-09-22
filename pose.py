
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

from . import face_pose

class Pose():
    def __init__(self, model_path, use_cpu):
        self.transformations = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
        self.model = face_pose.pose()
        self.device = torch.device('cpu' if use_cpu else 'cuda:{}'.format(0))
        saved_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(saved_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)
        self.idx_tensor_yaw = [idx for idx in range(120)]
        self.idx_tensor_yaw = torch.FloatTensor(self.idx_tensor_yaw).to(self.device)

    def get_face_image(self, frame, face_rect):
        x_min, y_min, x_max, y_max = face_rect

        x_min -= 50
        x_max += 50
        y_min -= 50
        y_max += 30
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        # Crop face loosely
        img = frame[y_min:y_max, x_min:x_max]
        return Image.fromarray(img)

    def preprocess(self, image):
        img = image
        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).to(self.device)
        return img

    def __call__(self, image, face_rect):
        """
        get face pose from image
        :param image:       org image
        :param face_rect:   face rectangle
        :return:            yaw,pitch,roll
        """
        frame = image

        # convert color
        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get face image
        face_image = self.get_face_image(cv2_frame, face_rect)

        # preprocess
        input_image = self.preprocess(face_image)

        with torch.no_grad():
            yaw, pitch, roll = self.model(input_image)

            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)

            yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor_yaw) * 3 - 180
            pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

        return yaw_predicted,pitch_predicted,roll_predicted

