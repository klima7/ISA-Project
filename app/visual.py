import cv2
import numpy as np
from app.constants import SignType, TrafficLightColor
from keras.models import load_model
import util
import math


SIGN_HEAD2STICK_FACTOR = 0.57
PERSPECTIVE_ANGLE = 70
CAMERA_IMAGE_HEIGHT = 256
CAMERA_IMAGE_WIDTH = 512
SIGN_REAL_HEIGHT = 0.48978


print('Loading signs Neural Network')
nn_signs = load_model("../nn_signs.h5")
nn_signs.compile()
nn_signs.predict(np.zeros((1, 16, 16, 1), dtype=np.float))

print('Loading lights Neural Network')
nn_lights = load_model("../nn_lights.h5")
nn_lights.compile()
nn_lights.predict(np.zeros((1, 16, 16, 1), dtype=np.float))


class FoundSign:

    def __init__(self, stick, image):
        self.stick = stick
        self.head_pos = self._locate_sign_head()
        self.head_image = self._cut_head_image(image)
        self.type = SignType.UNKNOWN
        self.color = TrafficLightColor.NONE
        self.recognize_sign()

    def _locate_sign_head(self):
        x, y, h = self.stick
        head_size = int(SIGN_HEAD2STICK_FACTOR * h)
        return x - head_size // 2, y - head_size, head_size

    def _cut_head_image(self, image):
        x, y, a = self.head_pos
        if x < 0 or y < 0 or x + a >= CAMERA_IMAGE_WIDTH or y + a >= CAMERA_IMAGE_HEIGHT:
            return None
        return image[y:y + a, x:x + a]

    def recognize_sign(self):
        if self.head_image is None:
            return SignType.UNKNOWN

        gray = cv2.cvtColor(cv2.cvtColor(self.head_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (16, 16))
        scaled = np.reshape(scaled, (16, 16, 1))
        scaled = scaled.astype(np.float32) / 255

        res = nn_signs.predict(np.array([scaled]))
        res = np.argmax(res[0])
        self.type = SignType(res)

    def recognize_light(self):
        if self.head_image is None:
            return SignType.UNKNOWN

        gray = cv2.cvtColor(cv2.cvtColor(self.head_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (16, 16))
        scaled = np.reshape(scaled, (16, 16, 1))
        scaled = scaled.astype(np.float32) / 255

        res = nn_lights.predict(np.array([scaled]))
        res = np.argmax(res[0])
        self.color = TrafficLightColor(res)

    def _is_reversed(self):
        mask = cv2.inRange(self.head_image, np.array([20, 0, 190]), np.array([40, 50, 250]))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        full_height = self.head_pos[2]
        for contour in contours:
            _, _, _, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour)/h >= 5 and h >= full_height / 4 * 3:
                return True
        return False

    @property
    def distance(self):
        height = self.stick[2]
        numerator = CAMERA_IMAGE_HEIGHT * SIGN_REAL_HEIGHT
        denominator = 2 * height * math.tan(util.deg2rad(PERSPECTIVE_ANGLE/2))
        return numerator / denominator

    @property
    def angle(self):
        x = self.stick[0]
        diff = x - CAMERA_IMAGE_WIDTH / 2
        a_diff = abs(diff)
        a_angle = a_diff * util.deg2rad(PERSPECTIVE_ANGLE/2) / (CAMERA_IMAGE_WIDTH / 2)
        angle = a_angle if diff > 0 else -a_angle
        return angle

    def draw(self, hsv_image):
        head_start = (self.head_pos[0], self.head_pos[1])
        head_end = (self.head_pos[0] + self.head_pos[2], self.head_pos[1] + self.head_pos[2])
        cv2.rectangle(hsv_image, head_start, head_end, (0, 0, 0), 2)

        stick_start = (self.stick[0], self.stick[1])
        stick_end = (self.stick[0], self.stick[1]+self.stick[2])
        cv2.line(hsv_image, stick_start, stick_end, (0, 0, 0), 2)

        font = cv2.FONT_HERSHEY_PLAIN

        if self.type != SignType.TRAFFIC_LIGHTS:
            cv2.putText(hsv_image, self.type.name, (head_start[0], head_start[1]), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            self.recognize_light()
            cv2.putText(hsv_image, self.color.name, (head_start[0], head_start[1]), font, 1, (0, 0, 0), 1, cv2.LINE_AA)


def find_signs(image):
    if image is None:
        return None, None

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    sticks = _find_sticks(hsv_image)
    signs = [FoundSign(stick, hsv_image) for stick in sticks]

    for sign in signs:
        if sign.head_image is not None:
            sign.draw(hsv_image)

    return signs, cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def _find_sticks(hsv_image):
    sticks = []
    mask = cv2.inRange(hsv_image, np.array([20, 0, 105]), np.array([40, 10, 120]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        if h > 25:
            sticks.append((x+w//2, y, h))
    return sticks


def recognize_light_color(image):
    if image is None:
        return TrafficLightColor.NONE

    image = image[:, image.shape[1] // 2:]

    if _check_light_color(image, [0, 240, 150], [5, 255, 190]):
        return TrafficLightColor.RED

    if _check_light_color(image, [58, 240, 150], [62, 255, 190]):
        return TrafficLightColor.GREEN

    if _check_light_color(image, [28, 240, 150], [30, 255, 190]):
        return TrafficLightColor.YELLOW

    return TrafficLightColor.NONE


def _check_light_color(img, lower, upper):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image, np.array(lower), np.array(upper))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (abs(w - h) <= 30) and cv2.contourArea(contour) > 40:
            return True

    return False

