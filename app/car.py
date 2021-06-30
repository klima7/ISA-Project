import math

import numpy as np

import util
from api.b0RemoteApi import RemoteApiClient
from util import Point


class Car:

    LENGTH = 0.316
    WIDTH = 0.213
    TURNING_ANGLE = math.pi / 3

    INDICATORS_DISABLED = 0
    INDICATORS_LEFT = 1
    INDICATORS_RIGHT = 2
    INDICATORS_HAZARD = 3

    BACK_POINT_DISTANCE = 0.25
    FRONT_POINT_DISTANCE = 0.25

    def __init__(self, client: RemoteApiClient):
        self._client = client
        _, self.camera_handle = self._client.simxGetObjectHandle('ViewCamera', self._client.simxServiceCall())
        _, self.view_visualization_handle = self._client.simxGetObjectHandle('ViewVisualization', self._client.simxServiceCall())
        self._client.simxGetVisionSensorImage(self.camera_handle, False, self._client.simxDefaultSubscriber(self._view_received_handler, 1))

        self.velocity = 0
        self.left_angle = 0
        self.right_angle = 0

        self.indicators_lights = Car.INDICATORS_DISABLED
        self.stop_lights = False
        self.running_lights = False
        self.reverse_lights = False

        self.gps = None
        self.orient = None
        self.view = None

        self.refresh()

    def _view_received_handler(self, data):
        _, size, view = data
        # self.set_view_visualization(view)
        view = [b for b in view]
        view = np.array(view, dtype=np.uint8)
        view.resize((size[1], size[0], 3))
        view = np.flipud(view)
        self.view = view

    def refresh(self):
        _, *data = self._client.simxCallScriptFunction("get_state@Car", "sim.scripttype_childscript", [], self._client.simxServiceCall())

        self.gps = Point(*data[0])
        self.orient = data[1]

        steering_data = data[2]
        self.velocity = steering_data[0]
        self.left_angle, self.right_angle = steering_data[1]

        lights_data = data[3]
        self.indicators_lights = lights_data[0]
        self.stop_lights = bool(lights_data[1])
        self.running_lights = bool(lights_data[2])
        self.reverse_lights = bool(lights_data[3])

        _, self.lidar_main_data = self._client.simxUnpackFloats(data[4], self._client.simxServiceCall())

    def set_planned_path_visualization(self, path):
        data = [[s.x, s.y] for s in path.samples]
        self._client.simxCallScriptFunction("set_path@PlannedRouteVisualization", "sim.scripttype_childscript", data, self._client.simxDefaultPublisher())

    def set_view_visualization(self, view_rgb):
        view = np.flipud(view_rgb)
        view = view.flatten(order='C')
        view = view.astype(np.uint8)
        image_bytes = view.tobytes()
        self._client.simxSetVisionSensorImage(self.view_visualization_handle, False, image_bytes, self._client.simxDefaultPublisher())

    def apply(self):
        steering_data = [self.velocity, self.left_angle, self.right_angle]
        lights_data = [self.indicators_lights, int(self.stop_lights), int(self.running_lights), int(self.reverse_lights)]

        data = [steering_data, lights_data]
        self._client.simxCallScriptFunction("set_state@Car", "sim.scripttype_childscript", data, self._client.simxServiceCall())

        self._client.simxSpinOnce()

    def navigate(self, target: Point):
        diff = [target.x-self.gps.x, target.y-self.gps.y]
        target_angle = util.get_vector_angle(diff)
        car_angle = self.orient
        diff_angle = car_angle - target_angle

        if diff_angle > math.pi:
            diff_angle = diff_angle - 2*math.pi
        if diff_angle < -math.pi:
            diff_angle = diff_angle + 2*math.pi

        diff_angle = min(self.TURNING_ANGLE, diff_angle)
        diff_angle = max(-self.TURNING_ANGLE, diff_angle)

        diff_angle = util.rad2deg(diff_angle)
        self.set_wheels_by_angle(diff_angle)

    def set_wheels_by_radius(self, radius):
        close_angle = math.atan(self.LENGTH / (abs(radius) - self.WIDTH/2))
        far_angle = math.atan(self.LENGTH / (abs(radius) + self.WIDTH / 2))

        if radius < 0:
            left_angle, right_angle = far_angle, close_angle
        else:
            left_angle, right_angle = -close_angle, -far_angle

        self.right_angle = left_angle
        self.left_angle = right_angle

    def set_wheels_by_angle(self, angle):
        radius = self.LENGTH / math.tan(util.deg2rad(angle)) if angle != 0 else math.inf
        self.set_wheels_by_radius(radius)

    @property
    def front_point(self):
        return util.move_forward(self.gps, self.orient, 0)

    @property
    def back_point(self):
        return util.move_forward(self.gps, self.orient, 0)

    def get_lidar_data(self):
        return self.lidar_main_data

