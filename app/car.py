from api.b0RemoteApi import RemoteApiClient
import numpy as np
import app.util as util
import math


class Car:

    def __init__(self, client: RemoteApiClient):
        self.client = client

        _, self.compass_reference = self.client.simxGetObjectHandle('CompassReference', client.simxServiceCall())
        _, self.compass = self.client.simxGetObjectHandle('Compass', client.simxServiceCall())

        _, self.fl_wheel = self.client.simxGetObjectHandle('FLSteerJoint', client.simxServiceCall())
        _, self.fr_wheel = self.client.simxGetObjectHandle('FRSteerJoint', client.simxServiceCall())
        _, self.bl_wheel = self.client.simxGetObjectHandle('BLWheelJoint', client.simxServiceCall())
        _, self.br_wheel = self.client.simxGetObjectHandle('BRWheelJoint', client.simxServiceCall())

        _, self.camera = self.client.simxGetObjectHandle('CarVision', client.simxServiceCall())

        self.width = util.calc_distance(self.client, self.fl_wheel, self.fr_wheel)
        self.length = util.calc_distance(self.client, self.fl_wheel, self.bl_wheel)

    def set_force(self, force):
        self.client.simxSetJointMaxForce(self.bl_wheel, force, self.client.simxDefaultPublisher())
        self.client.simxSetJointMaxForce(self.br_wheel, force, self.client.simxDefaultPublisher())

    def set_velocity(self, velocity):
        self.client.simxSetJointTargetVelocity(self.bl_wheel, velocity, self.client.simxDefaultPublisher())
        self.client.simxSetJointTargetVelocity(self.br_wheel, velocity, self.client.simxDefaultPublisher())

    # Ustawia koła zgodnie z podanym promieniem skrętu. Kąt dodatni w prawo, ujemny w lewo
    def set_wheels_by_radius(self, radius):

        close_angle = math.atan(self.length / (abs(radius) - self.width/2))
        far_angle = math.atan(self.length / (abs(radius) + self.width / 2))

        if radius < 0:
            left_angle, right_angle = far_angle, close_angle
        else:
            left_angle, right_angle = -close_angle, -far_angle

        self.client.simxSetJointTargetPosition(self.fr_wheel, left_angle, self.client.simxDefaultPublisher())
        self.client.simxSetJointTargetPosition(self.fl_wheel, right_angle, self.client.simxDefaultPublisher())

    # Ustawia kół pod podanym kątem. Kąt dodatni w prawo, ujemny w lewo
    def set_wheels_by_angle(self, angle):
        radius = self.length / math.tan(util.deg2rad(angle)) if angle != 0 else math.inf
        self.set_wheels_by_radius(radius)

    def get_camera_image(self):
        _, size, data = self.client.simxGetVisionSensorImage(self.camera, False, self.client.simxServiceCall())
        data = [x for x in data]
        image = np.array(data, dtype=np.uint8)
        image.resize([size[1], size[0], 3])
        return image

    def set_running_lights(self, enabled):
        sig_value = 1 if enabled else 0
        self.client.simxSetIntSignal("RunningLights", sig_value, self.client.simxServiceCall())

    def set_stop_lights(self, enabled):
        sig_value = 1 if enabled else 0
        self.client.simxSetIntSignal("StopLights", sig_value, self.client.simxServiceCall())

    def set_reversing_lights(self, enabled):
        sig_value = 1 if enabled else 0
        self.client.simxSetIntSignal("ReversingLights", sig_value, self.client.simxServiceCall())

    def set_hazard_lights(self, enabled):
        sig_value = 3 if enabled else 0
        self.client.simxSetIntSignal("IndicatorsLights", sig_value, self.client.simxServiceCall())

    def enable_left_indicators(self):
        self.client.simxSetIntSignal("IndicatorsLights", 1, self.client.simxServiceCall())

    def enable_right_indicators(self):
        self.client.simxSetIntSignal("IndicatorsLights", 2, self.client.simxServiceCall())

    def disable_indicators(self):
        self.client.simxSetIntSignal("IndicatorsLights", 0, self.client.simxServiceCall())

    # Zwraca orientację samochodu w stopniach (-180, 180)
    def get_orientation(self):
        _, orient = self.client.simxGetObjectOrientation(self.compass, self.compass_reference, self.client.simxServiceCall())
        gamma = util.rad2deg(orient[2])
        return gamma

