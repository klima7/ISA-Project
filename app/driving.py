from meta import Path, Crossing
from routing import Position, RouteFinder
from planning import RoutePlanner
from visual import TrafficLightColor, recognize_light_color, find_signs
from car import Car


class Task:

    def __init__(self, target, offset=1.0, backward=False):
        self.type = type
        self.target = target
        self.offset = offset
        self.backward = backward


class Driver:
    # Speed in simulation units per second
    NORMAL_SPEED = 1
    BACKWARD_SPEED = 1
    WALKWAY_SPEED = 0.4
    LIMITED_SPEED = 0.5

    def __init__(self, car, mm):
        self.car = car
        self.mm = mm
        self.planner = RoutePlanner()

        self.tasks = []
        self.cur_task = None

        self.route = None
        self.position = None

        self.orientations = []
        self.found_orientations = 0

        self.cur_path = Path.get_path_closest_to_point(self.mm.paths, self.car.gps)

    def drive_to_structure(self, structure_name, backward=False):
        structure = self.mm.get_structure_by_name(structure_name)
        task = Task(structure, 0.5, backward)
        self.tasks.append(task)

    def drive_to_path(self, path_name, offset, backward=False):
        path = self.mm.get_path_by_name(path_name)
        task = Task(path, offset, backward)
        self.tasks.append(task)

    def update(self):
        self.update_task()
        self.follow_route()
        self.update_speed()
        self.update_traffic_lights()
        self.update_car_lights()

        view = find_signs(self.car.view)
        if view is not None:
            self.car.set_view_visualization(view)

    def update_task(self):
        if self.cur_task is None:
            if self.tasks:
                self.cur_task = self.tasks.pop(0)
                self.position = Position(reversed=self.cur_task.backward)
                self.route = RouteFinder.find_route(self.cur_path, self.cur_task.target,
                                                    backward=self.cur_task.backward)

    def update_speed(self):
        if self.cur_task is None:
            self.car.velocity = 0
        elif self.cur_task.backward:
            self.car.velocity = -self.BACKWARD_SPEED
        else:
            self.car.velocity = self.NORMAL_SPEED

    def follow_route(self):
        if self.cur_task is None:
            return

        leading_point = self.car.front_point if not self.cur_task.backward else self.car.back_point
        self.position.offset = self.cur_path.get_closest_offset(leading_point)


        if len(self.orientations) < 15:
            self.orientations.append(self.car.orient)
        else:
            self.orientations[self.found_orientations] = self.car.orient
            self.found_orientations += 1
            if self.found_orientations == 15:
                self.found_orientations = 0

        orient = sum(self.orientations) / len(self.orientations)

        path, radius = self.planner.plan_route(self.route, self.position, leading_point, self.car.orient,
                                               backward=self.cur_task.backward)

        print("orient: ", orient, " radius: ", radius)
        if self.car.orient > orient and abs(self.car.orient - orient) > 0.0001:
            radius += 0.15
        elif self.car.orient < orient and abs(self.car.orient - orient) > 0.0001:
            radius -= 0.15

        self.car.set_wheels_by_radius(radius if not self.cur_task.backward else -radius)

        self.car.set_planned_path_visualization(path)

        if self.cur_path == self.route[
            len(self.route) - 1] and self.position.get_offset_from_start() >= self.cur_task.offset:  # TODO
            self.cur_task = None

        elif self.position.offset == self.position.get_end_offset():
            self.position.ordinal += 1
            self.position.offset = self.position.get_start_offset()
            self.cur_path = self.route[self.position]

    def update_traffic_lights(self):
        if self.cur_task is None or self.cur_task.backward:
            return

        color = recognize_light_color(self.car.view)
        if color == TrafficLightColor.RED or color == TrafficLightColor.YELLOW:
            self.car.velocity = 0

    def update_car_lights(self):
        self.car.running_lights = True
        self.car.stop_lights = self.car.velocity == 0

        if self.route is None:
            self.car.indicators_lights = Car.INDICATORS_DISABLED
            return

        next_crossing_pos = self.route.get_next_position(self.position, lambda path: isinstance(path.structure,
                                                                                                Crossing) or path.is_roundabout_exit())
        next_crossing_distance = self.route.get_distance_between(self.position, next_crossing_pos)

        prev_crossing_pos = self.route.get_prev_position(self.position, lambda path: isinstance(path.structure,
                                                                                                Crossing) or path.is_roundabout_exit())
        prev_crossing_distance = self.route.get_distance_between(prev_crossing_pos, self.position)

        if next_crossing_pos is not None and next_crossing_distance < 3.5:
            if 3.4 < next_crossing_distance < 3.5:
                self.angles = []
                self.found_angles = 0

            angle = self.route.get_angle(next_crossing_pos)
            if angle > 0.2:
                self.car.indicators_lights = Car.INDICATORS_LEFT
            elif angle < -0.2:
                self.car.indicators_lights = Car.INDICATORS_RIGHT

            self.car.velocity = 0.3
        if prev_crossing_pos is not None and 3 > prev_crossing_distance > 2:
            self.car.indicators_lights = Car.INDICATORS_DISABLED
            if 3 > prev_crossing_distance > 2.9:
                self.angles = []
                self.found_angles = 0
