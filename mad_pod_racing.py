import random
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
import uuid
import pygame
import math

from gymnasium import spaces
"""
Case sigmoid action space 9 discrete :
    - -1 for nothing
    - 0 rad for angle 0 or 360° or 2PI for angle 360
    - 45° or PI/4 for angle 45
    - 90° or PI/2 for angle 90
    - 135° or 3PI/4 for angle 135
    - 180° or PI for angle 180
    - 225° or 5PI/4 for angle 225
    - 270° or 3PI/2 for angle 270
    - 315° or 7PI/3 for angle 315

Case Relu :
    - 1 Box between 0 and 2PI


observation space : 
   - position (x,y)
   - next checkpoint position (x,y)
   - next checkpoint distance (one value d)
   - next checkpoint angle (one value in rad)
   - speed (vector x,y)

rewards:
    - -1 per turn without checkpoint
    - +20 when end
    - +5 when checkpoint

end :
    - lap over : i.e all checkpoint validated in right order 3 times
    - 100 round without reaching a checkpoint
"""
ENV_WIDTH = 16000
ENV_HEIGHT = 9000
MAX_SPEED = 15000
TIME_OUT = 100
CP_REWARD = 5
END_REWARD = 20
TRAVEL_REWARD = -0.01
OUT_SCREEN_REWARD = 0  # -100


class MapPodRacing(gym.Env):

    def __init__(self):
        self.cp_queue = None
        self.timeout = None
        self.my_pod = None
        self.trajectory_reward = None
        self.map = None
        self.seed = None
        self.cp_done = 0
        self.action_space = gym.spaces.Discrete(12)
        '''self.angle_map = np.array([
            0,  # 0°
            np.pi / 4,  # 45°
            np.pi / 2,  # 90°
            3 * np.pi / 4,  # 135°
            np.pi,  # 180°
            5 * np.pi / 4,  # 225°
            3 * np.pi / 2,  # 270°
            7 * np.pi / 4  # 315°
        ])'''

        self.angle_map = np.array([
            0,  # 0°
            np.pi / 6,  # 30°
            np.pi / 3,  # 60°
            np.pi / 2,  # 90°
            2 * np.pi / 3,  # 120°
            5 * np.pi / 6,  # 150°
            np.pi,  # 180°
            7 * np.pi / 6,  # 210°
            4 * np.pi / 3,  # 240°
            3 * np.pi / 2,  # 270°
            5 * np.pi / 3,  # 300°
            11 * np.pi / 6  # 330°
        ])

        """self.action_space = spaces.Box(
            low=0.0,
            high=2 * np.pi,
            shape=(1,),
            dtype=np.float32
        )"""

        # Observation space (Box of 8 values)
        '''low = np.array([
            -2000, -2000,  # position x, y
            0, 0,  # checkpoint x, y
            0.0,  # distance (>= 0)
            -np.pi * 2,  # angle (in radians)
            -MAX_SPEED, -MAX_SPEED  # speed x, y
        ], dtype=np.float32)'''

        low = np.array([
           0,  # angle (in radians)
        ], dtype=np.float32)


        '''high = np.array([
            ENV_WIDTH + 2000, ENV_HEIGHT + 2000,  # position x, y
            ENV_WIDTH, ENV_HEIGHT,  # checkpoint x, y
            ENV_WIDTH * 2,  # distance
            np.pi * 2,  # angle
            MAX_SPEED, MAX_SPEED  # speed x, y
        ], dtype=np.float32)'''

        high = np.array([
            np.pi * 2,  # angle
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # render
        self.image_ratio = 25
        self.image_width = ENV_WIDTH / self.image_ratio
        self.image_heigh = ENV_HEIGHT / self.image_ratio

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.seed = uuid.uuid4().int & ((1 << 64) - 1)
        random.seed(self.seed)
        self.map = Map(self.seed)
        self.trajectory_reward = 0
        # Player information
        self.my_pod = random.choice(self.map.pods)
        self.timeout = TIME_OUT
        self.cp_done = 0
        self.cp_queue = deque(maxlen=18)
        for _ in range(3):
            self.cp_queue.extend(self.map.check_points)
        last_cp = self.cp_queue.popleft()
        self.cp_queue.append(last_cp)
        observation = self.get_obs()
        return observation, {'cp': self.cp_queue}

    def get_obs(self):
        cp_x, cp_y = 0, 0
        if len(self.cp_queue) > 0:
            cp_x, cp_y = self.cp_queue[0]
        angle = from_vector(self.my_pod.position, Vector(cp_x, cp_y)).angle()
        return np.array([to_positive_radians(angle)], dtype=np.float32)

    def step(self, action):
        # apply action on my pod
        angle = self.angle_map[action]
        self.my_pod.update_acceleration_from_angle(angle, 100)
        self.my_pod.apply_force(self.my_pod.acceleration)
        self.my_pod.step()
        self.my_pod.apply_friction()
        self.my_pod.end_round()
        terminated = False
        truncated = False
        reward = 0

        if point_to_segment_distance(self.cp_queue[0][0], self.cp_queue[0][1],
                                     self.my_pod.last_position.x, self.my_pod.last_position.y,
                                     self.my_pod.position.x, self.my_pod.position.y) <= CHECKPOINT_RADIUS:
            self.cp_queue.popleft()
            self.timeout = TIME_OUT
            if len(self.cp_queue) == 0:
                reward = END_REWARD
                terminated = True
            else:
                reward = CP_REWARD
                self.cp_done += 1
        else:

            if not self.my_pod.is_moving_forward(self.cp_queue[0][0],self.cp_queue[0][1]):
                reward += -1
            self.timeout -= 1
            if self.timeout <= 0:
                reward += -1
                terminated = True
        self.trajectory_reward += reward
        obs = self.get_obs()
        info = {"cp_completion": 1 - (len(self.cp_queue) / (len(self.map.check_points) * 3))}
        return obs, reward, terminated, truncated, info

    def render(self):
        canvas = pygame.Surface((self.image_width, self.image_heigh))
        canvas.fill((255, 255, 255))
        pygame.font.init()
        red = (255, 0, 0)
        past_red = (255, 100, 0)
        blue = (0, 0, 255)
        dark_grey = (64, 64, 64)
        font = pygame.font.Font(None, 36)
        text = font.render(str(self.trajectory_reward), True, blue)
        canvas.blit(text, (10, 10))

        cp_order = 0
        for checkpoint in self.map.check_points:
            pygame.draw.circle(canvas, dark_grey, np.array(checkpoint) / self.image_ratio,
                               CHECKPOINT_RADIUS / self.image_ratio)
            cp_text = font.render(str(cp_order), True, blue)
            canvas.blit(cp_text, (checkpoint[0] / self.image_ratio, checkpoint[1] / self.image_ratio))
            cp_order += 1
        pygame.draw.circle(canvas, past_red, np.array(self.my_pod.last_position.get_tuple()) / self.image_ratio,
                           POD_RADIUS / self.image_ratio)
        pygame.draw.circle(canvas, red, np.array(self.my_pod.position.get_tuple()) / self.image_ratio,
                           POD_RADIUS / self.image_ratio)
        pygame.draw.line(canvas, blue, np.array(self.my_pod.last_position.get_tuple()) / self.image_ratio,
                         np.array(self.my_pod.position.get_tuple()) / self.image_ratio, 1)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    """def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()"""


def supervised_action_choose(observation):
    target_angle = observation[1]  # angle to cp (in radians)
    angles = np.array([
        0,  # 0°
        np.pi / 6,  # 30°
        np.pi / 3,  # 60°
        np.pi / 2,  # 90°
        2 * np.pi / 3,  # 120°
        5 * np.pi / 6,  # 150°
        np.pi,  # 180°
        7 * np.pi / 6,  # 210°
        4 * np.pi / 3,  # 240°
        3 * np.pi / 2,  # 270°
        5 * np.pi / 3,  # 300°
        11 * np.pi / 6  # 330°
    ])
    # Normalize angular differences to [-π, π]
    angle_diffs = np.abs((angles - target_angle + np.pi) % (2 * np.pi) - np.pi)

    action = np.argmin(angle_diffs)
    return action

def supervised_action_choose_from_angle(target_angle):
    angles = np.array([
        0,  # 0°
        np.pi / 6,  # 30°
        np.pi / 3,  # 60°
        np.pi / 2,  # 90°
        2 * np.pi / 3,  # 120°
        5 * np.pi / 6,  # 150°
        np.pi,  # 180°
        7 * np.pi / 6,  # 210°
        4 * np.pi / 3,  # 240°
        3 * np.pi / 2,  # 270°
        5 * np.pi / 3,  # 300°
        11 * np.pi / 6  # 330°
    ])
    # Normalize angular differences to [-π, π]
    angle_diffs = np.abs((angles - target_angle + np.pi) % (2 * np.pi) - np.pi)

    action = np.argmin(angle_diffs)
    return action

import math
import random


_MAX_ROTATION_PER_TURN = math.pi / 10
_FRICTION = 0.15
CHECKPOINT_GENERATION_MAX_GAP = 100
POD_RADIUS = 400
SPACE_BETWEEN_POD = 100
CHECKPOINT_RADIUS = 600

def from_vector(a,b):
    return Vector(b.x - a.x,b.y - a.y)

def from_tuple(a, b):
    return Vector(b[0] - a[0], b[1] - a[1])

def short_angle_dist(a0, a1):
    max_angle = math.pi * 2
    da = (a1 - a0) % max_angle
    return (2 * da) % max_angle - da

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Returns the distance from point (px, py) to the segment from (x1, y1) to (x2, y2)."""
    # Handle degenerate segment case
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return math.hypot(px - x1, py - y1)

    # Project point onto the segment, computing parameterized t
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp t to [0, 1]
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)

def to_positive_radians(theta):
    return theta % (2 * math.pi)

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def angle(self):
        return math.atan2(self.y, self.x)

    def length(self):
        return math.hypot(self.x, self.y)

    def normalize(self):
        length = self.length()
        if length == 0:
            return Vector(0, 0)
        return Vector(self.x / length, self.y / length)

    def mult(self, factor):
        return Vector(self.x * factor, self.y * factor)

    def add(self, v):
        return Vector(self.x + v.x, self.y + v.y)

    def round(self):
        return Vector(round(self.x), round(self.y))

    def truncate(self):
        return Vector(math.trunc(self.x), math.trunc(self.y))

    def cross(self, s):
        return Vector(-s * self.y, s * self.x)

    def translate(self, p):
        return Vector(self.x + p[0], self.y + p[1])

    def get_tuple(self):
        return (self.x, self.y)

class Pod:
    def __init__(self, x, y):
        self.position = Vector(x, y)
        self.last_position = Vector(x, y)
        self.speed = Vector(0, 0)
        self.last_angle = None
        self.acceleration = Vector(0, 0)
        self.speed = Vector(0, 0)

    def turn_update(self, x, y):
        self.last_position.x = self.position.x
        self.last_position.y = self.position.y
        self.position.x = x
        self.position.y = y

    def apply_force(self, force):
        self.speed = self.speed.add(force).mult(1)

    def update_acceleration_from_angle(self,angle,power):
        if self.last_angle is not None:
            relative_angle = short_angle_dist(self.last_angle, angle)
            if abs(relative_angle) >= _MAX_ROTATION_PER_TURN:
                angle = self.last_angle + _MAX_ROTATION_PER_TURN * math.copysign(1, relative_angle)
        self.last_angle = angle

        direction = Vector(math.cos(angle), math.sin(angle))
        self.acceleration = direction.normalize().mult(power)


    def update_acceleration(self, x, y, power):
        if self.position.x != x or self.position.y != y:
            angle = from_vector(self.position, Vector(x, y)).angle()
            self.update_acceleration_from_angle(angle, power)
        else:
            self.acceleration = Vector(0, 0)

    def step(self):
        self.last_position.x = self.position.x
        self.last_position.y = self.position.y
        self.position = self.position.add(self.speed.mult(1))

    def apply_friction(self):
        self.speed = self.speed.mult(1 - _FRICTION)

    def end_round(self):
        self.position = self.position.round()
        self.speed = self.speed.truncate()

    def is_moving_forward(self, cp_x,cp_y):
        return math.hypot(self.last_position.x - cp_x, self.last_position.y - cp_y) > math.hypot(self.position.x - cp_x, self.position.y - cp_y)


maps = [
    [(12460, 1350), (10540, 5980), (3580, 5180), (13580, 7600)],
    [(3600, 5280), (13840, 5080), (10680, 2280), (8700, 7460), (7200, 2160)],
    [(4560, 2180), (7350, 4940), (3320, 7230), (14580, 7700), (10560, 5060), (13100, 2320)],
    [(5010, 5260), (11480, 6080), (9100, 1840)],
    [(14660, 1410), (3450, 7220), (9420, 7240), (5970, 4240)],
    [(3640, 4420), (8000, 7900), (13300, 5540), (9560, 1400)],
    [(4100, 7420), (13500, 2340), (12940, 7220), (5640, 2580)],
    [(14520, 7780), (6320, 4290), (7800, 860), (7660, 5970), (3140, 7540), (9520, 4380)],
    [(10040, 5970), (13920, 1940), (8020, 3260), (2670, 7020)],
    [(7500, 6940), (6000, 5360), (11300, 2820)],
    [(4060, 4660), (13040, 1900), (6560, 7840), (7480, 1360), (12700, 7100)],
    [(3020, 5190), (6280, 7760), (14100, 7760), (13880, 1220), (10240, 4920), (6100, 2200)],
    [(10323, 3366), (11203, 5425), (7259, 6656), (5425, 2838)],
]

class Map:
    def __init__(self, seed):
        random.seed(seed)
        points = random.choice(maps)
        # Rotate list by a random amount
        rotation = random.randint(0, len(points) - 1)
        points = points[rotation:] + points[:rotation]

        # Generate checkpoint list with random offset
        self.check_points = []
        for x, y in points:
            offset_x = random.randint(-CHECKPOINT_GENERATION_MAX_GAP, CHECKPOINT_GENERATION_MAX_GAP)
            offset_y = random.randint(-CHECKPOINT_GENERATION_MAX_GAP, CHECKPOINT_GENERATION_MAX_GAP)
            self.check_points.append((x + offset_x, y + offset_y))

        start_point = self.check_points[0]
        direction = from_tuple(start_point,self.check_points[1]).normalize().cross(1)
        podCount = 2

        pods = []
        for i in range(podCount):
            offset = ((-1 if i % 2 == 0 else 1) * (i // 2 * 2 + 1) + podCount % 2)
            position = direction.mult(offset * (POD_RADIUS + SPACE_BETWEEN_POD)).translate(start_point).round()
            pod = Pod(position.x,position.y)
            pods.append(pod)

        self.pods = pods