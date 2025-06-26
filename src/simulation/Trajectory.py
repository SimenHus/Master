

from src.util import Geometry

import numpy as np


class SeaStates:
    def __init__(self):
        self.roll_amplitude = 0.1 # [rad]
        self.pitch_amplitude = 0.05 # [rad]

        self.sway_amplitude = 0.2 # [m]
        self.heave_amplitude = 0.3 # [m]

        self.bobbing_frequency = 0.5 # [Hz]
        self.steps = 100
        self.dt = 0.1

    def set_roll_amplitude(self, value, degrees=False) -> None:
        if degrees: value *= np.pi / 180
        self.roll_amplitude = value
    
    def set_pitch_amplitude(self, value, degrees=False) -> None:
        if degrees: value *= np.pi / 180
        self.pitch_amplitude = value

    def set_sway_amplitude(self, value) -> None:
        self.sway_amplitude = value

    def set_heave_amplitude(self, value) -> None:
        self.heave_amplitude = value

    def set_bobbing_frequency(self, value) -> None:
        self.bobbing_frequency = value

    def set_steps(self, value) -> None:
        self.steps = value

    def set_dt(self, value) -> None:
        self.dt = value

    @staticmethod
    def preset_calm() -> 'SeaStates':
        settings = SeaStates()
        settings.set_roll_amplitude(0)
        settings.set_pitch_amplitude(0)
        settings.set_sway_amplitude(0)
        settings.set_heave_amplitude(0)
        settings.set_bobbing_frequency(0)
        return settings

    @staticmethod
    def preset_sway() -> 'SeaStates':
        settings = SeaStates()
        settings.set_roll_amplitude(0)
        settings.set_pitch_amplitude(0)
        settings.set_heave_amplitude(0)
        return settings


class TrajectoryGenerator:


    @classmethod
    def bob(clc, settings: SeaStates, t) -> float:
        omega = 2 * np.pi * settings.bobbing_frequency * t
        roll = settings.pitch_amplitude * np.sin(omega)
        pitch = settings.pitch_amplitude * np.sin(omega + np.pi / 4)
        sway = settings.sway_amplitude * np.sin(omega + np.pi / 3)
        z = settings.heave_amplitude * np.sin(omega)
        return roll, pitch, sway, z

    @classmethod
    def semi_circular(clc, w=1.0, settings = SeaStates()) -> list[Geometry.SE3]:
        
        dt = settings.dt
        circ_factor = 1/2
        radius = 30.0
        circumference = 2 * np.pi * radius * circ_factor
        total_movement = circumference + radius

        step_length = total_movement / settings.steps
        entry_steps = int(radius/step_length)
        circular_steps = settings.steps - entry_steps
        delta_trans = [step_length, 0, 0]

        # ENTRY PART
        t = 0
        x0 = Geometry.SE3(Geometry.SO3(), [-radius, -radius, 0])
        trajectory = [x0]

        entry_odom = Geometry.SE3(Geometry.SO3(), delta_trans)
        yaw = 0
        last_base = x0
        for _ in range(entry_steps):
            t += dt
            next_base = last_base.compose(entry_odom)

            dx = np.cos(yaw)
            dy = np.sin(yaw)
            nx = -dy
            ny = dx

            roll, pitch, sway, z = clc.bob(settings, t)
            pos = next_base.translation()
            x = pos[0] + sway * nx
            y = pos[1] + sway * ny

            next_pose = Geometry.SE3.from_vector([roll, pitch, yaw, x, y, z])
            trajectory.append(next_pose)
            last_base = next_base
        
        # CIRCULAR PART

        total_yaw = 2 * np.pi * w * circ_factor
        delta_yaw = total_yaw / circular_steps

        delta_rot = Geometry.SO3.Yaw(delta_yaw)

        circle_odom = Geometry.SE3(delta_rot, delta_trans)

        for _ in range(circular_steps):
            t += dt
            next_base = last_base.compose(circle_odom)
            
            yaw += delta_yaw
            dx = np.cos(yaw)
            dy = np.sin(yaw)
            nx = -dy
            ny = dx

            roll, pitch, sway, z = clc.bob(settings, t)

            pos = next_base.translation()
            x = pos[0] + sway * nx
            y = pos[1] + sway * ny

            next_pose = Geometry.SE3.from_vector([roll, pitch, yaw, x, y, z])

            trajectory.append(next_pose)
            last_base = next_base

        return trajectory


def boat_straight_movement(steps=100) -> list[Geometry.SE3]:
    trajectory = []

    start_x = -30.0  # match circular motion start
    start_y = -30.0
    start_z = 0
    end_x = 30.0
    total_distance = end_x - start_x
    delta_x = total_distance / (steps - 1)  # to include both start and end points

    for i in range(steps):
        x = start_x + i * delta_x
        y = start_y
        z = start_z

        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        trajectory.append(Geometry.SE3.from_vector(np.array([roll, pitch, yaw, x, y, z]), radians=False))

    return trajectory


    # @classmethod
    # def circular(clc, w=1.0, steps=100, delta_t=0.1, settings = SeaStates()) -> list[Geometry.SE3]:
    #     """
    #     w: turn radius as percentage of full circle
    #     """
    #     trajectory = []

    #     radius = 30.0  # meters
    #     total_yaw = 2 * np.pi * w # [rad]
    #     delta_yaw = total_yaw / steps

    #     dx = 

    #     for i in range(steps):
    #         t = i * delta_t
    #         theta = np.pi / 4 + angular_speed * t

    #         # Base circular position
    #         base_x = radius * np.cos(theta)
    #         base_y = radius * np.sin(theta)

    #         # Tangent and normal
    #         dx = -np.sin(theta)
    #         dy = np.cos(theta)
    #         nx = -dy
    #         ny = dx

    #         # Add lateral sway in normal direction
    #         sway = sway_amp * np.sin(2 * np.pi * bobbing_freq * t + np.pi / 3)
    #         x = base_x + sway * nx
    #         y = base_y + sway * ny

    #         # Z bobbing due to waves
    #         z = z_amp * np.sin(2 * np.pi * bobbing_freq * t)

    #         # Oscillatory roll and pitch
    #         roll = roll_amp * np.sin(2 * np.pi * bobbing_freq * t)
    #         pitch = pitch_amp * np.sin(2 * np.pi * bobbing_freq * t + np.pi / 4)

    #         # Yaw: facing tangent to path
    #         yaw = theta + np.pi / 2

    #         trajectory.append(Geometry.SE3.from_vector(np.array([roll, pitch, yaw, x, y, z])))

    #     return trajectory