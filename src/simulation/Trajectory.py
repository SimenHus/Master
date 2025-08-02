

from src.util import Geometry

import numpy as np


class SeaStates:
    calm_att = 0.01 # [deg]
    calm_pos = 0.001 # [m]

    moderate_att = 3.
    moderate_pos = 1.

    rough_att = 15.
    rough_pos = 10.

    def __init__(self):
        self.roll_amplitude = 0.1 # [rad]
        self.pitch_amplitude = 0.05 # [rad]

        self.sway_amplitude = 0.2 # [m]
        self.heave_amplitude = 0.3 # [m]

        self.bobbing_frequency = 0.3 # [Hz]
        self.steps = 100
        self.dt = 0.1

        self.distance = 30

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

    def set_distance(self, value) -> None:
        self.distance = value

    @classmethod
    def preset_calm(clc) -> 'SeaStates':
        settings = SeaStates()
        settings.set_roll_amplitude(clc.calm_att, degrees=True)
        settings.set_pitch_amplitude(clc.calm_att, degrees=True)
        settings.set_sway_amplitude(clc.calm_pos)
        settings.set_heave_amplitude(clc.calm_pos)
        return settings

    @classmethod
    def preset_moderate(clc) -> 'SeaStates':
        settings = SeaStates()
        settings.set_roll_amplitude(clc.moderate_att, degrees=True)
        settings.set_pitch_amplitude(clc.moderate_att, degrees=True)
        settings.set_sway_amplitude(clc.moderate_pos)
        settings.set_heave_amplitude(clc.moderate_pos)
        return settings
    

    @classmethod
    def preset_rough(clc) -> 'SeaStates':
        settings = SeaStates()
        settings.set_roll_amplitude(clc.rough_att, degrees=True)
        settings.set_pitch_amplitude(clc.rough_att, degrees=True)
        settings.set_sway_amplitude(clc.rough_pos)
        settings.set_heave_amplitude(clc.rough_pos)
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
        radius = settings.distance
        circumference = 2 * np.pi * radius * circ_factor
        total_movement = circumference + radius

        step_length = total_movement / settings.steps
        entry_steps = int(radius/step_length)
        circular_steps = settings.steps - entry_steps
        delta_trans = [step_length, 0, 0]

        # ENTRY PART
        ROLL_180 = Geometry.SE3.from_vector([180, 0, 0, 0, 0, 0], radians=False)
        t = 0
        x0 = Geometry.SE3(Geometry.SO3(), [-radius, -radius, 0])
        trajectory = [x0.compose(ROLL_180)]

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

            next_pose = Geometry.SE3.from_vector([roll, pitch, yaw, x, y, z], radians=True)
            next_pose = next_pose.compose(ROLL_180)
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

            next_pose = Geometry.SE3.from_vector([roll, pitch, yaw, x, y, z], radians=True)
            next_pose = next_pose.compose(ROLL_180)

            trajectory.append(next_pose)
            last_base = next_base

        return trajectory