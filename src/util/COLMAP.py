
from src.util import Time, Geometry
from src.util import DataLoader
import numpy as np
from pycolmap import Reconstruction
from collections import defaultdict
from src.util.ProgressBar import ProgressBar


class COLMAP:
    MAX_REPROJECTION_ERROR = 1.0 # [pixels]
    MIN_OBSERVATIONS = 3
    MIN_TRIANGULATION_ANGLE = 10.0 # [deg]
    MIN_DIST_FROM_CAMERA = 1 # [m]
    MAX_DIST_FROM_CAMERA = 80 # [m]
    POINTS_PER_BIN = 3
    GRID_SIZE = (12, 10)

    def __init__(self) -> None:
        COLMAP_project_path = DataLoader.reconstrucion_folder()
        sparse_model_path = f'{COLMAP_project_path}/sparse/'
        scene = Reconstruction(sparse_model_path)

        self.images = scene.images
        self.points3D = scene.points3D
        example_img = next(iter(self.images.values()))
        img_w, img_h = example_img.camera.width, example_img.camera.height
        self.image_size = [img_w, img_h]

        self.poses = {}
        for img in self.images.values():
            timestep = Time.TimeConversion.generic_to_POSIX(img.name.strip('.jpg'))

            rot = Geometry.SO3(img.cam_from_world.rotation.matrix())
            trans = img.cam_from_world.translation
            pose = Geometry.SE3(rot, trans).inverse()

            self.poses[timestep] = pose


        self.landmarks, self.landmark_scores = self.determine_scores()

        self.update_observations()

    def determine_scores(self) -> tuple[dict, dict]:
        landmark_scores = {}
        landmarks = {}
        points3D = self.points3D
        print(f'Scoring {len(points3D)} landmarks')
        pb = ProgressBar(len(points3D))
        before = len(points3D)
        for i, (point3D_id, pt) in enumerate(points3D.items()):
            pb.print(i)
            if pt.error > self.MAX_REPROJECTION_ERROR: continue
            if pt.track.length() < self.MIN_OBSERVATIONS: continue

            cam_centers = []
            for elem in pt.track.elements:
                img = self.images[elem.image_id]
                timestep = Time.TimeConversion.generic_to_POSIX(img.name.strip('.jpg'))
                cam_centers.append(self.poses[timestep].translation())

            tri_angle = self.triangulation_angle(pt.xyz, cam_centers)
            if tri_angle < self.MIN_TRIANGULATION_ANGLE: continue

            score = (1 / (pt.error + 1e-3)) + pt.track.length() + tri_angle
            landmark_scores[point3D_id] = score
            landmarks[point3D_id] = pt.xyz

        pb.print(len(points3D))
        print(f'Removed {before - len(landmarks)} landmarks based bad reprojection, track or triangulation angle')
        return landmarks, landmark_scores
    
    @classmethod
    def triangulation_angle(clc, pt3D, views) -> float:
        vecs = [pt3D - c for c in views]
        # Pick the most distant two camera centers
        dists = [np.linalg.norm(v) for v in vecs]
        i, j = np.argmax(dists), np.argmin(dists)
        v1, v2 = vecs[i], vecs[j]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle
    
    @classmethod
    def voxel_index(clc, xyz, voxel_size=1.0):
        return tuple((xyz / voxel_size).astype(int))
    
    def get_keypoint_position(self, pt3D, only_first=True):
        total = np.zeros(2)
        count = 0
        for element in pt3D.track.elements:
            point2D_id = element.point2D_idx
            img = self.images[element.image_id]
            kp = img.points2D[point2D_id]
            total += kp.xy
            count += 1
            if only_first: break
        return total / count if count > 0 else None


    def filter_spatial(self) -> None:
        print('Performing spatial filtering of landmarks')
        pb = ProgressBar(len(self.landmarks))
        before = len(self.landmarks)
        voxels = defaultdict(list)
        pb.print(0)
        for i, (pt_id, pos3D) in enumerate(self.landmarks.items()):
            v_idx = self.voxel_index(pos3D, voxel_size=3.0)
            score = self.landmark_scores[pt_id]
            voxels[v_idx].append((score, pt_id, pos3D))
            pb.print(i+1)

        selected = {}
        for j, (v, pts) in enumerate(voxels.items()):
            pts.sort(reverse=True)  # highest score first
            selected_pts = pts[:self.POINTS_PER_BIN]  # keep top x per voxel
            for _, pt_id, pos3D in selected_pts:
                selected[pt_id] = pos3D

        self.landmarks = selected
        self.update_observations()
        print(f'Removed {before - len(self.landmarks)} landmarks based on spatial proximity')
    
    def filter_binned(self) -> None:
        print('Performing binning of landmarks')
        pb = ProgressBar(len(self.landmarks))
        before = len(self.landmarks)
        bins = defaultdict(list)
        pb.print(0)
        for i, (pt_id, pos3D) in enumerate(self.landmarks.items()):
            # Use first visible image to bin it
            score = self.landmark_scores[pt_id]
            pt3D = self.points3D[pt_id]
            x, y = self.get_keypoint_position(pt3D, only_first=True)
            col = int((x / self.image_size[0]) * self.GRID_SIZE[0])
            row = int((y / self.image_size[1]) * self.GRID_SIZE[1])
            bins[(col, row)].append((score, pt_id, pos3D))
            pb.print(i+1)

        selected = {}
        for bin_key, pts in bins.items():
            pts.sort(reverse=True)  # sort by score
            for i in range(min(self.POINTS_PER_BIN, len(pts))):  # top 3 per bin
                _, pt_id, pos3D = pts[i]
                selected[pt_id] = pos3D

        self.landmarks = selected
        self.update_observations()
        print(f'Removed {before - len(self.landmarks)} landmarks based on binning')

    def filter_distance(self, poses_NED: dict[int: Geometry.SE3]) -> None:
        print('Performing filtering of landmarks based on distance')
        pb = ProgressBar(len(self.landmarks))
        before = len(self.landmarks)
        filtered = {}
        pb.print(0)
        for i, (pt_id, pos3D) in enumerate(self.landmarks.items()):
            pb.print(i+1)
            for pose in poses_NED.values():
                if np.linalg.norm(pos3D - pose.translation()) < self.MIN_DIST_FROM_CAMERA: continue
                if np.linalg.norm(pos3D - pose.translation()) > self.MAX_DIST_FROM_CAMERA: continue
                filtered[pt_id] = pos3D

        self.landmarks = filtered
        self.update_observations()
        print(f'Removed {before - len(self.landmarks)} landmarks based on distance')


    def align(self, poses_NED: dict[int: Geometry.SE3], Trc = None) -> None:
        print('Aligning frames')
        NED_poses = []
        COLMAP_poses = []
        for timestep, colmap_pose in self.poses.items():

            ned_pose = poses_NED[timestep]
            if Trc is not None: ned_pose = ned_pose.compose(Trc)
            NED_poses.append(ned_pose)
            COLMAP_poses.append(colmap_pose)

        c, T_align = Geometry.SE3.align_frames(COLMAP_poses, NED_poses)
        for mp_id, point3d in self.landmarks.items():
            self.landmarks[mp_id] = c * T_align.rotation().matrix() @ point3d + T_align.translation()
        
        self.update_observations()
        print('Frames aligned successfully')



    def update_observations(self) -> None:
        obs_dict = {} # Timestep: list[(pixels, mp_id)]
        for img in self.images.values():
            timestep = Time.TimeConversion.generic_to_POSIX(img.name.strip('.jpg'))

            observations = img.get_observation_points2D()
            obs_list = []
            for obs in observations:
                mp_id = obs.point3D_id
                if mp_id not in self.landmarks: continue

                obs_list.append((obs.xy, mp_id))

            obs_dict[timestep] = obs_list

        self.observations = obs_dict