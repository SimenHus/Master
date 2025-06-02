
from src import System
from src.structs import Camera, ImageData
from src.util import DataLoader

class CameraExtrinsicCalibration:

    def __init__(self) -> None:
        camera, Twc = DataLoader.load_stx_camera()

        self.SLAM = System()
        self.SLAM.tracker.set_camera(camera)

        self.images: list[ImageData] = DataLoader.load_stx_images()


    def start(self) -> None:
        
        for image in self.images:
            self.SLAM.track_monocular(image.image, image.timestep, mask=image.mask)

    def save_keyframes(self, filename: 'str') -> None:
        self.SLAM.save_keyframes(filename)

    def save_map_points(self, filename: 'str') -> None:
        self.SLAM.save_map_points(filename)

if __name__ == '__main__':
    app = CameraExtrinsicCalibration()
    app.start()

    OUTPUT_FOLDER = './output'
    app.save_keyframes(f'{OUTPUT_FOLDER}/KeyFrames')
    app.save_map_points(f'{OUTPUT_FOLDER}/MapPoints')
    
    exit()