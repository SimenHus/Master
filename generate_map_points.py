
from src import System
from src.structs import Camera, ImageData
from src.util import DataLoader

class CameraExtrinsicCalibration:

    def __init__(self) -> None:
        # base_folder = '/mnt/c/Users/shustad/Desktop/Skole/Prosjektoppgave/data/osl'
        # data_folder = base_folder + '/2024-02-08/2024-02-08-14/Cam1/Lens0'
        # dataloader_file = base_folder + '/dataloader.json'

        # data_folder = '/mnt/c/Users/simen/Desktop/Prog/Dataset/mav0/cam0/data'
        
        # with open(dataloader_file, 'r') as f: camera_dict = json.load(f)['Cam1']['Lens0']


        base_folder = './datasets/osl'
        data_folder = base_folder + '/data'
        camera, Twc = DataLoader.load_stx_camera(base_folder)

        self.SLAM = System()
        self.SLAM.tracker.set_camera(camera)



        n = 50
        # self.images: list[list[cv2.Mat, int]] = data_load_home(data_folder, n)
        self.images: list[ImageData] = DataLoader.load_stx_images(data_folder, n)


    def start(self) -> None:
        
        for image in self.images:
            self.SLAM.track_monocular(image.image, image.timestep)

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