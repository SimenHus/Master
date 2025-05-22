
from src.util import DataLoader, Time

base_folder = './datasets/osl'
data_folder = base_folder + '/data'
camera, Twc = DataLoader.load_stx_camera(base_folder)


images = DataLoader.load_stx_images(data_folder, 5)

for image in images:
    timestep = image.timestep
    calced_time = Time.TimeConversion.POSIX_to_STX(timestep)
    print(timestep, calced_time, image.filename)
