import utils.pytorch as torch_util
import utils.video_proc as video_util

while True:
    folder_path = video_util.get_directory()
    if folder_path is not None:
        break

bbox_pth = "models/bbox_cage.pth"
model, transform, device = torch_util.initialize_model(bbox_pth)