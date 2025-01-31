import utils.pytorch as torch_util
import utils.video_proc as video_util

bbox_pth = "models/bbox_cage.pth"
model, transform, device = torch_util.initialize_model(bbox_pth)

while True:
    folder_path, file_list = video_util.get_directory()
    if folder_path is not None:
        break

desktop_folders = video_util.process_videos_to_frames(file_list)
desktop_folders = video_util.bbox_folders(model, transform, device, desktop_folders)

video_util.vid_processor(desktop_folders)
    