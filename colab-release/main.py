import utils.pytorch as torch_util
import utils.video_proc as video_util
import utils.terminal as terminal_util
import utils.deeplabcut as dlc_util

bbox_pth = "models/bbox_cage.pth"
model, transform, device = torch_util.initialize_model(bbox_pth)

desktop_folders = video_util.process_videos_to_frames(file_list)
desktop_folders = video_util.bbox_folders(model, transform, device, desktop_folders)

video_util.vid_processor(desktop_folders)