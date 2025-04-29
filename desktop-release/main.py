import curses
import utils.pytorch as torch_util
import utils.video_proc as video_util
import utils.terminal as terminal_util

def main(stdscr):
    terminal_util.opening_page(stdscr)

    bbox_pth = "models/bbox_cage.pth"
    model, transform, device = torch_util.initialize_model(bbox_pth)

    terminal_util.step1_func(stdscr)

    desktop_folders, file_list = video_util.get_directory()
    while True:
        if not file_list:
            terminal_util.step1_func_err1(stdscr)
            desktop_folders, file_list = video_util.get_directory()
        elif not all(f.lower().endswith('.mp4') for f in file_list):
            terminal_util.step1_func_err2(stdscr)
            desktop_folders, file_list = video_util.get_directory()
        else:
            break
    
    desktop_folders = terminal_util.run_with_ascii_loading(
        stdscr,
        terminal_util.frame_split,  # you can make another art like ascii_processing1 if you want
        video_util.process_videos_to_frames,
        file_list
    )

    desktop_folders = terminal_util.run_with_ascii_loading(
        stdscr,
        terminal_util.cropping_cage,
        video_util.bbox_folders,
        model, transform, device, desktop_folders
    )

    terminal_util.run_with_ascii_loading(
        stdscr,
        terminal_util.image_stitch,
        video_util.vid_processor,
        desktop_folders
    )

    import utils.deeplabcut as dlc_util

    # You can even do a closing curses screen if you want
    stdscr.clear()
    stdscr.addstr(0, 0, "Processing complete. Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()

curses.wrapper(main)