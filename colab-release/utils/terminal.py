import curses

def opening_page(stdscr):
    draw_centered_art(stdscr,ascii_open)

def step1_func(stdscr):
    draw_centered_art_enter(stdscr,step1)

def draw_centered_art(stdscr, ascii_art):
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    art_height = len(ascii_art)
    art_width = max(len(line) for line in ascii_art)
    start_y = (height - art_height) // 2
    start_x = (width - art_width) // 2

    stdscr.nodelay(True)

    for i, line in enumerate(ascii_art):
        try:
            stdscr.addstr(start_y + i, start_x, line)
            stdscr.refresh()
            if stdscr.getch() != -1:
                break
            curses.napms(50)
        except curses.error:
            pass

    curses.napms(1800)

def draw_centered_art_enter(stdscr, ascii_art):
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    art_height = len(ascii_art)
    art_width = max(len(line) for line in ascii_art)
    start_y = (height - art_height) // 2
    start_x = (width - art_width) // 2

    stdscr.nodelay(False)

    for i, line in enumerate(ascii_art):
        try:
            stdscr.addstr(start_y + i, start_x, line)
            stdscr.refresh()
            curses.napms(50)  # delay between lines
        except curses.error:
            pass

    while True:
        key = stdscr.getch()
        if key == ord('\n') or key == curses.KEY_ENTER:
            break


step1 = [
    "███████ ████████ ███████ ██████       ██    ",
    "██         ██    ██      ██   ██     ███ ██ ",
    "███████    ██    █████   ██████       ██    ",
    "     ██    ██    ██      ██           ██ ██ ",
    "███████    ██    ███████ ██           ██    ",
    "                                            ",
    "                                            ",
    "                                            ",
    "                                            ",
    "      Please select a folder with only      ",
    "        MP4s, anything else will be         ",
    "                rejected.                   ",
    "                                            ",
    "                                            ",
    "                                            ",
    "                                            ",
    "         [PRESS ENTER TO CONTINUE]          "
]

ascii_open = [
    "███████ ██████  ██████   █████   ██████  ██    ██ ███████ ",
    "██      ██   ██ ██   ██ ██   ██ ██       ██    ██ ██      ",
    "███████ ██████  ██████  ███████ ██   ███ ██    ██ █████   ",
    "     ██ ██      ██   ██ ██   ██ ██    ██ ██    ██ ██      ",
    "███████ ██      ██   ██ ██   ██  ██████   ██████  ███████ ",
    "                                                           ",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⠤⠾⠟⠓⠚⠓⠲⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠋⠀⠀⠀⠀⠀⠀⢀⣄⣠⣌⣙⠶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀⣴⡚⠋⠉⠁⠈⠉⠑⠺⣦⡀⠀⢀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡿⠁⠀⠀⠀⠀⢀⡠⠟⠃⠀⠀⠀⠀⠀⠀⠀⠀⣀⠉⠉⠉⠉⠙⠻⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠿⠦⠑⠀⠀⠀⠀⠀⠀⠈⠛⢦⡀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢤⣿⠆⠀⠀⠀⠀⠀⣿⣿⡆⠀⠀⠀⠙⢶⣄⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⣠⠴⠚⠉⣸⡇⠀⠀⠀⠀⢲⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢽⠇⠀⠀⠀⠀⠀⠀⣀⣀⣀⣠⣤⡖⠴⢤⡽⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⣠⠞⠁⢀⣤⡴⡿⠀⠀⠀⠀⠀⢸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠿⡷⠀⠀⠀⠒⠈⠉⡠⠄⠒⣲⠖⠁⣠⣣⢟⠦⡀⠀",
    "⠀⠀⠀⠀⢠⠞⠁⣠⠞⠉⠀⠀⡇⠀⠀⠀⠀⠀⢘⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠐⠉⠀⠀⠀⢀⡠⠊⠁⠀⣠⠾⠛⠋⠉⠁⠀⠈⢣⠈⠣",
    "⠀⠀⠀⣰⠋⣠⠞⠁⠀⠀⠀⢸⠷⢶⣆⣀⠀⠀⠀⢹⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⢀⣀⡜⠁⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀",
    "⠀⠀⣰⠃⢰⠃⠀⠀⠀⠀⠀⢸⡄⠈⠙⣿⡉⠉⠉⠉⠛⠳⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣇⡴⠊⡘⠛⠒⢲⣦⡀⠀⠀⠀⠀⠀⠈⠀",
    "⠀⠀⡟⠀⡏⠀⠀⠀⠀⠀⠀⣿⠀⣄⠠⡘⢿⣦⠀⠀⠀⠀⠈⠙⠳⣦⣶⡶⠂⡀⣠⣠⡦⠟⠻⠷⢴⣇⢠⣤⣙⢾⡷⠀⠀⠀⠀⠀⠀⠀",
    "⠀⢸⡇⠀⡇⠀⠀⠀⠀⠀⠀⠙⣧⡟⢷⡵⠼⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⡋⠉⠘⢷⣤⣀⠀⢸⠙⠛⠃⠉⠛⠁⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠘⣧⢰⡇⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡟⢸⡄⢆⢳⣽⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⢠⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠞⢳⠛⢿⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⣼⢣⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⣸⡿⠋⠀⠀                                                   ",
    "               ███████ ██    ██ ███████ ",
    "               ██       ██  ██  ██      ",
    "               █████     ████   █████   ",
    "               ██         ██    ██      ",
    "               ███████    ██    ███████ "

]