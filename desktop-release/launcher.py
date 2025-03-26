import os
import platform
import subprocess


if platform.system() == "Darwin":
    
    # Get the absolute path of the script and its directory
    script_path = os.path.abspath("desktop-release/main.py")
    script_dir = os.path.dirname(script_path)

    # Create the AppleScript command
    apple_script = f'''
    tell application "Terminal"
        do script "cd \\"{script_dir}\\"; printf \\"\\\\e[8;40;100t\\"; conda activate DEEPLABCUT; python3 \\"{script_path}\\""
    end tell
    '''

    # Run the AppleScript
    subprocess.run(["osascript", "-e", apple_script])


if platform.system()== 'Windows':
    os.system('start cmd /k "mode con: cols=100 lines=40 && python main.py"')