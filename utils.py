Skip to content
Hordis252
LAR
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Security and quality
Insights
Settings
Files
Go to file
t
init.txt
main.py
state_machine.py
utils.py
LAR
/
utils.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

4
Line wrap mode

No wrap
Editing utils.py file contents

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
import cv2
import numpy as np
from typing import Any, Tuple, Optional

# --- Constants & Global Variables ---

# Constants for GREEN ball
#LOWER_GREEN = np.array([40, 70, 50])
#UPPER_GREEN = np.array([90, 255, 255])

LOWER_GREEN = np.array([30, 70, 50])
UPPER_GREEN = np.array([90, 255, 255])

# Camera parameters
WIDTH = 640
HEIGHT = 480

# Global variable for robot starting via button
robot_started: bool = True

# --- Callbacks ---

def button_callback(event: Any) -> None:
    """Handle the physical button press event to start the robot.

    Args:
        event (Any): The button event data containing the state.
    """
    global robot_started
    if event.state == 1:
        print("[BUTTON PRESSED] Starting program...")
        robot_started = True

# --- Image Processing & Computer Vision ---

def filter_image(im: np.ndarray, target_object: str) -> Tuple[np.ndarray, np.ndarray]:
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
 
