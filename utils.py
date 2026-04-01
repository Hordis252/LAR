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
    """Apply HSV masks to filter the target object from the image.

    Masks out the top portion of the image to ignore background noise,
    then applies specific HSV thresholds based on the target.

    Args:
        im (np.ndarray): The raw BGR image from the camera.
        target_object (str): The object to filter ("ball" or "garage").

    Returns:
        Tuple[np.ndarray, np.ndarray]: The binary mask and the modified image.
    """
    im[0:100, :] = 0
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Nové: odstraní šum a zaplní díry od odlesků
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # odstraní šum
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # zaplní díry

    return mask, im

def get_target_position(
    im: np.ndarray, pc: Optional[np.ndarray], num_labels: int, 
    labels: np.ndarray, stats: np.ndarray
) -> Tuple[int, float, bool]:
    """Calculate the center position and distance to the ball/pylon.

    Finds the largest connected component, fits an enclosing circle,
    and applies an offset to determine the target driving center.

    Args:
        im (np.ndarray): The current camera image for drawing debug visuals.
        pc (Optional[np.ndarray]): The pointcloud data for depth estimation.
        num_labels (int): Number of detected components.
        labels (np.ndarray): The labeled image array.
        stats (np.ndarray): Statistics for each connected component.

    Returns:
        Tuple[int, float, bool]: The target X coordinate, distance in meters, 
            and a boolean indicating if the target was found.
    """
    pilon_found = False
    best_center_x = 0
    best_center_y_pos = 0
    distance_to_pilon = 0.0

    if num_labels > 1:
        max_area = 0
        largest_idx = -1
        
        # Find the largest component matching basic aspect ratio criteria
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            if height == 0: continue
            
            aspect_ratio = float(width) / float(height)
            if area > max_area and 0.5 < aspect_ratio < 2.0:
                max_area = area
                largest_idx = i
        
        if max_area > 150 and largest_idx != -1:
            component_mask = (labels == largest_idx).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                (circle_x, circle_y), radius = cv2.minEnclosingCircle(c)
               
                # Apply offset to aim slightly to the left of the true center
                OFFSET_X = 0
                pilon_found = True

                best_center_x = int(circle_x) - OFFSET_X
                best_center_x = max(0, min(WIDTH - 1, best_center_x)) 
                best_center_y_pos = int(circle_y)
                
                # Draw visual debug elements
                # TODO: ORANŽOVÁ PRO EFEKT 
                cv2.circle(im, (int(circle_x), best_center_y_pos), int(radius), (0, 127, 255), 2)
                cv2.circle(im, (best_center_x, best_center_y_pos), 5, (0, 0, 255), -1)

            # Calculate distance using pointcloud data if available
            if pc is not None:
                y_min = max(0, best_center_y_pos - 2)
                y_max = min(480, best_center_y_pos + 3)
                x_min = max(0, best_center_x - 2)
                x_max = min(640, best_center_x + 3)
                z_window = pc[y_min:y_max, x_min:x_max, 2]
                
                z_valid_numbers = z_window[~np.isnan(z_window)]
                valid_z = z_valid_numbers[z_valid_numbers > 0.0]
                
                if len(valid_z) > 0:
                    distance_to_pilon = float(np.median(valid_z))
                cv2.putText(im, f" {distance_to_pilon:.2f} m", (best_center_x + 10, best_center_y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return best_center_x, distance_to_pilon, pilon_found

# --- PointCloud Processing ---

def get_zone_distance(
    pc: Optional[np.ndarray], x_min: float, x_max: float, 
    y_min: float, y_max: float, z_max: float = 3.0, 
    percentile: int = 30, min_points: int = 50
) -> float:
    """Calculate a robust distance to an obstacle within a defined 3D zone.

    Args:
        pc (Optional[np.ndarray]): The pointcloud array.
        x_min (float): Minimum X bound (horizontal).
        x_max (float): Maximum X bound (horizontal).
        y_min (float): Minimum Y bound (vertical).
        y_max (float): Maximum Y bound (vertical).
        z_max (float, optional): Maximum depth to consider. Defaults to 3.0.
        percentile (int, optional): The percentile of depth values to return. Defaults to 30.
        min_points (int, optional): Minimum valid points required. Defaults to 50.

    Returns:
        float: The representative distance to the obstacle in meters, or 999.0 if clear.
    """
    if pc is None:
        return 999.0
        
    x, y, z = pc[:, :, 0], pc[:, :, 1], pc[:, :, 2]
    valid = ~np.isnan(z)
    
    mask = valid & (z < z_max) & (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    data = z[mask]
    
    if data.size > min_points:
        return float(np.percentile(data, percentile))
    return 999.0
