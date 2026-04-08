from robolab_turtlebot import Turtlebot, Rate
import cv2
import numpy as np

from state_machine import StateMachine
from utils import *

cameras = False

def main() -> None:
    """Main execution loop for the Turtlebot.

    Initializes the robot, sensors, and state machine. Continuously
    fetches sensor data, processes it via computer vision and point cloud
    utilities, and updates the state machine to control the robot's velocities.
    """
    # --- Robot Initialization ---
    turtle = Turtlebot(rgb=True, pc=True)
    print("[INIT] Waiting for sensor initialization...")
    turtle.wait_for_point_cloud()
    print("[INIT] Sensors initialized, starting!")
    
    turtle.reset_odometry()
    
    # --- State Machine Setup ---
    state_machine = StateMachine(turtle)
    turtle.register_bumper_event_cb(state_machine.state_crashed)
    turtle.register_button_event_cb(button_callback)

    rate = Rate(10)
    
    if cameras == True:
        cv2.namedWindow("Kamera")
        cv2.namedWindow("Maska")
    
    # --- Main Control Loop ---
    while not turtle.is_shutting_down() and not state_machine.crash_detected and not state_machine.is_finished:
        if not robot_started:
            rate.sleep()
            continue
        
        # --- Sensor Data Acquisition ---
        im = None
        pc = None
        
        if state_machine.use_camera:
            im = turtle.get_rgb_image()
            if im is None: 
                rate.sleep()
                continue
                
        if state_machine.use_pointcloud:
            pc = turtle.get_point_cloud()
            if pc is None: 
                rate.sleep()
                continue

        # --- Default Sensor Values ---
        obstacle_dist: float = 999.0
        left_dist: float = 999.0
        right_dist: float = 999.0
        target_found: bool = False
        center_x: int = 0
        target_distance: float = 0.0

        # --- Computer Vision Processing ---
        if state_machine.use_camera and im is not None:
            # Filter image based on the current target object (ball or garage)
            mask, im_work = filter_image(im, state_machine.target_object)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            
            # Decide which function to use based on the target
            if state_machine.target_object == "ball":
                # NOTE: 'centroids' parameter removed as it is not needed for the ball/pylon
                center_x, target_distance, target_found = get_target_position(
                    im_work, pc, num_labels, labels, stats
                )

            if cameras == True:
                cv2.imshow("Maska", mask)
                cv2.imshow("Kamera", im_work)
                cv2.waitKey(1)

        # --- PointCloud Processing ---
        if state_machine.use_pointcloud and pc is not None:
            obstacle_dist = get_zone_distance(pc, x_min=-0.15, x_max=0.15, y_min=-0.2, y_max=0.05, percentile=70)

            left_dist = get_zone_distance(pc, -0.3, -0.1, -0.2, 0.1, z_max=1.5, percentile=10, min_points=30)
            right_dist = get_zone_distance(pc, 0.1, 0.3, -0.2, 0.1, z_max=1.5, percentile=10, min_points=30)
            
        # --- State Machine Update & Robot Control ---
        linear_vel, angular_vel = state_machine.update(
            target_found, center_x, target_distance, obstacle_dist, left_dist, right_dist
        )
        
        turtle.cmd_velocity(linear=linear_vel, angular=angular_vel)

        rate.sleep()

    # --- Shutdown Sequence ---
    print("[SHUTDOWN] Stopping robot...")
    turtle.cmd_velocity(linear=0.0, angular=0.0)

if __name__ == '__main__':
    main()
