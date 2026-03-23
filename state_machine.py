import numpy as np
from typing import Any, Tuple, Optional

from utils import WIDTH

class StateMachine:
    """Control Turtlebot behavior based on defined states

    This class implements a finite state machine for robot navigation,
    object tracking, and automatic garage docking.

    Attributes:
        turtle (Any): Reference to the robot instance (robolab_turtlebot).
        state (str): The current state of the state machine.
        next_state (str): The next state after the current operation/state.
        target_object (str): The object currently being
            searched for ("ball" or "garage").
        crash_detected (bool): Flag indicating if the robot
            has crashed into an obstacle.
        is_finished (bool): Flag indicating if the overall mission
            is successfully completed.
    """

    def __init__(self, turtle: Any) -> None:
        self.turtle = turtle

        # --- State Management ---
        self.state: str = "LEAVE_GARAGE"
        self.next_state: str = ""
        self.target_object: str = "ball"  # Goals: "ball", "garage"
        self.crash_detected: bool = False
        self.is_finished: bool = False

        # --- Target & Control Parameters
        self.ball_target_dist: float = 0.6
        self.kp_approach: float = 0.004#0.003
        self.last_center_x: int = 0

        # --- Odometry & Navigation Tracks
        self.drive_start_x: Optional[float] = None
        self.drive_start_y: Optional[float] = None

        self.turn_accumulated_yaw: float = 0.0
        self.last_turn_yaw: Optional[float] = None

        self.orbit_accumulated_yaw: float = 0.0
        self.last_orbit_yaw: Optional[float] = None

        # --- Sensor flags ---
        self.use_camera: bool = False
        self.use_pointcloud: bool = True

        # --- Sensor Data Cache ---
        self.target_detected: bool = False
        self.center_x: int = 0
        self.target_dist: float = 0.0
        self.obstacle_dist: float = 999.0
        self.left_dist: float = 999.0
        self.right_dist: float = 999.0

        #TODO: Zde proběhla úprava
        # --- Proměnné pro skenování garáže ---
        self.scan_data = []  # Bude ukládat n-tice: (yaw, obstacle_dist)
        self.last_scan_yaw: Optional[float] = None
        self.scan_accumulated_yaw: float = 0.0
        self.target_exit_yaw: Optional[float] = None

    def update(
        self, target_detected: bool, center_x: int, target_dist: float,
        obstacle_dist: float, left_dist: float, right_dist: float
    ) -> Tuple[float, float]:
        """Update the internal state based on the latest sensor data.

        Args:
            target_detected (bool): True if the target is found in the image.
            center_x (int): The X-coordinates of the target's center.
            target_dist (float): Distance to the target in meters.
            obstacle_dist (float): Distance to the obstacle
                directly in front of the robot.
            left_dist (float): Distance to the obstacle on the left.
            right_dist (float): Distance to the obstacle on the right.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """

        # --- Update states with new data ---
        self.target_detected = target_detected
        self.center_x = center_x
        self.target_dist = target_dist
        self.obstacle_dist = obstacle_dist
        self.left_dist = left_dist
        self.right_dist = right_dist

        if self.state == "CRASHED":
            return 0.0, 0.0
        elif self.state == "LOOK":
            return self.state_look()
        elif self.state == "SCAN_GARAGE":
            return self.state_scan_garage()
        elif self.state == "ALIGN_EXIT":
            return self.state_align_exit()
        elif self.state == "LEAVE_GARAGE":
            return self.state_leave_garage()
        elif self.state == "SEARCH":
            return self.state_search()
        elif self.state == "APPROACH":
            return self.state_approach()
        elif self.state == "MOVE_CLOSER":
            return self.state_move_closer()
        elif self.state == "PIVOT":
            return self.state_pivot()
        elif self.state == "ORBIT":
            return self.state_orbit()
        elif self.state == "PIVOT_GARAGE":
            return self.state_pivot_garage()
        elif self.state == "DOCKING":
            return self.state_docking()
        elif self.state == "DONE":
            return self.state_done()
        else:
            return 0.0, 0.0

    def drive_straight(
        self, target_dist: float, target_next_state: str, speed: float = 0.2
    ) -> Tuple[float, float]:
        """Drive the robot straight for defined distance.

        Args:
            target_dist (float): The target distance to travel in meters.
            target_next_state (str): The state to transition to after
                reaching the distance.
            speed (float, optional): Linear speed of the robot. Default 0.2.

        Returns:
            Tuple[float, float]: Velocities (linear, angular)
        """
        x, y, _ = self.turtle.get_odometry()

        if self.drive_start_x is None or self.drive_start_y is None:
            self.drive_start_x = x
            self.drive_start_y = y

        dx = x - self.drive_start_x
        dy = y - self.drive_start_y
        dist = float(np.sqrt(dx**2 + dy**2))

        if dist >= target_dist:
            self.state = target_next_state
            self.drive_start_x = None
            self.drive_start_y = None
            return 0.0, 0.0

        return speed, 0.0

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to the range [-pi, pi].

        Args:
            angle (float): The input angle to normalize in radians.

        Returns:
            float: The normalized angle in radians.
        """
        if angle > np.pi:
            angle -= 2 * float(np.pi)
        elif angle < -np.pi:
            angle += 2 * float(np.pi)
        return angle

    def state_look(self) -> Tuple[float, float]:
        """LOOK:
        Keep the robot stationary. Useful for debugging camera and filters.

        Returns:
            Tuple[float, float]: Zero velocities (linear, angular).
        """
        self.use_camera = True
        self.use_pointcloud = True

        return 0.0, 0.0

    def state_crashed(self, msg: Any) -> None:
        """CRASHED:
        Handle the bumper event when a crash is detected

        Args:
            msg (Any): The bumper sensor message.
        """
        print("[CRASHED]")
        if msg.state == 1:
            self.turtle.cmd_velocity(linear=0.0, angular=0.0)
            self.crash_detected = True
            self.state = "CRASHED"

    def state_scan_garage(self) -> Tuple[float, float]:
        """SCAN_GARAGE:
        Otočí robota o 360 stupňů, namapuje vzdálenosti a najde střed vrat.
        """
        self.use_camera = False
        self.use_pointcloud = True

        _, _, yaw = self.turtle.get_odometry()

        # Inicializace začátku skenu
        if self.last_scan_yaw is None:
            print("[SCAN GARAGE]")
            self.last_scan_yaw = yaw
            self.scan_accumulated_yaw = 0.0
            self.scan_data = []

        # Výpočet ujetého úhlu
        dyaw = self.normalize_angle(yaw - self.last_scan_yaw)
        self.scan_accumulated_yaw += dyaw
        self.last_scan_yaw = yaw

        # Ukládání dat (aktuální úhel a vzdálenost před robotem)
        self.scan_data.append((yaw, self.obstacle_dist))

        # Konec skenování po celém kruhu (2 * pi)
        if abs(self.scan_accumulated_yaw) >= 2 * float(np.pi):
            print("[SCAN GARAGE] Mapování dokončeno. Hledám vrata...")
            
            # --- ZPRACOVÁNÍ SKENU (Hledání mezery) ---
            best_gap = []
            current_gap = []
            
            # Pole spojíme dvakrát za sebe. 
            # Důvod: Kdybychom začali skenovat přesně uprostřed vrat,
            # mezera by se nám rozdělila na začátek a konec pole.
            extended_data = self.scan_data + self.scan_data
            
            for yaw_val, dist in extended_data:
                # 0.7 metru bereme jako "zde je volno / vrata"
                if dist > 0.7: 
                    current_gap.append(yaw_val)
                else:
                    if len(current_gap) > len(best_gap):
                        best_gap = current_gap
                    current_gap = []
                    
            if len(current_gap) > len(best_gap):
                best_gap = current_gap

            # Pokud jsme našli vrata, určíme středový úhel
            if best_gap:
                mid_index = len(best_gap) // 2
                self.target_exit_yaw = best_gap[mid_index]
                print(f"[SCAN GARAGE] Vrata nalezena. Cílový úhel: {self.target_exit_yaw:.2f} rad")
                self.state = "ALIGN_EXIT"
            else:
                print("[SCAN GARAGE] CHYBA: Vrata nenalezena!")
                self.state = "DONE" # Fallback, pokud se něco hodně pokazí
                
            # Úklid proměnných pro případné další použití
            self.last_scan_yaw = None
            self.scan_accumulated_yaw = 0.0
            return 0.0, 0.0

        # Rychlost otáčení při skenování
        return 0.0, 0.3

    def state_align_exit(self) -> Tuple[float, float]:
        """ALIGN_EXIT:
        Natočí robota přesně na vypočítaný střed vrat z předchozího skenu.
        """
        self.use_camera = False
        self.use_pointcloud = False

        _, _, yaw = self.turtle.get_odometry()

        # Odchylka mezi aktuálním natočením a cílem
        yaw_error = self.normalize_angle(self.target_exit_yaw - yaw)

        # Pokud jsme natočeni přesně (odchylka menší než ~3 stupně)
        if abs(yaw_error) < 0.05:
            print("[ALIGN EXIT]")
            self.state = "LEAVE_GARAGE"
            self.turtle.reset_odometry()
            return 0.0, 0.0

        # P-regulátor pro plynulé dotočení
        kp = 1.2
        angular_vel = float(np.clip(yaw_error * kp, -0.5, 0.5))
        
        # Ochrana proti uvíznutí na tření při nízké rychlosti
        if 0 < angular_vel < 0.15: angular_vel = 0.15
        if 0 > angular_vel > -0.15: angular_vel = -0.15

        return 0.0, angular_vel

    def state_leave_garage(self) -> Tuple[float, float]:
        """LEAVE_GARAGE:
        Drive the robot straight out of the garage.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """
        print("[LEAVE GARAGE]")

        self.use_camera = False
        self.use_pointcloud = False

        return self.drive_straight(
            target_dist=0.25,
            target_next_state="SEARCH"
        )

    def state_search(self) -> Tuple[float, float]:
        """SEARCH:
        Search for the current target by spinning in place.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """
        print("[SEARCH]")

        self.use_camera = True
        self.use_pointcloud = False

        if self.last_center_x < (WIDTH / 2):
            angular_vel = 0.35
        else:
            angular_vel = -0.35

        if self.target_detected:
            self.state = "APPROACH"
            return 0.0, 0.0

        return 0.0, angular_vel

    def state_approach(self) -> Tuple[float, float]:
        """APPROACH:
        Navigate the robot toward the detected target.
        For the ball, uses visual servoing. 
        For the garage, uses Odometry to reach a perfect alignment point.
        """
        self.use_camera = True
        self.use_pointcloud = True

        if self.center_x != 0:
            self.last_center_x = self.center_x

        # --- LOGIKA PRO MÍČEK (Vizuální) ---
        if self.target_object == "ball":
            print(f"[APPROACH BALL] Distance: {self.target_dist:.2f} m")
            
            # Pokud ztratíme míček, hledáme ho
            if not self.target_detected:
                self.state = "SEARCH"
                return 0.0, 0.0

            if 0.0 < self.target_dist <= self.ball_target_dist:
                self.state = "MOVE_CLOSER"
                return 0.0, 0.0

            # P-Regulátor pro míček
            ang_error = (WIDTH / 2) - self.center_x
            angular_vel = float(np.clip(self.kp_approach * ang_error, -0.4, 0.4))

            dist_error = self.target_dist - self.ball_target_dist
            linear_vel = float(np.clip(0.5 * dist_error, 0.05, 0.15))

            return linear_vel, angular_vel

        else:
            # Bod přesně 80 cm před garáží, na středové ose
            target_x = 0.4
            target_y = 0.0
            
            x, y, yaw = self.turtle.get_odometry()
            
            # Výpočet rozdílu pozic
            dx = target_x - x
            dy = target_y - y
            distance_to_target = float(np.hypot(dx, dy))
            
            print(f"[APPROACH GARAGE] Naviguji na bod před garáží. Zbývá: {distance_to_target:.2f} m")
            
            # Pokud jsme od dokonalého bodu méně než 2 cm, jsme na místě!
            if distance_to_target < 0.02:
                self.state = "PIVOT_GARAGE"  # Následně se otočíme přesně na garáž
                self.next_state = "DOCKING"
                return 0.0, 0.0
                
            # Spočítáme si úhel, pod kterým tento dokonalý bod leží
            target_angle = float(np.arctan2(dy, dx))
            angle_error = self.normalize_angle(target_angle - yaw)
            
            # Rychlost natáčení směrem k bodu
            angular_vel = float(np.clip(angle_error * 1.5, -0.4, 0.4))
            
            # Rychlost jízdy vpřed
            linear_vel = float(np.clip(distance_to_target * 0.5, 0.05, 0.15))
            
            if abs(angle_error) > 0.4:
                linear_vel = 0.0
                
            return linear_vel, angular_vel

    def state_move_closer(self) -> Tuple[float, float]:
        """MOVE_CLOSER:
        Drive the robot a fixed distance closer to the pylon.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """
        print("[MOVE CLOSER]")

        self.use_camera = False
        self.use_pointcloud = False
        self.next_state = "ORBIT"

        return self.drive_straight(
            target_dist=0.2,
            target_next_state="PIVOT",
            speed=0.15
        )

    def state_pivot(self) -> Tuple[float, float]:
        #TODO: Proběhla úprava kódu
        """PIVOT:
        Rotate the robot 90 degrees to the right using odometry.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """
        print("[PIVOT]")

        self.use_camera = False
        self.use_pointcloud = False

        _, _, yaw = self.turtle.get_odometry()

        if self.last_turn_yaw is None:
            self.last_turn_yaw = yaw
            self.turn_accumulated_yaw = 0.0

        dyaw = self.normalize_angle(yaw - self.last_turn_yaw)

        self.turn_accumulated_yaw += dyaw
        self.last_turn_yaw = yaw

        target_angle = -(float(np.pi) / 2)

        error = target_angle - self.turn_accumulated_yaw

        if abs(error) < 0.05:
            self.state = self.next_state
            self.last_turn_yaw = None
            self.turn_accumulated_yaw = 0.0
            return 0.0, 0.0

        kp = 1.5
        angular_vel = error * kp

        angular_vel = float(np.clip(angular_vel, -0.8, -0.2))

        return 0.0, angular_vel

    def state_orbit(self) -> Tuple[float, float]:
        """ORBIT:
        Drive the robot in a full 360-degree circle
        around the object using odometry.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """
        print("[ORBIT]")

        self.use_camera = False
        self.use_pointcloud = False

        _, _, yaw = self.turtle.get_odometry()

        if self.last_orbit_yaw is None:
            self.last_orbit_yaw = yaw
            self.orbit_accumulated_yaw = 0.0

        dyaw = self.normalize_angle(yaw - self.last_orbit_yaw)

        self.orbit_accumulated_yaw += dyaw
        self.last_orbit_yaw = yaw

        if abs(self.orbit_accumulated_yaw) >= 2 * float(np.pi):
            self.state = "PIVOT"
            self.next_state = "APPROACH"
            self.target_object = "garage"

            self.last_orbit_yaw = None
            self.orbit_accumulated_yaw = 0.0

            return 0.0, 0.0

        return 0.15, 0.45

    def state_pivot_garage(self) -> Tuple[float, float]:
        """PIVOT_GARAGE:
        Rotate the robot to face the garage perfectly (global yaw = PI).
        """
        print("[PIVOT GARAGE]")

        self.use_camera = False
        self.use_pointcloud = False

        # Zjistíme aktuální natočení ve světě
        _, _, yaw = self.turtle.get_odometry()

        # Chceme koukat přesně zpátky do garáže (úhel PI = 180 stupňů)
        target_yaw = float(np.pi)
        
        # O kolik se ještě musíme dotočit
        yaw_error = self.normalize_angle(target_yaw - yaw)

        # Tolerance: pokud jsme od cíle méně než 0.1 radiánu (~5.7 stupně), končíme točení
        if abs(yaw_error) < 0.1:
            self.state = self.next_state
            return 0.0, 0.0

        # Plynulé otáčení - P-regulátor
        angular_vel = float(np.clip(yaw_error * 1.5, -0.6, 0.6))

        return 0.0, angular_vel

    def state_docking(self) -> Tuple[float, float]:
        """DOCKING:
        Docking into garage using PointCloud data.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """
        print("[DOCKING]")

        self.use_camera = False
        self.use_pointcloud = True

        if self.obstacle_dist <= 0.4:
            self.state = "DONE"
            return 0.0, 0.0

        dist_left = min(self.left_dist, 1.5)
        dist_right = min(self.right_dist, 1.5)

        diff = dist_left - dist_right
        angular_vel = float(np.clip(diff * 1, -0.4, 0.4))

        return 0.15, angular_vel

    def state_done(self) -> Tuple[float, float]:
        """DONE:
        Stop the robot and mark the overall mission as completed.

        Returns:
            Tuple[float, float]: Velocities (linear, angular).
        """
        print("[DONE]")

        self.is_finished = True

        return 0.0, 0.0
