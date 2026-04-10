import numpy as np
from typing import Any, Tuple, Optional

from utils import WIDTH

class StateMachine:
    """Control Turtlebot behavior based on defined states"""

    def __init__(self, turtle: Any) -> None:
        self.turtle = turtle

        # --- State Management ---
        self.state: str = "SCAN_GARAGE"
        self.next_state: str = ""
        self.target_object: str = "ball"  # Goals: "ball", "garage"
        self.crash_detected: bool = False
        self.is_finished: bool = False

        # --- Target & Control Parameters ---
        self.ball_target_dist: float = 0.65
        self.kp_approach: float = 0.003
        self.last_center_x: int = 0

        # --- Rampa zrychlení (Plynulý rozjezd) ---
        self.current_lin_vel: float = 0.0
        self.current_ang_vel: float = 0.0
        self.max_lin_accel: float = 0.03  # Max změna lin. rychlosti za jeden update (zvyš pro prudší rozjezd)
        self.max_ang_accel: float = 0.06   # Max změna úhlové rychlosti za jeden update

        # --- Odometry & Navigation Tracks ---
        self.drive_start_x: Optional[float] = None
        self.drive_start_y: Optional[float] = None

        self.pivot_accumulated_yaw: float = 0.0
        self.last_pivot_yaw: Optional[float] = None

        self.orbit_accumulated_yaw: float = 0.0
        self.last_orbit_yaw: Optional[float] = None

        self.scan_accumulated_yaw: float = 0.0
        self.last_scan_yaw: Optional[float] = None

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

        # --- Proměnné pro skenování garáže ---
        self.scan_data = []
        self.target_exit_yaw: Optional[float] = None

        # --- Paměť pro pozici míčku z garáže ---
        self.best_ball_error: float = 999.0
        self.memorized_ball_yaw: Optional[float] = None
        self.relative_ball_yaw: float = 0.0

    def update(
        self, target_detected: bool, center_x: int, target_dist: float,
        obstacle_dist: float, left_dist: float, right_dist: float
    ) -> Tuple[float, float]:
        """Update the internal state based on the latest sensor data."""

        self.target_detected = target_detected
        self.center_x = center_x
        self.target_dist = target_dist
        self.obstacle_dist = obstacle_dist
        self.left_dist = left_dist
        self.right_dist = right_dist

        old_state = self.state

        target_lin, target_ang = 0.0, 0.0

        if self.state == "CRASHED":
            self.current_lin_vel, self.current_ang_vel = 0.0, 0.0
            return 0.0, 0.0
        elif self.state == "DONE":
            self.current_lin_vel, self.current_ang_vel = 0.0, 0.0
            return self.state_done()
            
        elif self.state == "LOOK":
            target_lin, target_ang = self.state_look()
        elif self.state == "SEARCH_START":
            target_lin, target_ang = self.state_search_start()
        elif self.state == "SCAN_GARAGE":
            target_lin, target_ang = self.state_scan_garage()
        elif self.state == "ALIGN_EXIT":
            target_lin, target_ang = self.state_align_exit()
        elif self.state == "LEAVE_GARAGE":
            target_lin, target_ang = self.state_leave_garage()
        elif self.state == "SEARCH":
            target_lin, target_ang = self.state_search()
        elif self.state == "APPROACH_BALL":
            target_lin, target_ang = self.state_approach_ball()
        elif self.state == "APPROACH_GARAGE":
            target_lin, target_ang = self.state_approach_garage()
        elif self.state == "MOVE_CLOSER":
            target_lin, target_ang = self.state_move_closer()
        elif self.state == "PIVOT":
            target_lin, target_ang = self.state_pivot()
        elif self.state == "ORBIT":
            target_lin, target_ang = self.state_orbit()
        elif self.state == "PIVOT_GARAGE":
            target_lin, target_ang = self.state_pivot_garage()
        elif self.state == "DOCKING":
            target_lin, target_ang = self.state_docking()

        if self.state != old_state:
            print(f"[TRANSITION] Měním stav z {old_state} na {self.state} -> Nuluji rychlosti")
            self.current_lin_vel = 0.0
            self.current_ang_vel = 0.0
            # Vracíme 0, aby robot dostal okamžitý příkaz zastavit motory 
            # (než se v dalším tiku začne rozjíždět na nový cíl)
            return 0.0, 0.0

        # Aplikování plynulého rozjezdu a brždění
        return self.smooth_velocity(target_lin, target_ang)

    def smooth_velocity(self, target_lin: float, target_ang: float) -> Tuple[float, float]:
        """Zabrání prudkému cukání - omezuje, o kolik se může rychlost změnit v jednom kroku."""
        # Lineární zrychlení/zpomalení
        if target_lin > self.current_lin_vel:
            self.current_lin_vel = min(target_lin, self.current_lin_vel + self.max_lin_accel)
        elif target_lin < self.current_lin_vel:
            self.current_lin_vel = max(target_lin, self.current_lin_vel - self.max_lin_accel)
            
        # Úhlové zrychlení/zpomalení
        if target_ang > self.current_ang_vel:
            self.current_ang_vel = min(target_ang, self.current_ang_vel + self.max_ang_accel)
        elif target_ang < self.current_ang_vel:
            self.current_ang_vel = max(target_ang, self.current_ang_vel - self.max_ang_accel)
            
        return self.current_lin_vel, self.current_ang_vel

    def drive_straight(self, target_dist: float, target_next_state: str, speed: float = 0.2) -> Tuple[float, float]:
        x, y, _ = self.turtle.get_odometry()
        if self.drive_start_x is None or self.drive_start_y is None:
            self.drive_start_x = x
            self.drive_start_y = y

        dx = x - self.drive_start_x
        dy = y - self.drive_start_y
        dist = float(np.sqrt(dx**2 + dy**2))

        # Zde si definujeme naši odchylku (kolik metrů ještě zbývá ujet)
        dist_error = target_dist - dist

        # Pokud jsme v toleranci (např. chybí méně než 1 cm), jsme v cíli
        if dist_error <= 0.01:
            self.state = target_next_state
            self.drive_start_x = None
            self.drive_start_y = None
            return 0.0, 0.0

        linear_vel = self.robust_p_control(
            error=dist_error, 
            kp=1.0, 
            min_vel=0.08, 
            max_vel=speed, 
            tolerance=0.01
        )

        return linear_vel, 0.0

    def normalize_angle(self, angle: float) -> float:
        if angle > np.pi:
            angle -= 2 * float(np.pi)
        elif angle < -np.pi:
            angle += 2 * float(np.pi)
        return angle

    def robust_p_control(self, error: float, kp: float, min_vel: float, max_vel: float, tolerance: float = 0.01) -> float:
        if abs(error) <= tolerance:
            return 0.0
            
        vel = error * kp
        sign = 1.0 if vel >= 0 else -1.0
        abs_vel = abs(vel)
        
        if abs_vel < min_vel:
            abs_vel = min_vel
        elif abs_vel > max_vel:
            abs_vel = max_vel
            
        return float(sign * abs_vel)

    def state_look(self) -> Tuple[float, float]:
        self.use_camera = True
        self.use_pointcloud = True
        return 0.0, 0.0

    def state_crashed(self, msg: Any) -> None:
        print("[CRASHED]")
        if msg.state == 1:
            self.turtle.cmd_velocity(linear=0.0, angular=0.0)
            self.crash_detected = True
            self.state = "CRASHED"

    def state_scan_garage(self) -> Tuple[float, float]:
        self.use_camera = True
        self.use_pointcloud = True

        _, _, yaw = self.turtle.get_odometry()

        if self.last_scan_yaw is None:
            print("[SCAN GARAGE]")
            self.last_scan_yaw = yaw
            self.scan_accumulated_yaw = 0.0
            self.scan_data = []
            self.best_ball_error = 999.0
            self.memorized_ball_yaw = None

        dyaw = self.normalize_angle(yaw - self.last_scan_yaw)
        self.scan_accumulated_yaw += dyaw
        self.last_scan_yaw = yaw
        self.scan_data.append((yaw, self.obstacle_dist))

        if self.target_detected:
            pixel_error = abs((WIDTH / 2) - self.center_x)
            if pixel_error < self.best_ball_error:
                self.best_ball_error = pixel_error
                self.memorized_ball_yaw = yaw

        if abs(self.scan_accumulated_yaw) >= 2 * float(np.pi):
            print("[SCAN GARAGE]")
            best_gap = []
            current_gap = []
            extended_data = self.scan_data + self.scan_data
            
            for yaw_val, dist in extended_data:
                if dist > 0.7: 
                    current_gap.append(yaw_val)
                else:
                    if len(current_gap) > len(best_gap):
                        best_gap = current_gap
                    current_gap = []
                    
            if len(current_gap) > len(best_gap):
                best_gap = current_gap

            if best_gap:
                mid_index = len(best_gap) // 2
                self.target_exit_yaw = best_gap[mid_index]
                print(f"[SCAN GARAGE] Vrata nalezena. Cílový úhel: {self.target_exit_yaw:.2f} rad")
                
                if self.memorized_ball_yaw is not None:
                    self.relative_ball_yaw = self.normalize_angle(self.memorized_ball_yaw - self.target_exit_yaw)
                    print(f"[SCAN GARAGE] Míček nalezen a uložen na relativním úhlu: {self.relative_ball_yaw:.2f} rad vůči výjezdu.")
                else:
                    print("[SCAN GARAGE] VAROVÁNÍ: Míček nebyl během skenu vůbec spatřen!")
                    
                self.state = "ALIGN_EXIT"
            else:
                print("[SCAN GARAGE] CHYBA: Vrata nenalezena!")
                self.state = "DONE"
                
            self.last_scan_yaw = None
            self.scan_accumulated_yaw = 0.0
            return 0.0, 0.0

        return 0.0, 0.25

    def state_align_exit(self) -> Tuple[float, float]:
        self.use_camera = False
        self.use_pointcloud = False

        _, _, yaw = self.turtle.get_odometry()
        yaw_error = self.normalize_angle(self.target_exit_yaw - yaw)

        if abs(yaw_error) < 0.08:
            print("[ALIGN EXIT]")
            self.state = "LEAVE_GARAGE"
            self.turtle.reset_odometry()
            return 0.0, 0.0

        angular_vel = self.robust_p_control(yaw_error, kp=1.5, min_vel=0.2, max_vel=0.4, tolerance=0.08)
        return 0.0, angular_vel

    def state_leave_garage(self) -> Tuple[float, float]:
        print("[LEAVE GARAGE]")
        self.use_camera = False
        self.use_pointcloud = False
        
        if abs(self.relative_ball_yaw) < (float(np.pi)/6):
            distance = 0.2
        else:
            distance = 0.4
        
        return self.drive_straight(target_dist=distance, target_next_state="SEARCH")

    def state_search(self) -> Tuple[float, float]:
        print("[SEARCH]")
        self.use_camera = True
        self.use_pointcloud = True

        if not self.target_detected:
            if self.last_center_x != 0:
                angular_vel = 0.4 if self.last_center_x < (WIDTH / 2) else -0.4
            else:
                angular_vel = 0.4 if self.relative_ball_yaw > 0 else -0.4
            return 0.0, angular_vel

        else:
            error = (WIDTH / 2) - self.center_x
            if abs(error) < 15:
                print("[SEARCH] Cíl vycentrován! Přecházím na APPROACH.")
                self.state = "APPROACH_BALL"
                return 0.0, 0.0
            
            # Zvýšeno min_vel na 0.2
            angular_vel = self.robust_p_control(error, kp=self.kp_approach, min_vel=0.2, max_vel=0.4, tolerance=15.0)
            return 0.0, angular_vel

    def state_approach_ball(self) -> Tuple[float, float]:
        self.use_camera = True
        self.use_pointcloud = True

        if self.center_x != 0:
            self.last_center_x = self.center_x

        if self.target_object == "ball":
            print(f"[APPROACH BALL] Distance: {self.target_dist:.2f} m")
            
            if not self.target_detected:
                self.state = "SEARCH"
                return 0.0, 0.0

            if 0.0 < self.target_dist < self.ball_target_dist:
                self.state = "MOVE_CLOSER"
                return 0.0, 0.0

            ang_error = (WIDTH / 2) - self.center_x
            angular_vel = self.robust_p_control(ang_error, kp=self.kp_approach, min_vel=0.1, max_vel=0.3, tolerance=15.0)

            dist_error = max(0.0, self.target_dist - self.ball_target_dist)
            linear_vel = self.robust_p_control(dist_error, kp=0.6, min_vel=0.1, max_vel=0.2, tolerance=0.03)

            return linear_vel, angular_vel

    def state_approach_garage(self) -> Tuple[float, float]:
        target_x, target_y = 0.4, 0.0
        x, y, yaw = self.turtle.get_odometry()
        dx = target_x - x
        dy = target_y - y
        distance_to_target = float(np.hypot(dx, dy))
        
        print("[APPROACH_GARAGE]")
        
        if distance_to_target < 0.03:
            self.state = "PIVOT_GARAGE"
            self.next_state = "DOCKING"
            return 0.0, 0.0
            
        target_angle = float(np.arctan2(dy, dx))
        angle_error = self.normalize_angle(target_angle - yaw)
        
        angular_vel = self.robust_p_control(angle_error, kp=1.5, min_vel=0.1, max_vel=0.3, tolerance=0.03)
        linear_vel = self.robust_p_control(distance_to_target, kp=0.5, min_vel=0.08, max_vel=0.2, tolerance=0.03)
        
        if abs(angle_error) > 0.4:
            linear_vel = 0.0
            
        return linear_vel, angular_vel

    def state_move_closer(self) -> Tuple[float, float]:
        print("[MOVE CLOSER]")

        self.use_camera = False
        self.use_pointcloud = False
        self.next_state = "ORBIT"

        return self.drive_straight(
            target_dist=0.3,
            target_next_state="PIVOT",
            speed=0.15
        )

    def state_pivot(self) -> Tuple[float, float]:
        print("[PIVOT]")

        self.use_camera = False
        self.use_pointcloud = False

        _, _, yaw = self.turtle.get_odometry()

        if self.last_pivot_yaw is None:
            self.last_pivot_yaw = yaw
            self.pivot_accumulated_yaw = 0.0

        dyaw = self.normalize_angle(yaw - self.last_pivot_yaw)

        self.pivot_accumulated_yaw += dyaw
        self.last_pivot_yaw = yaw

        target_angle = -(float(np.pi) / 2)

        error = target_angle - self.pivot_accumulated_yaw

        if abs(error) < 0.05:
            self.state = self.next_state
            self.last_pivot_yaw = None
            self.pivot_accumulated_yaw = 0.0
            return 0.0, 0.0

        angular_vel = self.robust_p_control(error, kp=1.5, min_vel=0.25, max_vel=0.8, tolerance=0.05)

        return 0.0, angular_vel

    def state_orbit(self) -> Tuple[float, float]:
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
            self.next_state = "APPROACH_GARAGE"
            self.target_object = "garage"

            self.last_orbit_yaw = None
            self.orbit_accumulated_yaw = 0.0

            return 0.0, 0.0

        return 0.17, 0.45

    def state_pivot_garage(self) -> Tuple[float, float]:
        print("[PIVOT_GARAGE]")
        self.use_camera = False
        self.use_pointcloud = False

        _, _, yaw = self.turtle.get_odometry()
        target_yaw = float(np.pi)
        yaw_error = self.normalize_angle(target_yaw - yaw)

        if abs(yaw_error) < 0.1:
            self.state = self.next_state
            return 0.0, 0.0

        # Zvýšeno min_vel na 0.2
        angular_vel = self.robust_p_control(yaw_error, kp=1.5, min_vel=0.15, max_vel=0.5, tolerance=0.1)
        return 0.0, angular_vel

    def state_docking(self) -> Tuple[float, float]:
        print("[DOCKING]")
        self.use_camera = False
        self.use_pointcloud = True

        if self.obstacle_dist <= 0.32:
            self.state = "DONE"
            return 0.0, 0.0

        dist_left = min(self.left_dist, 1.5)
        dist_right = min(self.right_dist, 1.5)

        diff = dist_left - dist_right
        angular_vel = self.robust_p_control(error=diff, kp=1.0, min_vel=0.0, max_vel=0.4, tolerance=0.05)

        dist_error = self.obstacle_dist - 0.32
        linear_vel = self.robust_p_control(dist_error, kp=0.5, min_vel=0.08, max_vel=0.25, tolerance=0.03)

        return linear_vel, angular_vel

    def state_done(self) -> Tuple[float, float]:
        print("[DONE]")
        self.is_finished = True
        return 0.0, 0.0