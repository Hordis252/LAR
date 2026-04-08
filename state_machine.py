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

        # --- Target & Control Parameters
        self.ball_target_dist: float = 0.8
        self.kp_approach: float = 0.003
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

        # --- Proměnné pro skenování garáže ---
        self.scan_data = []
        self.last_scan_yaw: Optional[float] = None
        self.scan_accumulated_yaw: float = 0.0
        self.target_exit_yaw: Optional[float] = None

        # --- Proměnné pro plynulý oblet (Míček) ---
        self.ball_x: Optional[float] = None
        self.ball_y: Optional[float] = None
        self.orbit_radius: float = 0.35
        self.orbit_start_yaw: Optional[float] = None

        # --- Paměť pro pozici míčku z garáže ---
        self.best_ball_error: float = 999.0
        self.memorized_ball_yaw: Optional[float] = None
        self.relative_ball_yaw: float = 0.0

        # --- PLYNULÝ NÁBĚH (Ramping) ---
        self.last_lin_vel: float = 0.0
        self.last_ang_vel: float = 0.0
        # Tyto hodnoty určují, o kolik maximálně se smí rychlost změnit za jeden tik smyčky.
        # Menší číslo = plynulejší, ale línější rozjezd.
        self.max_lin_accel: float = 0.03  
        self.max_ang_accel: float = 0.08  

    def apply_ramp(self, target: float, current: float, max_step: float) -> float:
        """Plynulý náběh (omezení zrychlení). Nedovolí rychlosti skočit o více než max_step."""
        diff = target - current
        if abs(diff) > max_step:
            return current + (max_step if diff > 0 else -max_step)
        return target

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

        # 1. Zjistíme CÍLOVOU rychlost od stavů
        target_lin, target_ang = 0.0, 0.0

        if self.state == "CRASHED":
            target_lin, target_ang = 0.0, 0.0
        elif self.state == "LOOK":
            target_lin, target_ang = self.state_look()
        elif self.state == "SCAN_GARAGE":
            target_lin, target_ang = self.state_scan_garage()
        elif self.state == "ALIGN_EXIT":
            target_lin, target_ang = self.state_align_exit()
        elif self.state == "LEAVE_GARAGE":
            target_lin, target_ang = self.state_leave_garage()
        elif self.state == "SEARCH":
            target_lin, target_ang = self.state_search()
        elif self.state == "APPROACH":
            target_lin, target_ang = self.state_approach()
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
        elif self.state == "DONE":
            target_lin, target_ang = self.state_done()

        if self.state in ["ORBIT", "LEAVE_GARAGE", "DOCKING"]:
            self.last_lin_vel = target_lin
            self.last_ang_vel = target_ang
            return target_lin, target_ang

        # 2. Aplikujeme PLYNULÝ NÁBĚH před odesláním
        lin_vel = self.apply_ramp(target_lin, self.last_lin_vel, self.max_lin_accel)
        ang_vel = self.apply_ramp(target_ang, self.last_ang_vel, self.max_ang_accel)

        # Uložíme pro další tik
        self.last_lin_vel = lin_vel
        self.last_ang_vel = ang_vel

        return lin_vel, ang_vel

    def drive_straight(self, target_dist: float, target_next_state: str, speed: float = 0.2) -> Tuple[float, float]:
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
        if angle > np.pi:
            angle -= 2 * float(np.pi)
        elif angle < -np.pi:
            angle += 2 * float(np.pi)
        return angle

    def regulator(
        self, error: float, current_vel: float, kp: float, damping: float = 0.8, 
        min_vel: float, max_vel: float, tolerance: float = 0.01
    ) -> float:
        """P-regulátor s integrovaným plynulým nasčítáváním (Spring-Damper)."""
        
        # 1. P-složka funguje jako zrychlení (síla táhnoucí k cíli)
        acceleration = error * kp
        
        # 2. Nasčítání! Plynulé tlumení staré rychlosti + přidání nového tahu
        # damping = 0.0 znamená okamžité změny (žádná plynulost)
        # damping = 0.8 znamená hezky plynulý dojezd/rozjezd
        new_vel = (current_vel * damping) + acceleration
        
        # 3. Pokud jsme v cíli (v toleranci), chceme už jen plynule dobrzdit na nulu
        if abs(error) <= tolerance:
            new_vel = current_vel * damping # Necháme působit jen tření
            if abs(new_vel) < min_vel:
                return 0.0

        # 4. Standardní omezení na tvé bezpečné limity
        sign = 1.0 if new_vel >= 0 else -1.0
        abs_vel = abs(new_vel)
        
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
            print("[SCAN GARAGE] Začínám mapovat vrata a hledat míček...")
            self.last_scan_yaw = yaw
            self.scan_accumulated_yaw = 0.0
            self.scan_data = []
            self.best_ball_error = 999.0
            self.memorized_ball_yaw = None

        dyaw = self.normalize_angle(yaw - self.last_scan_yaw)
        self.scan_accumulated_yaw += dyaw
        self.last_scan_yaw = yaw
        self.scan_data.append((yaw, self.obstacle_dist))

        if self.target_detected and self.target_object == "ball":
            pixel_error = abs((WIDTH / 2) - self.center_x)
            if pixel_error < self.best_ball_error:
                self.best_ball_error = pixel_error
                self.memorized_ball_yaw = yaw

        if abs(self.scan_accumulated_yaw) >= 2 * float(np.pi):
            print("[SCAN GARAGE] Mapování dokončeno. Zpracovávám data...")
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

        angular_vel = self.regulator(yaw_error, kp=1.5, min_vel=0.15, max_vel=0.4, tolerance=0.08)
        return 0.0, angular_vel

    def state_leave_garage(self) -> Tuple[float, float]:
        print("[LEAVE GARAGE]")
        self.use_camera = False
        self.use_pointcloud = False
        return self.drive_straight(target_dist=0.25, target_next_state="SEARCH")

    def state_search(self) -> Tuple[float, float]:
        print("[SEARCH]")
        self.use_camera = True
        self.use_pointcloud = False

        if not self.target_detected:
            if self.last_center_x != 0:
                angular_vel = 0.4 if self.last_center_x < (WIDTH / 2) else -0.4
            else:
                angular_vel = 0.4 if self.relative_ball_yaw > 0 else -0.4
            return 0.05, angular_vel

        else:
            error = (WIDTH / 2) - self.center_x
            if abs(error) < 15:
                print("[SEARCH] Cíl vycentrován! Přecházím na APPROACH.")
                self.state = "APPROACH"
                return 0.0, 0.0
            
            angular_vel = self.regulator(error, kp=self.kp_approach, min_vel=0.15, max_vel=0.4, tolerance=15.0)
            return 0.0, angular_vel

    def state_approach(self) -> Tuple[float, float]:
        self.use_camera = True
        self.use_pointcloud = True

        if self.center_x != 0:
            self.last_center_x = self.center_x

        if self.target_object == "ball":
            print(f"[APPROACH BALL] Distance: {self.target_dist:.2f} m")
            
            if not self.target_detected:
                self.state = "SEARCH"
                return 0.0, 0.0

            if 0.0 < self.target_dist <= self.ball_target_dist:
                x, y, yaw = self.turtle.get_odometry()
                self.ball_x = x + self.target_dist * np.cos(yaw)
                self.ball_y = y + self.target_dist * np.sin(yaw)
                self.orbit_start_yaw = yaw
                print(f"[APPROACH] Míček zaměřen na: X={self.ball_x:.2f}, Y={self.ball_y:.2f}")
                self.state = "MOVE_CLOSER"
                return 0.0, 0.0

            # ZMĚNA: Nahrazen np.clip robustním regulátorem + zakázáno couvání!
            ang_error = (WIDTH / 2) - self.center_x
            angular_vel = self.regulator(ang_error, kp=self.kp_approach, min_vel=0.15, max_vel=0.4, tolerance=15.0)

            # Pojistka: dist_error nesmí být menší než 0, robot tím pádem nikdy nebude couvat
            dist_error = max(0.0, self.target_dist - self.ball_target_dist)
            linear_vel = self.regulator(dist_error, kp=0.6, min_vel=0.08, max_vel=0.2, tolerance=0.03)

            return linear_vel, angular_vel

        else:
            target_x, target_y = 0.4, 0.0
            x, y, yaw = self.turtle.get_odometry()
            dx = target_x - x
            dy = target_y - y
            distance_to_target = float(np.hypot(dx, dy))
            
            print(f"[APPROACH GARAGE] Naviguji na bod před garáží. Zbývá: {distance_to_target:.2f} m")
            
            if distance_to_target < 0.03:
                self.state = "PIVOT_GARAGE"
                self.next_state = "DOCKING"
                return 0.0, 0.0
                
            target_angle = float(np.arctan2(dy, dx))
            angle_error = self.normalize_angle(target_angle - yaw)
            
            angular_vel = self.regulator(angle_error, kp=1.5, min_vel=0.15, max_vel=0.4, tolerance=0.03)
            linear_vel = self.regulator(distance_to_target, kp=0.5, min_vel=0.08, max_vel=0.2, tolerance=0.03)
            
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
            target_dist=0.25,
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

        #TODO: bylo angular_vel = float(np.clip(angular_vel, -0.8, -0.2))
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
        print("[PIVOT GARAGE]")
        self.use_camera = False
        self.use_pointcloud = False

        _, _, yaw = self.turtle.get_odometry()
        target_yaw = float(np.pi)
        yaw_error = self.normalize_angle(target_yaw - yaw)

        if abs(yaw_error) < 0.1:
            self.state = self.next_state
            return 0.0, 0.0

        angular_vel = self.regulator(yaw_error, kp=1.5, min_vel=0.15, max_vel=0.5, tolerance=0.1)
        return 0.0, angular_vel

    def state_docking(self) -> Tuple[float, float]:
        print("[DOCKING]")
        self.use_camera = False
        self.use_pointcloud = True

        if self.obstacle_dist <= 0.4:
            self.state = "DONE"
            return 0.0, 0.0

        dist_left = min(self.left_dist, 1.5)
        dist_right = min(self.right_dist, 1.5)

        diff = dist_left - dist_right
        angular_vel = float(np.clip(diff * 1.5, -0.4, 0.4))

        return 0.15, angular_vel

    def state_done(self) -> Tuple[float, float]:
        print("[DONE]")
        self.is_finished = True
        return 0.0, 0.0
