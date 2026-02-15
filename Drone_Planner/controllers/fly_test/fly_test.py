"""
Optimized Mavic 2 Pro Controller for Webots
Controller untuk drone FLY_TEST
"""

from controller import Robot, Motor, Gyro, InertialUnit, GPS, Compass, Camera
import math
import time

class Mavic2ProController:
    def __init__(self):
        # Initialize robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Mavic 2 Pro specifications
        self.mass = 1.2  # kg
        self.thrust_to_weight_ratio = 2.0  # Mavic bisa angkat 2x beratnya
        self.max_motor_speed = 600  # RPM maksimum
        
        # Initialize motors - Mavic 2 Pro configuration
        print("Initializing Mavic 2 Pro motors...")
        self.motors = []
        motor_config = [
            ('front left propeller', 'CCW'),   # Front left - Counter Clockwise
            ('front right propeller', 'CW'),    # Front right - Clockwise
            ('rear left propeller', 'CW'),      # Rear left - Clockwise
            ('rear right propeller', 'CCW')     # Rear right - Counter Clockwise
        ]
        
        for motor_name, direction in motor_config:
            motor = self.robot.getDevice(motor_name)
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0)
                # Set motor properties for Mavic
                try:
                    motor.setAvailableTorque(0.2)  # N·m
                    motor.setControlPID(1.0, 0.1, 0.01)  # PID untuk motor control
                except:
                    pass
                self.motors.append(motor)
                print(f"✓ Motor {motor_name} ({direction}): OK")
            else:
                print(f"✗ Motor {motor_name}: NOT FOUND!")
        
        # Initialize sensors
        print("\nInitializing sensors...")
        
        # GPS for altitude and position
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
            print("✓ GPS: OK")
        else:
            print("✗ GPS: NOT FOUND!")
            self.gps = None
        
        # Inertial Unit (IMU) for orientation
        self.imu = self.robot.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.timestep)
            print("✓ IMU: OK")
        else:
            print("✗ IMU: NOT FOUND!")
            self.imu = None
        
        # Gyro for angular velocity
        self.gyro = self.robot.getDevice('gyro')
        if self.gyro:
            self.gyro.enable(self.timestep)
            print("✓ Gyro: OK")
        else:
            print("✗ Gyro: NOT FOUND!")
            self.gyro = None
        
        # Compass for heading
        self.compass = self.robot.getDevice('compass')
        if self.compass:
            self.compass.enable(self.timestep)
            print("✓ Compass: OK")
        else:
            print("✗ Compass: NOT FOUND!")
            self.compass = None
        
        # Camera (optional)
        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)
            print("✓ Camera: OK")
        
        # Control parameters optimized for Mavic 2 Pro
        self.hover_throttle = 0.58  # Throttle untuk hover (58% dari max)
        self.Kp_alt = 0.8           # Proportional gain untuk altitude
        self.Ki_alt = 0.05          # Integral gain untuk altitude
        self.Kd_alt = 0.2           # Derivative gain untuk altitude
        
        self.Kp_roll = 0.6          # Roll control
        self.Kp_pitch = 0.6         # Pitch control
        self.Kp_yaw = 0.4           # Yaw control
        
        # State variables
        self.target_altitude = 0.0
        self.current_altitude = 0.0
        self.altitude_error_sum = 0.0
        self.last_altitude_error = 0.0
        
        self.target_yaw = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        
        # Motor speeds for logging
        self.motor_speeds = [0, 0, 0, 0]
        
        # Calibrate
        self.calibrate_sensors()
        
        print(f"\n{'='*50}")
        print("MAVIC 2 PRO CONTROLLER INITIALIZED")
        print(f"{'='*50}")
        print(f"Robot name: {self.robot.getName()}")
        print(f"Timestep: {self.timestep} ms")
        print(f"Number of motors: {len(self.motors)}")
        
    def calibrate_sensors(self):
        """Calibrate sensors"""
        print("\nCalibrating sensors...")
        for i in range(20):
            self.robot.step(self.timestep)
            if i % 5 == 0:
                print(f"  Calibration step {i+1}/20")
        
        # Reset gyro if available
        if self.gyro:
            self.gyro.getValues()
        
        # Get initial altitude
        if self.gps:
            position = self.gps.getValues()
            self.current_altitude = position[2]
            print(f"  Initial altitude: {self.current_altitude:.3f} m")
        
        print("Calibration complete!")
    
    def get_current_altitude(self):
        """Get current altitude from GPS"""
        if self.gps:
            position = self.gps.getValues()
            return position[2]
        return 0.0
    
    def get_current_attitude(self):
        """Get current roll, pitch, yaw from IMU"""
        if self.imu:
            rpy = self.imu.getRollPitchYaw()
            return {
                'roll': rpy[0],
                'pitch': rpy[1],
                'yaw': rpy[2]
            }
        return {'roll': 0, 'pitch': 0, 'yaw': 0}
    
    def get_angular_velocity(self):
        """Get angular velocity from gyro"""
        if self.gyro:
            return self.gyro.getValues()
        return [0, 0, 0]
    
    def calculate_motor_outputs(self, throttle, roll_cmd, pitch_cmd, yaw_cmd):
        """
        Calculate motor outputs for quadcopter X configuration
        
        Mavic 2 Pro motor configuration:
        FL: CCW, FR: CW, RL: CW, RR: CCW
        
        Mixing formula:
        FL = throttle + pitch + roll - yaw
        FR = throttle + pitch - roll + yaw
        RL = throttle - pitch + roll + yaw
        RR = throttle - pitch - roll - yaw
        """
        # Calculate base motor outputs
        fl = throttle + pitch_cmd + roll_cmd - yaw_cmd
        fr = throttle + pitch_cmd - roll_cmd + yaw_cmd
        rl = throttle - pitch_cmd + roll_cmd + yaw_cmd
        rr = throttle - pitch_cmd - roll_cmd - yaw_cmd
        
        # Clamp to valid range [0, 1]
        fl = max(0.0, min(1.0, fl))
        fr = max(0.0, min(1.0, fr))
        rl = max(0.0, min(1.0, rl))
        rr = max(0.0, min(1.0, rr))
        
        # Convert to motor speed (0 to max_motor_speed)
        fl_speed = fl * self.max_motor_speed
        fr_speed = fr * self.max_motor_speed
        rl_speed = rl * self.max_motor_speed
        rr_speed = rr * self.max_motor_speed
        
        return fl_speed, fr_speed, rl_speed, rr_speed
    
    def set_motor_speeds(self, fl, fr, rl, rr):
        """Set motor speeds with validation"""
        # Store for logging
        self.motor_speeds = [fl, fr, rl, rr]
        
        # Set each motor
        for i, speed in enumerate([fl, fr, rl, rr]):
            if i < len(self.motors):
                # Clamp speed to safe range
                safe_speed = max(0, min(self.max_motor_speed, speed))
                self.motors[i].setVelocity(safe_speed)
    
    def altitude_control(self, target_alt, current_alt, dt):
        """PID control for altitude"""
        error = target_alt - current_alt
        
        # PID terms
        P = self.Kp_alt * error
        self.altitude_error_sum += error * dt
        I = self.Ki_alt * self.altitude_error_sum
        D = self.Kd_alt * (error - self.last_altitude_error) / dt if dt > 0 else 0
        
        # Update last error
        self.last_altitude_error = error
        
        # Calculate throttle (0 to 1)
        throttle = self.hover_throttle + P + I + D
        
        # Clamp throttle
        throttle = max(0.2, min(0.9, throttle))
        
        return throttle
    
    def attitude_control(self, target_rpy, current_rpy, gyro_rates):
        """PID control for attitude"""
        # Calculate errors
        roll_error = target_rpy['roll'] - current_rpy['roll']
        pitch_error = target_rpy['pitch'] - current_rpy['pitch']
        yaw_error = target_rpy['yaw'] - current_rpy['yaw']
        
        # Normalize yaw error to [-π, π]
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi
        
        # P control with gyro damping
        roll_cmd = self.Kp_roll * roll_error - 0.1 * gyro_rates[0]
        pitch_cmd = self.Kp_pitch * pitch_error - 0.1 * gyro_rates[1]
        yaw_cmd = self.Kp_yaw * yaw_error - 0.1 * gyro_rates[2]
        
        # Scale commands
        roll_cmd = max(-0.3, min(0.3, roll_cmd))
        pitch_cmd = max(-0.3, min(0.3, pitch_cmd))
        yaw_cmd = max(-0.2, min(0.2, yaw_cmd))
        
        return roll_cmd, pitch_cmd, yaw_cmd
    
    def stabilize(self, duration_ms):
        """Stabilize drone at current target"""
        steps = int(duration_ms / self.timestep)
        
        for i in range(steps):
            # Get sensor data
            self.current_altitude = self.get_current_altitude()
            current_attitude = self.get_current_attitude()
            gyro_rates = self.get_angular_velocity()
            
            # Calculate time step in seconds
            dt = self.timestep / 1000.0
            
            # Altitude control
            throttle = self.altitude_control(
                self.target_altitude, 
                self.current_altitude, 
                dt
            )
            
            # Attitude control
            target_attitude = {
                'roll': self.target_roll,
                'pitch': self.target_pitch,
                'yaw': self.target_yaw
            }
            roll_cmd, pitch_cmd, yaw_cmd = self.attitude_control(
                target_attitude, current_attitude, gyro_rates
            )
            
            # Calculate motor speeds
            fl, fr, rl, rr = self.calculate_motor_outputs(
                throttle, roll_cmd, pitch_cmd, yaw_cmd
            )
            
            # Apply motor speeds
            self.set_motor_speeds(fl, fr, rl, rr)
            
            # Log every 100 steps
            if i % 100 == 0:
                print(f"[{i*self.timestep/1000:.1f}s] Alt: {self.current_altitude:.2f}m, "
                      f"Throttle: {throttle:.2f}, Motors: {fl:.0f},{fr:.0f},{rl:.0f},{rr:.0f}")
            
            # Step simulation
            if self.robot.step(self.timestep) == -1:
                break
    
    def takeoff(self, target_height=2.0):
        """Takeoff sequence"""
        print(f"\n{'='*50}")
        print(f"TAKEOFF TO {target_height} METERS")
        print(f"{'='*50}")
        
        # Get starting altitude
        start_alt = self.get_current_altitude()
        self.target_altitude = start_alt + target_height
        
        print(f"Start altitude: {start_alt:.3f} m")
        print(f"Target altitude: {self.target_altitude:.3f} m")
        
        # Step 1: Arm motors (low speed)
        print("\n1. Arming motors...")
        for speed in range(0, 200, 20):
            self.set_motor_speeds(speed, speed, speed, speed)
            for _ in range(5):
                self.robot.step(self.timestep)
        
        # Step 2: Ascend
        print("2. Ascending...")
        ascend_time = 4000  # 4 seconds
        self.stabilize(ascend_time)
        
        # Step 3: Hover at target
        print("3. Stabilizing at target altitude...")
        self.stabilize(2000)
        
        final_alt = self.get_current_altitude()
        print(f"\nTakeoff complete! Final altitude: {final_alt:.3f} m")
        
        return True
    
    def move_forward(self, duration=2000, speed=0.2):
        """Move forward"""
        print(f"\nMOVING FORWARD ({duration}ms)")
        
        # Pitch forward slightly
        self.target_pitch = -speed  # Negative pitch = forward
        
        # Move
        self.stabilize(duration)
        
        # Level out
        self.target_pitch = 0.0
        self.stabilize(500)
        
        print("Forward movement complete")
        return True
    
    def move_backward(self, duration=2000, speed=0.2):
        """Move backward"""
        print(f"\nMOVING BACKWARD ({duration}ms)")
        
        # Pitch backward slightly
        self.target_pitch = speed  # Positive pitch = backward
        
        # Move
        self.stabilize(duration)
        
        # Level out
        self.target_pitch = 0.0
        self.stabilize(500)
        
        print("Backward movement complete")
        return True
    
    def yaw_turn(self, angle_degrees):
        """Yaw turn"""
        print(f"\nYAW TURN {angle_degrees}°")
        
        current_attitude = self.get_current_attitude()
        target_yaw_rad = current_attitude['yaw'] + math.radians(angle_degrees)
        
        # Normalize to [-π, π]
        if target_yaw_rad > math.pi:
            target_yaw_rad -= 2 * math.pi
        elif target_yaw_rad < -math.pi:
            target_yaw_rad += 2 * math.pi
        
        self.target_yaw = target_yaw_rad
        
        # Turn
        self.stabilize(3000)  # 3 seconds for turn
        
        # Reset yaw target
        self.target_yaw = 0.0
        
        print(f"Yaw turn complete. Current heading: {math.degrees(current_attitude['yaw']):.1f}°")
        return True
    
    def hover(self, duration_ms):
        """Hover in place"""
        print(f"\nHOVERING ({duration_ms/1000:.1f}s)")
        self.stabilize(duration_ms)
        return True
    
    def landing(self):
        """Landing sequence"""
        print(f"\n{'='*50}")
        print("LANDING SEQUENCE")
        print(f"{'='*50}")
        
        # Step 1: Descend to 0.5m above ground
        print("1. Descending...")
        descend_time = 3000  # 3 seconds
        self.target_altitude = 0.5
        self.stabilize(descend_time)
        
        # Step 2: Final descent
        print("2. Final descent...")
        for i in range(10, 0, -1):
            self.target_altitude = i * 0.05  # 0.5m down to 0m
            self.stabilize(200)
        
        # Step 3: Cut power
        print("3. Cutting motor power...")
        for speed in range(200, 0, -20):
            self.set_motor_speeds(speed, speed, speed, speed)
            for _ in range(5):
                self.robot.step(self.timestep)
        
        # Step 4: Stop motors
        self.set_motor_speeds(0, 0, 0, 0)
        
        print("\n✓ Landing complete!")
        return True
    
    def emergency_stop(self):
        """Emergency stop"""
        print("\n!!! EMERGENCY STOP !!!")
        self.set_motor_speeds(0, 0, 0, 0)
        return False
    
    def run_mission(self):
        """Run complete mission sequence"""
        print(f"\n{'='*60}")
        print("MAVIC 2 PRO MISSION SEQUENCE")
        print(f"{'='*60}")
        
        try:
            # 1. Takeoff 2m
            self.takeoff(2.0)
            
            # 2. Hover 2s
            self.hover(2000)
            
            # 3. Maju 2m
            self.move_forward(2000, 0.15)
            
            # 4. Hover 1s
            self.hover(1000)
            
            # 5. Mundur 2m
            self.move_backward(2000, 0.15)
            
            # 6. Hover 1s
            self.hover(1000)
            
            # 7. Yaw 90°
            self.yaw_turn(90)
            
            # 8. Hover 1s
            self.hover(1000)
            
            # 9. Yaw -90° (back)
            self.yaw_turn(-90)
            
            # 10. Hover 2s
            self.hover(2000)
            
            # 11. Landing
            self.landing()
            
            print(f"\n{'='*60}")
            print("✓ MISSION COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n✗ MISSION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return self.emergency_stop()
        
        return True

# Main execution
if __name__ == "__main__":
    print("Starting Mavic 2 Pro Controller...")
    
    # Create controller
    controller = Mavic2ProController()
    
    # Run mission
    success = controller.run_mission()
    
    if success:
        print("\n✅ Drone mission accomplished!")
    else:
        print("\n❌ Drone mission failed!")
    
    # Keep simulation running
    for _ in range(100):
        if controller.robot.step(controller.timestep) == -1:
            break