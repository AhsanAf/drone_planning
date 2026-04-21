from controller import Supervisor
import math

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.008

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def __call__(self, measurement):
        error = self.setpoint - measurement
        self.integral += error * self.dt
        self.integral = max(-10, min(10, self.integral))
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class DroneSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        # Devices
        self.imu = self.getDevice("inertial unit")
        self.gps = self.getDevice("gps")
        self.gyro = self.getDevice("gyro")
        self.compass = self.getDevice("compass")
        self.imu.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.compass.enable(self.timestep)

        # Motors
        motor_names = ["front left propeller", "front right propeller",
                       "rear left propeller", "rear right propeller"]
        self.motors = []
        for name in motor_names:
            m = self.getDevice(name)
            if m:
                m.setPosition(float('inf'))
                m.setVelocity(0)
                self.motors.append(m)
        print(f"✅ {len(self.motors)} propeller ditemukan")

        # Constants
        self.k_vertical_thrust = 68.5
        self.k_vertical_offset = 0.6
        self.k_vertical_p = 3.0
        self.k_roll_p = 50.0
        self.k_pitch_p = 30.0
        self.target_altitude = 2.0

        # Test sequence
        self.test_phase = 0          # 0=takeoff, 1=yaw kanan, 2=maju, 3=mundur, 4=yaw kiri, 5=maju, 6=mundur, 7=landing, 8=selesai
        self.phase_start_time = 0
        self.phase_duration = 3.0
        self.yaw_target = 0.0
        self.move_speed = 0.8
        self.yaw_angle = math.radians(90)

        self.yaw_pid = PID(1.5, 0.05, 0.1)
        self.yaw_pid.setpoint = 0.0
        self.last_print_time = 0

        # Landing parameters
        self.landing_start_altitude = 0
        self.landing_speed = 0.5  # m/s descent rate

    def run(self):
        print("Memulai test sequence...")
        while self.step(self.timestep) != -1:
            t = self.getTime()
            pos = self.gps.getValues()
            altitude = pos[2]
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            roll_vel = self.gyro.getValues()[0]
            pitch_vel = self.gyro.getValues()[1]

            # Altitude control (cubic)
            diff = self.target_altitude - altitude + self.k_vertical_offset
            clamped_diff = max(-1.0, min(1.0, diff))
            vertical_input = self.k_vertical_p * (clamped_diff ** 3)

            roll_dist = 0.0
            pitch_dist = 0.0
            yaw_input = 0.0

            # State machine
            if self.test_phase == 0:          # Take-off
                if altitude >= 1.95:
                    self.test_phase = 1
                    self.phase_start_time = t
                    print("Take-off selesai -> Yaw kanan")
            elif self.test_phase == 1:        # Yaw kanan
                if not hasattr(self, 'yaw_initial'):
                    self.yaw_initial = yaw
                    self.yaw_target = yaw + self.yaw_angle
                    print(f"Target yaw: {math.degrees(self.yaw_target):.1f}°")
                self.yaw_pid.setpoint = self.yaw_target
                yaw_input = self.yaw_pid(yaw)
                yaw_input = max(-1.0, min(1.0, yaw_input))
                if t - self.phase_start_time > self.phase_duration or abs(yaw - self.yaw_target) < 0.05:
                    print("Yaw kanan selesai -> Maju")
                    self.test_phase = 2
                    self.phase_start_time = t
                    delattr(self, 'yaw_initial')
                    self.yaw_pid.reset()
            elif self.test_phase == 2:        # Maju
                pitch_dist = -self.move_speed
                if t - self.phase_start_time > self.phase_duration:
                    print("Maju selesai -> Mundur")
                    self.test_phase = 3
                    self.phase_start_time = t
            elif self.test_phase == 3:        # Mundur
                pitch_dist = self.move_speed
                if t - self.phase_start_time > self.phase_duration:
                    print("Mundur selesai -> Yaw kiri")
                    self.test_phase = 4
                    self.phase_start_time = t
            elif self.test_phase == 4:        # Yaw kiri
                if not hasattr(self, 'yaw_initial'):
                    self.yaw_initial = yaw
                    self.yaw_target = yaw - self.yaw_angle
                    print(f"Target yaw: {math.degrees(self.yaw_target):.1f}°")
                self.yaw_pid.setpoint = self.yaw_target
                yaw_input = self.yaw_pid(yaw)
                yaw_input = max(-1.0, min(1.0, yaw_input))
                if t - self.phase_start_time > self.phase_duration or abs(yaw - self.yaw_target) < 0.05:
                    print("Yaw kiri selesai -> Maju lagi")
                    self.test_phase = 5
                    self.phase_start_time = t
                    delattr(self, 'yaw_initial')
                    self.yaw_pid.reset()
            elif self.test_phase == 5:        # Maju lagi
                pitch_dist = -self.move_speed
                if t - self.phase_start_time > self.phase_duration:
                    print("Maju lagi selesai -> Mundur lagi")
                    self.test_phase = 6
                    self.phase_start_time = t
            elif self.test_phase == 6:        # Mundur lagi
                pitch_dist = self.move_speed
                if t - self.phase_start_time > self.phase_duration:
                    print("Mundur lagi selesai -> Landing")
                    self.test_phase = 7
                    self.phase_start_time = t
                    self.landing_start_altitude = altitude
                    # Set target altitude ke 0 untuk landing
                    self.target_altitude = 0.0
            elif self.test_phase == 7:        # Landing
                # Selama landing, tidak ada disturbance
                roll_dist = 0.0
                pitch_dist = 0.0
                yaw_input = 0.0
                # Cek apakah sudah menyentuh tanah
                if altitude <= 0.1:
                    print("Drone telah mendarat, matikan motor")
                    self.test_phase = 8
                    # Hentikan motor segera
                    for m in self.motors:
                        m.setVelocity(0)
                    break
            elif self.test_phase == 8:
                break

            # Print status setiap 1 detik
            if t - self.last_print_time >= 1.0:
                self.last_print_time = t
                print(f"t={t:.1f} | Phase={self.test_phase} | Alt={altitude:.2f}m | Yaw={math.degrees(yaw):.1f}° | Pitch_dist={pitch_dist:.2f}")

            # Stabilization
            clamped_roll = max(-1.0, min(1.0, roll))
            clamped_pitch = max(-1.0, min(1.0, pitch))
            roll_input = self.k_roll_p * clamped_roll + roll_vel + roll_dist
            pitch_input = self.k_pitch_p * clamped_pitch + pitch_vel + pitch_dist

            # Motor mixing dengan tanda yaw dibalik
            front_left  = self.k_vertical_thrust + vertical_input - roll_input + pitch_input + yaw_input
            front_right = self.k_vertical_thrust + vertical_input + roll_input + pitch_input - yaw_input
            rear_left   = self.k_vertical_thrust + vertical_input - roll_input - pitch_input - yaw_input
            rear_right  = self.k_vertical_thrust + vertical_input + roll_input - pitch_input + yaw_input

            self.motors[0].setVelocity(front_left)
            self.motors[1].setVelocity(-front_right)
            self.motors[2].setVelocity(-rear_left)
            self.motors[3].setVelocity(rear_right)

        # Pastikan motor mati
        for m in self.motors:
            m.setVelocity(0)
        print("Test sequence selesai.")

if __name__ == "__main__":
    supervisor = DroneSupervisor()
    supervisor.run()