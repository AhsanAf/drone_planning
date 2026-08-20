from controller import Supervisor
import socket
import json
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

        # Socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', 65432))
        self.server.listen(1)
        self.server.setblocking(False)
        self.conn = None
        print("✅ Supervisor siap. Menunggu GUI...")

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

        # Constants (Semua Settingan YAW/PITCH TETAP SAMA PERSIS SEPERTI ASLI)
        self.k_vertical_thrust = 68.5
        self.k_vertical_offset = 0.6
        self.k_vertical_p = 3.0
        self.k_roll_p = 50.0
        self.k_pitch_p = 30.0
        self.target_altitude = 2.0

        # Waypoint navigation
        self.waypoints = []
        self.current_wp = 0
        self.flying = False
        self.takeoff_done = False
        self.k_pos = 0.8
        self.wp_threshold = 0.5
        self.decel_distance = 1.2
        self.debug_counter = 0

        # Yaw PID
        self.yaw_pid = PID(0.8, 0.01, 0.1)
        self.yaw_pid.setpoint = 0.0

    # ============================================================
    #  GET MAP DATA (3D)
    # ============================================================
    def get_map_data(self):
        drone_pos = self.getSelf().getPosition()
        start = [round(drone_pos[0], 3), round(drone_pos[1], 3), round(drone_pos[2], 3)]

        goal_node = self.getFromDef("TARGET")
        if goal_node:
            g = goal_node.getPosition()
            goal = [round(g[0], 3), round(g[1], 3), round(g[2], 3)]
        else:
            goal = [6.59, 6.40, 2.0]

        obstacles = []
        children = self.getRoot().getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            if node.getDef() and "OBSTACLE" in node.getDef():
                p = node.getPosition()
                s = node.getField("size").getSFVec3f()
                rot_field = node.getField("rotation")
                if rot_field:
                    rot = rot_field.getSFRotation()
                    if abs(rot[2]) > 0.9:
                        angle = rot[3]
                    else:
                        angle = 0.0
                else:
                    angle = 0.0
                obstacles.append({
                    "x": round(p[0], 3),
                    "y": round(p[1], 3),
                    "z": round(p[2], 3),
                    "width": round(s[0], 2),
                    "depth": round(s[1], 2),
                    "height": round(s[2], 2),
                    "rot": round(angle, 4)
                })
        return {"start": start, "goal": goal, "obstacles": obstacles}

    # ============================================================
    #  FOLLOW WAYPOINTS (3D)
    # ============================================================
    def follow_waypoints(self, path):
        if not path or len(path) < 2:
            print("❌ Path kosong atau terlalu pendek")
            return

        self.waypoints = [[p[0], p[1], p[2]] for p in path]
        self.current_wp = 0
        self.flying = True
        self.takeoff_done = False
        first_z = self.waypoints[0][2]
        self.target_altitude = max(2.0, first_z)

        print(f"🚀 Menerima {len(self.waypoints)} waypoint (3D)")
        for i, wp in enumerate(self.waypoints[:5]):
            print(f"   WP{i}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
        print(f"   Target altitude awal: {self.target_altitude:.2f}")

    # ============================================================
    #  MAIN LOOP
    # ============================================================
    def run(self):
        while self.step(self.timestep) != -1:
            # Socket handling
            if not self.conn:
                try:
                    self.conn, addr = self.server.accept()
                    self.conn.setblocking(False)
                    print(f"✅ GUI terhubung dari {addr}")
                except BlockingIOError:
                    pass
                except:
                    pass
            else:
                try:
                    data = self.conn.recv(65536).decode()
                    if data:
                        msg = json.loads(data)
                        cmd = msg.get("command")
                        if cmd == "GET_MAP":
                            self.conn.sendall(json.dumps(self.get_map_data()).encode())
                            print("📤 Map data dikirim ke GUI")
                        elif cmd == "START_SIM":
                            self.follow_waypoints(msg.get("path", []))
                        elif cmd == "RESET":
                            self.getSelf().getField("translation").setSFVec3f([-6.685, -6.228, 0.065])
                            self.waypoints = []
                            self.current_wp = 0
                            self.flying = False
                            self.takeoff_done = False
                            for m in self.motors:
                                m.setVelocity(0)
                            print("🔄 Drone di-reset")
                except BlockingIOError:
                    pass
                except Exception as e:
                    print(f"Socket error: {e}")
                    try:
                        self.conn.close()
                    except:
                        pass
                    self.conn = None

            # Kontrol drone
            if self.flying and len(self.motors) == 4:
                pos = self.gps.getValues()
                altitude = pos[2]
                pos_x, pos_y = pos[0], pos[1]
                roll, pitch, yaw = self.imu.getRollPitchYaw()
                roll_vel = self.gyro.getValues()[0]
                pitch_vel = self.gyro.getValues()[1]

                # Altitude control (Motor Thrust)
                diff = self.target_altitude - altitude + self.k_vertical_offset
                clamped_diff = max(-1.0, min(1.0, diff))
                vertical_input = self.k_vertical_p * (clamped_diff ** 3)

                roll_dist = 0.0
                pitch_dist = 0.0
                yaw_input = 0.0

                if self.takeoff_done and self.waypoints and self.current_wp < len(self.waypoints):
                    target = self.waypoints[self.current_wp]
                    self.target_altitude = target[2]

                    error_x = target[0] - pos_x
                    error_y = target[1] - pos_y
                    dist = math.hypot(error_x, error_y)

                    if dist < self.decel_distance:
                        gain = self.k_pos * (dist / self.decel_distance)
                    else:
                        gain = self.k_pos
                    gain = max(0.2, gain)

                    roll_dist = -gain * error_y
                    pitch_dist = gain * error_x

                    max_dist = 1.0
                    roll_dist = max(-max_dist, min(max_dist, roll_dist))
                    pitch_dist = max(-max_dist, min(max_dist, pitch_dist))

                    # ======================================================================
                    # PERBAIKAN PENTING! (TIDAK MENGUBAH YAW/PITCH, HANYA MENAMBAH PENGAWAS)
                    # 1. Saat ketinggian masih 0-0.5m, prioritas naik (gerak horizontal dibatasi)
                    # 2. Menambahkan kompensasi tenaga (anti-stall) agar tidak jatuh saat miring
                    # ======================================================================
                    if altitude < 0.5:
                        roll_dist *= 0.2  # Hanya boleh miring 20%
                        pitch_dist *= 0.2
                        vertical_input += 2.0  # Tambahan thrust agar cepat naik
                    elif altitude < 1.0:
                        roll_dist *= 0.5  # Hanya boleh miring 50%
                        pitch_dist *= 0.5
                        vertical_input += 1.0

                    # Kompensasi gaya angkat yang hilang akibat kemiringan drone
                    vertical_input += (abs(roll_dist) * 0.8 + abs(pitch_dist) * 0.8)
                    # ======================================================================

                    self.debug_counter += 1
                    if self.debug_counter >= 100:
                        self.debug_counter = 0
                        print(f"Pos: ({pos_x:.2f}, {pos_y:.2f}, {altitude:.2f}) "
                              f"Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) "
                              f"Dist: {dist:.2f}")

                    if dist < self.wp_threshold:
                        self.current_wp += 1
                        print(f"✅ Waypoint {self.current_wp} dicapai")
                        if self.current_wp >= len(self.waypoints):
                            print("🏁 Semua waypoint selesai, landing...")
                            self.target_altitude = 0.0

                # Yaw control (Persis seperti asli)
                self.yaw_pid.setpoint = 0.0
                yaw_input = self.yaw_pid(yaw)
                yaw_input = max(-0.5, min(0.5, yaw_input))

                # Stabilization (Persis seperti asli)
                clamped_roll = max(-1.0, min(1.0, roll))
                clamped_pitch = max(-1.0, min(1.0, pitch))
                roll_input = self.k_roll_p * clamped_roll + roll_vel + roll_dist
                pitch_input = self.k_pitch_p * clamped_pitch + pitch_vel + pitch_dist

                # Motor mixing (Persis seperti asli)
                front_left  = self.k_vertical_thrust + vertical_input - roll_input + pitch_input + yaw_input
                front_right = self.k_vertical_thrust + vertical_input + roll_input + pitch_input - yaw_input
                rear_left   = self.k_vertical_thrust + vertical_input - roll_input - pitch_input - yaw_input
                rear_right  = self.k_vertical_thrust + vertical_input + roll_input - pitch_input + yaw_input

                self.motors[0].setVelocity(front_left)
                self.motors[1].setVelocity(-front_right)
                self.motors[2].setVelocity(-rear_left)
                self.motors[3].setVelocity(rear_right)

                if not self.takeoff_done and altitude >= 1.95:
                    self.takeoff_done = True
                    print("✅ Take-off selesai, mulai navigasi")

                if self.target_altitude == 0.0 and altitude <= 0.1:
                    print("🛬 Drone mendarat, matikan motor")
                    for m in self.motors:
                        m.setVelocity(0)
                    self.flying = False
                    break

if __name__ == "__main__":
    supervisor = DroneSupervisor()
    supervisor.run()