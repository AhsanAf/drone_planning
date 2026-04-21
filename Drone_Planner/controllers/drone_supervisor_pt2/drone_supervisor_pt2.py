from controller import Supervisor
import socket
import json
import math

# ====================== PID MANUAL ======================
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.008

    def __call__(self, measurement):
        error = self.setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# ====================== SUPERVISOR ======================
class DroneSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # SOCKET
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', 65432))
        self.server.listen(1)
        self.server.setblocking(False)
        self.conn = None
        print("✅ Supervisor siap. Menunggu GUI...")

        # DEVICES
        self.gps = self.getDevice("gps")
        self.imu = self.getDevice("inertial unit")
        self.gyro = self.getDevice("gyro")
        self.compass = self.getDevice("compass")
        self.gps.enable(self.timestep)
        self.imu.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.compass.enable(self.timestep)

        # MOTOR (nama resmi Mavic 2 Pro)
        motor_names = ["front left propeller", "front right propeller",
                       "rear left propeller", "rear right propeller"]
        self.motors = []
        for name in motor_names:
            m = self.getDevice(name)
            if m:
                m.setPosition(float('inf'))
                m.setVelocity(0)
                self.motors.append(m)
        print(f"✅ {len(self.motors)} propeller ditemukan!")

        # PID
        self.pid_roll  = PID(1.3, 0.1, 0.6)
        self.pid_pitch = PID(1.3, 0.1, 0.6)
        self.pid_yaw   = PID(0.9, 0.0, 0.5)
        self.pid_alt   = PID(4.5, 0.8, 2.0, setpoint=3.0)   # lebih kuat
        
        # Outer loop position control
        self.k_pos = 0.8   # gain posisi → sudut (rad)

        self.waypoints = []
        self.current_wp = 0
        self.flying = False
        self.takeoff_done = False

    def get_map_data(self):
        drone_pos = self.getSelf().getPosition()
        start = [round(drone_pos[0], 3), round(drone_pos[1], 3)]
        
        goal_node = self.getFromDef("TARGET")
        goal = [round(goal_node.getPosition()[0], 3), round(goal_node.getPosition()[1], 3)] if goal_node else [6.59, 6.40]
        
        obstacles = []
        children = self.getRoot().getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            if node.getDef() and "OBSTACLE" in node.getDef():
                p = node.getPosition()
                s = node.getField("size").getSFVec3f()
                obstacles.append({"x": round(p[0],3), "y": round(p[1],3), "w": round(s[0],2), "h": round(s[1],2), "rot": 1.5708})
        return {"start": start, "goal": goal, "obstacles": obstacles}

    def follow_waypoints(self, path):
        self.waypoints = [[p[0], p[1], 3.0] for p in path]  # target altitude 3m
        self.current_wp = 0
        self.flying = True
        self.takeoff_done = False
        print(f"🚀 Mulai terbang ke {len(self.waypoints)} waypoint")

    def run(self):
        while self.step(self.timestep) != -1:
            # SOCKET
            if not self.conn:
                try:
                    self.conn, addr = self.server.accept()
                    self.conn.setblocking(False)
                    print(f"✅ GUI terhubung dari {addr}")
                except: pass
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
                            self.flying = False
                            self.takeoff_done = False
                            for m in self.motors: m.setVelocity(0)
                            print("🔄 Drone di-reset")
                except: pass

            # === CONTROL ===
            if self.flying and self.waypoints and len(self.motors) == 4:
                pos = self.gps.getValues()
                roll, pitch, yaw = self.imu.getRollPitchYaw()
                target = self.waypoints[self.current_wp]

                # POSITION CONTROL (outer loop)
                error_x = target[0] - pos[0]
                error_y = target[1] - pos[1]
                desired_roll  = max(-0.5, min(0.5, self.k_pos * error_y))   # roll untuk gerak samping
                desired_pitch = max(-0.5, min(0.5, -self.k_pos * error_x))  # pitch untuk maju/mundur

                # Setpoint attitude
                self.pid_roll.setpoint = desired_roll
                self.pid_pitch.setpoint = desired_pitch
                self.pid_yaw.setpoint = 0

                roll_cmd  = self.pid_roll(roll)
                pitch_cmd = self.pid_pitch(pitch)
                yaw_cmd   = self.pid_yaw(yaw)

                # ALTITUDE
                alt_error = target[2] - pos[2]
                if not self.takeoff_done and pos[2] < 2.5:
                    thrust = 280.0   # boost takeoff
                    if pos[2] > 2.5:
                        self.takeoff_done = True
                else:
                    self.pid_alt.setpoint = target[2]
                    thrust = self.pid_alt(pos[2])
                    thrust = max(120, min(380, thrust))

                # MOTOR MIXING (X configuration)
                self.motors[0].setVelocity(thrust - roll_cmd + pitch_cmd + yaw_cmd)   # front left
                self.motors[1].setVelocity(thrust + roll_cmd + pitch_cmd - yaw_cmd)   # front right
                self.motors[2].setVelocity(thrust + roll_cmd - pitch_cmd + yaw_cmd)   # rear left
                self.motors[3].setVelocity(thrust - roll_cmd - pitch_cmd - yaw_cmd)   # rear right

                # Cek sampai waypoint
                dist = math.hypot(error_x, error_y)
                if dist < 0.4 and abs(alt_error) < 0.5:
                    self.current_wp += 1
                    if self.current_wp >= len(self.waypoints):
                        self.flying = False
                        print("🏁 Sampai tujuan!")
                        for m in self.motors: m.setVelocity(0)

if __name__ == "__main__":
    supervisor = DroneSupervisor()
    supervisor.run()