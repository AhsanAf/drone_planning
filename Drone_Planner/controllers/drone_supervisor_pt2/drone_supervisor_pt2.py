from controller import Supervisor, GPS, InertialUnit, Gyro, Compass
import socket
import json
import time
import math
from simple_pid import PID   # pip install simple_pid (jalankan sekali di terminal)

class DroneSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # === SOCKET SERVER ===
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', 65432))
        self.server.listen(1)
        print("✅ Supervisor: Menunggu koneksi dari GUI...")
        self.conn, _ = self.server.accept()
        print("✅ GUI terhubung!")
        
        # === DEVICES ===
        self.gps = self.getDevice("gps")
        self.imu = self.getDevice("inertial unit")
        self.gyro = self.getDevice("gyro")
        self.compass = self.getDevice("compass")
        
        self.gps.enable(self.timestep)
        self.imu.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.compass.enable(self.timestep)
        
        # Motor 4 propeller
        self.motors = [self.getDevice(f"motor{i+1}") for i in range(4)]
        for m in self.motors:
            m.setPosition(float('inf'))
            m.setVelocity(0)
        
        # === PID (sudah ditune untuk Mavic2Pro + timestep 8 + damping 0.5) ===
        self.pid_roll  = PID(1.2, 0.08, 0.6, setpoint=0)
        self.pid_pitch = PID(1.2, 0.08, 0.6, setpoint=0)
        self.pid_yaw   = PID(0.8, 0.0, 0.4, setpoint=0)
        self.pid_alt   = PID(2.5, 0.4, 1.2, setpoint=2.0)  # target tinggi default 2m
        
        self.waypoints = []
        self.current_wp = 0
        self.flying = False
        self.target_alt = 2.0

    def get_map_data(self):
        """Kirim start, goal, obstacles ke GUI"""
        # Drone position (start)
        drone_pos = self.getSelf().getPosition()
        start = [round(drone_pos[0], 3), round(drone_pos[1], 3)]
        
        # Goal (Solid bernama GOAL_POINT)
        goal_node = self.getFromDef("GOAL_POINT")
        goal_pos = goal_node.getPosition()
        goal = [round(goal_pos[0], 3), round(goal_pos[1], 3)]
        
        # Obstacles (semua Wall dengan name OBSTACLE*)
        obstacles = []
        i = 1
        while True:
            node = self.getFromDef(f"OBSTACLE({i})") or self.getFromDef(f"OBSTACLE{i}")
            if not node:
                break
            pos = node.getPosition()
            rot = node.getRotation()  # ambil sudut yaw
            size = node.getField("size").getSFVec3f()
            obs = {
                "x": round(pos[0], 3),
                "y": round(pos[1], 3),
                "w": round(size[0], 2),
                "h": round(size[1], 2),
                "rot": round(math.atan2(rot[3], rot[0]), 4) if abs(rot[0]) > 0.1 else 1.5708
            }
            obstacles.append(obs)
            i += 1
        
        return {"start": start, "goal": goal, "obstacles": obstacles}

    def follow_waypoints(self, path):
        """Terima path dari GUI lalu terbang waypoint per waypoint"""
        self.waypoints = [[p[0], p[1], self.target_alt] for p in path]  # z = altitude tetap
        self.current_wp = 0
        self.flying = True
        print(f"🚀 Mulai terbang mengikuti {len(self.waypoints)} waypoint")

    def run(self):
        while self.step(self.timestep) != -1:
            # Terima command dari GUI
            try:
                self.conn.settimeout(0.01)
                data = self.conn.recv(65536)
                if data:
                    msg = json.loads(data.decode())
                    cmd = msg.get("command")
                    
                    if cmd == "GET_MAP":
                        map_data = self.get_map_data()
                        self.conn.sendall(json.dumps(map_data).encode())
                    
                    elif cmd == "RESET":
                        self.getSelf().getField("translation").setSFVec3f([-6.685, -6.228, 0.065])
                        self.waypoints = []
                        self.flying = False
                        print("🔄 Drone di-reset")
                    
                    elif cmd == "START_SIM":
                        self.follow_waypoints(msg.get("path", []))
            except:
                pass  # timeout normal
            
            # === LOW-LEVEL CONTROL (PID) ===
            if self.flying and self.waypoints:
                current_pos = self.gps.getValues()
                roll, pitch, yaw = self.imu.getRollPitchYaw()
                
                # Target waypoint saat ini
                target = self.waypoints[self.current_wp]
                error_x = target[0] - current_pos[0]
                error_y = target[1] - current_pos[1]
                
                # Hitung command (roll & pitch untuk gerak horizontal)
                self.pid_roll.setpoint = 0
                self.pid_pitch.setpoint = 0
                roll_cmd  = self.pid_roll(roll)
                pitch_cmd = self.pid_pitch(pitch)
                
                # Yaw selalu menghadap ke depan (bisa diubah)
                yaw_cmd = self.pid_yaw(yaw)
                
                # Altitude
                alt_cmd = self.pid_alt(current_pos[2])
                
                # Thrust + mixing motor (Mavic2Pro configuration)
                thrust = max(0, min(200, alt_cmd))  # 0-200 rad/s
                
                self.motors[0].setVelocity(thrust - roll_cmd + pitch_cmd + yaw_cmd)   # motor 1
                self.motors[1].setVelocity(thrust + roll_cmd + pitch_cmd - yaw_cmd)   # motor 2
                self.motors[2].setVelocity(thrust + roll_cmd - pitch_cmd + yaw_cmd)   # motor 3
                self.motors[3].setVelocity(thrust - roll_cmd - pitch_cmd - yaw_cmd)   # motor 4
                
                # Cek apakah sudah sampai waypoint (±0.3m)
                dist = math.hypot(error_x, error_y)
                if dist < 0.3:
                    self.current_wp += 1
                    if self.current_wp >= len(self.waypoints):
                        self.flying = False
                        print("🏁 Sampai tujuan!")
                        # Matikan motor
                        for m in self.motors:
                            m.setVelocity(0)

if __name__ == "__main__":
    supervisor = DroneSupervisor()
    supervisor.run()