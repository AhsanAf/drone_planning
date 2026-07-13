import tkinter as tk
from tkinter import messagebox, ttk
import socket, json, threading, math, random, time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# ============================================================
#  HELPER 3D (dengan margin radius drone)
# ============================================================

def is_point_in_box(px, py, pz, obs, margin=0.0):
    tx, ty, tz = px - obs['x'], py - obs['y'], pz - obs['z']
    c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
    lx, ly = tx * c - ty * s, tx * s + ty * c
    lz = tz
    return (abs(lx) <= obs['width']/2 + margin and
            abs(ly) <= obs['depth']/2 + margin and
            abs(lz) <= obs['height']/2 + margin)

def get_box_corners(obs):
    c, s = math.cos(obs['rot']), math.sin(obs['rot'])
    hw, hd, hh = obs['width']/2, obs['depth']/2, obs['height']/2
    pts = []
    for x in [-hw, hw]:
        for y in [-hd, hd]:
            for z in [-hh, hh]:
                rx = obs['x'] + (x*c - y*s)
                ry = obs['y'] + (x*s + y*c)
                pts.append([rx, ry, obs['z'] + z])
    return pts

def calculate_path_cost_3d(path):
    if not path or len(path) < 2:
        return 0
    return sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))

# ============================================================
#  PLANNING ENGINE 3D (RRT + RRT*)
# ============================================================

class PlanningEngine3D:
    def __init__(self, start, goal, obstacles, drone_radius=0.8, bounds=(-10, 10)):
        if len(start) == 2:
            start = [start[0], start[1], 0.0]
        if len(goal) == 2:
            goal = [goal[0], goal[1], 0.0]
        self.start = {"x": start[0], "y": start[1], "z": start[2], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1], "z": goal[2]}
        self.obstacles = obstacles
        self.bounds = bounds
        self.drone_radius = drone_radius
        self.node_list = []

        self.expand_dis = 0.5
        self.search_radius = 2.0
        self.max_gaussian_attempts = 50
        self.goal_bias = 0.1
        self.gaussian_bias = 0.5
        self.uniform_bias = 0.4

    def check_collision(self, x, y, z):
        b = self.bounds
        if (x < b[0] + self.drone_radius or x > b[1] - self.drone_radius or
            y < b[0] + self.drone_radius or y > b[1] - self.drone_radius or
            z < b[0] + self.drone_radius or z > b[1] - self.drone_radius):
            return True
        for o in self.obstacles:
            if is_point_in_box(x, y, z, o, margin=self.drone_radius):
                return True
        return False

    def is_line_safe(self, p1, p2):
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / 0.1))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            z = p1[2] + t * (p2[2] - p1[2])
            if self.check_collision(x, y, z):
                return False
        return True

    def get_gaussian_sample(self):
        b = self.bounds
        for _ in range(self.max_gaussian_attempts):
            x1 = random.uniform(b[0] + self.drone_radius, b[1] - self.drone_radius)
            y1 = random.uniform(b[0] + self.drone_radius, b[1] - self.drone_radius)
            z1 = random.uniform(b[0] + self.drone_radius, b[1] - self.drone_radius)
            if self.check_collision(x1, y1, z1):
                continue
            sigma = random.uniform(0.5, 2.0)
            x2 = max(b[0]+self.drone_radius, min(b[1]-self.drone_radius, random.gauss(x1, sigma)))
            y2 = max(b[0]+self.drone_radius, min(b[1]-self.drone_radius, random.gauss(y1, sigma)))
            z2 = max(b[0]+self.drone_radius, min(b[1]-self.drone_radius, random.gauss(z1, sigma)))
            if self.check_collision(x2, y2, z2):
                mx, my, mz = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
                if not self.check_collision(mx, my, mz):
                    return [mx, my, mz]
        return [random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius),
                random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius),
                random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius)]

    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"], node["z"]], [self.goal["x"], self.goal["y"], self.goal["z"]])
        if dist <= self.expand_dis * 1.5:
            if self.is_line_safe([node["x"], node["y"], node["z"]], [self.goal["x"], self.goal["y"], self.goal["z"]]):
                return True
        return False

    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]
        b = self.bounds
        for _ in range(max_iter):
            p = random.random()
            if p < self.goal_bias:
                rnd = [self.goal["x"], self.goal["y"], self.goal["z"]]
            elif p < self.goal_bias + self.gaussian_bias:
                rnd = self.get_gaussian_sample()
            else:
                rnd = [random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius),
                       random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius),
                       random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius)]

            nearest = min(self.node_list, key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2 + (n["z"]-rnd[2])**2)
            dist = math.dist([nearest["x"], nearest["y"], nearest["z"]], rnd)
            theta = math.atan2(rnd[1] - nearest["y"], rnd[0] - nearest["x"])
            step = min(self.expand_dis, dist)
            t_step = step / dist if dist > 0 else 0
            new_node = {
                "x": nearest["x"] + step * math.cos(theta),
                "y": nearest["y"] + step * math.sin(theta),
                "z": nearest["z"] + t_step * (rnd[2] - nearest["z"]),
                "parent": nearest,
                "cost": nearest["cost"] + step
            }
            if self.check_collision(new_node["x"], new_node["y"], new_node["z"]):
                continue
            if not self.is_line_safe([nearest["x"], nearest["y"], nearest["z"]], [new_node["x"], new_node["y"], new_node["z"]]):
                continue
            self.node_list.append(new_node)
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)
        return None

    def solve_rrt_star(self, max_iter=500):
        self.node_list = [self.start]
        b = self.bounds
        for _ in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"], self.goal["z"]]
            else:
                rnd = [random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius),
                       random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius),
                       random.uniform(b[0]+self.drone_radius, b[1]-self.drone_radius)]

            nearest = min(self.node_list, key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2 + (n["z"]-rnd[2])**2)
            dist = math.dist([nearest["x"], nearest["y"], nearest["z"]], rnd)
            theta = math.atan2(rnd[1] - nearest["y"], rnd[0] - nearest["x"])
            step = min(self.expand_dis, dist)
            t_step = step / dist if dist > 0 else 0
            new_node = {
                "x": nearest["x"] + step * math.cos(theta),
                "y": nearest["y"] + step * math.sin(theta),
                "z": nearest["z"] + t_step * (rnd[2] - nearest["z"]),
                "parent": nearest,
                "cost": nearest["cost"] + step
            }
            if self.check_collision(new_node["x"], new_node["y"], new_node["z"]):
                continue
            if not self.is_line_safe([nearest["x"], nearest["y"], nearest["z"]], [new_node["x"], new_node["y"], new_node["z"]]):
                continue

            near_nodes = [n for n in self.node_list if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 + (n["z"]-new_node["z"])**2 <= self.search_radius**2]
            for near in near_nodes:
                d = math.dist([near["x"], near["y"], near["z"]], [new_node["x"], new_node["y"], new_node["z"]])
                if near["cost"] + d < new_node["cost"]:
                    if self.is_line_safe([near["x"], near["y"], near["z"]], [new_node["x"], new_node["y"], new_node["z"]]):
                        new_node["cost"] = near["cost"] + d
                        new_node["parent"] = near

            self.node_list.append(new_node)
            for near in near_nodes:
                if near == new_node["parent"]:
                    continue
                d = math.dist([new_node["x"], new_node["y"], new_node["z"]], [near["x"], near["y"], near["z"]])
                if new_node["cost"] + d < near["cost"]:
                    if self.is_line_safe([new_node["x"], new_node["y"], new_node["z"]], [near["x"], near["y"], near["z"]]):
                        near["parent"] = new_node
                        near["cost"] = new_node["cost"] + d

        reachable = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable:
            best = min(reachable, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"], self.goal["z"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"], curr["z"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=30):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        curr = 0
        while curr < len(path) - 1:
            found = False
            for test in range(len(path)-1, curr, -1):
                if self.is_line_safe(path[curr], path[test]):
                    smoothed.append(path[test])
                    curr = test
                    found = True
                    break
            if not found:
                curr += 1
                if curr < len(path):
                    smoothed.append(path[curr])
        return smoothed


# ============================================================
#  GUI 3D
# ============================================================

class DroneApp3D:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Bias RRT vs RRT* 3D")

        self.sock = None
        self.data = {"start": [0,0,0], "goal": [5,5,3], "obs": [], "path": []}
        self.running = False

        side = tk.Frame(root, width=300, bg="#2c3e50")
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)

        tk.Label(side, text="ALGORITHM", bg="#2c3e50", fg="white", font=("Arial", 11, "bold")).pack(pady=15)
        self.algo_var = tk.StringVar(value="multibias")
        tk.Radiobutton(side, text="RRT*", variable=self.algo_var, value="rrtstar", bg="#2c3e50", fg="white", selectcolor="#34495e").pack(anchor="w", padx=20, pady=2)
        tk.Radiobutton(side, text="Multi-Bias RRT", variable=self.algo_var, value="multibias", bg="#2c3e50", fg="white", selectcolor="#34495e").pack(anchor="w", padx=20, pady=2)

        iter_frame = tk.Frame(side, bg="#2c3e50")
        iter_frame.pack(fill='x', padx=20, pady=15)
        tk.Label(iter_frame, text="MAX ITERATIONS", bg="#2c3e50", fg="white", font=("Arial", 9, "bold")).pack(anchor="w")
        self.iter_var = tk.StringVar(value="500")
        tk.Entry(iter_frame, textvariable=self.iter_var, width=15).pack(pady=5)

        radius_frame = tk.Frame(side, bg="#2c3e50")
        radius_frame.pack(fill='x', padx=20, pady=10)
        tk.Label(radius_frame, text="DRONE RADIUS (m)", bg="#2c3e50", fg="white", font=("Arial", 9, "bold")).pack(anchor="w")
        self.radius_var = tk.StringVar(value="1.2")  # Dinaikkan dari 0.8 ke 1.2 agar path lebih aman dari tembok
        tk.Entry(radius_frame, textvariable=self.radius_var, width=10).pack(pady=2)
        tk.Label(radius_frame, text="(clearance from obstacles)", bg="#2c3e50", fg="#bdc3c7", font=("Arial", 7)).pack()

        self.btn_map = ttk.Button(side, text="1. LOAD MAP", command=self.get_map)
        self.btn_map.pack(fill='x', padx=20, pady=5)
        self.btn_run = ttk.Button(side, text="2. RUN PLANNING", command=self.start_thread, state="disabled")
        self.btn_run.pack(fill='x', padx=20, pady=5)
        self.btn_fly = ttk.Button(side, text="3. EXECUTE (WEBOTS)", command=self.fly, state="disabled")
        self.btn_fly.pack(fill='x', padx=20, pady=5)
        self.btn_reset = tk.Button(side, text="⚠ RESET DRONE", command=self.reset_sim, bg="#e74c3c", fg="white")
        self.btn_reset.pack(fill='x', padx=20, pady=15)

        self.lbl_metrics = tk.Label(side, text="Status: Ready", bg="#2c3e50", fg="#bdc3c7", justify=tk.LEFT, font=("Consolas", 9))
        self.lbl_metrics.pack(pady=10, padx=10, anchor="w")

        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.connect_socket()

    def connect_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2)
            self.sock.connect(('127.0.0.1', 65432))
            self.lbl_metrics.config(text="Status: Connected to Webots")
        except:
            self.sock = None
            self.root.after(2000, self.connect_socket)

    def normalize_point(self, point, name="point"):
        if not isinstance(point, list):
            raise TypeError(f"{name} harus berupa list")
        if len(point) == 2:
            return [float(point[0]), float(point[1]), 0.0]
        elif len(point) == 3:
            return [float(point[0]), float(point[1]), float(point[2])]
        else:
            raise ValueError(f"{name} harus memiliki 2 atau 3 elemen, got {len(point)}")

    def normalize_obstacle(self, obs):
        return {
            "x": float(obs.get("x", 0)),
            "y": float(obs.get("y", 0)),
            "z": float(obs.get("z", 0)),
            "width": float(obs.get("width", 1)),
            "depth": float(obs.get("depth", 1)),
            "height": float(obs.get("height", 1)),
            "rot": float(obs.get("rot", 0))
        }

    def get_map(self):
        if not self.sock:
            messagebox.showwarning("Koneksi", "Tidak terhubung ke Webots")
            return
        try:
            self.sock.sendall(json.dumps({"command": "GET_MAP"}).encode())
            raw = self.sock.recv(65536)
            if not raw:
                raise Exception("Tidak ada data dari Webots")
            data = json.loads(raw.decode())
            print("[DEBUG] Data dari Webots:", data)

            if 'start' in data:
                self.data['start'] = self.normalize_point(data['start'], "start")
            if 'goal' in data:
                self.data['goal'] = self.normalize_point(data['goal'], "goal")
                print(f"[INFO] Goal position diterima: {self.data['goal']}")

            if 'obstacles' in data and isinstance(data['obstacles'], list):
                self.data['obs'] = [self.normalize_obstacle(o) for o in data['obstacles']]
            else:
                self.data['obs'] = []

            self.draw_world()
            self.btn_run.config(state="normal")
            self.lbl_metrics.config(text="Status: Map loaded (3D)")
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Gagal parsing JSON:\n{e}\nData mentah: {raw[:200]}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal load map:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def draw_world(self, tree_edges=[]):
        self.ax.clear()
        if self.data.get('obs'):
            for o in self.data['obs']:
                try:
                    corners = get_box_corners(o)
                    faces = [
                        [corners[0], corners[1], corners[3], corners[2]],
                        [corners[4], corners[5], corners[7], corners[6]],
                        [corners[0], corners[1], corners[5], corners[4]],
                        [corners[2], corners[3], corners[7], corners[6]],
                        [corners[0], corners[2], corners[6], corners[4]],
                        [corners[1], corners[3], corners[7], corners[5]]
                    ]
                    self.ax.add_collection3d(Poly3DCollection(faces, color='#34495e', alpha=0.7))
                except Exception as e:
                    print(f"Error drawing obstacle: {e}")
        
        if tree_edges:
            for e in tree_edges:
                self.ax.plot3D(*zip(*e), color='#bdc3c7', linewidth=0.5, alpha=0.5)
        
        start = self.data.get('start')
        if start and len(start)==3:
            self.ax.scatter(*start, color='green', s=80, label='Start')
        goal = self.data.get('goal')
        if goal and len(goal)==3:
            self.ax.scatter(*goal, color='red', s=100, marker='*', label='Goal')
        
        path = self.data.get('path')
        if path and len(path) > 1:
            p_np = np.array(path)
            c = 'cyan' if self.algo_var.get() == "multibias" else 'magenta'
            lbl = 'Multi-Bias Path' if self.algo_var.get() == "multibias" else 'RRT* Path'
            self.ax.plot3D(p_np[:,0], p_np[:,1], p_np[:,2], color=c, linewidth=3, label=lbl)
        
        bounds = (-10, 10)
        self.ax.set_xlim(bounds)
        self.ax.set_ylim(bounds)
        self.ax.set_zlim(bounds)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"3D Environment ({len(self.data.get('obs', []))} Obstacles)")
        self.ax.legend(loc='upper right')
        self.canvas.draw_idle()

    def start_thread(self):
        if self.running:
            return
        self.running = True
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.solve, args=(int(self.iter_var.get()),), daemon=True).start()

    def solve(self, max_iter):
        t0 = time.time()
        try:
            drone_radius = float(self.radius_var.get())
            if drone_radius < 0.1:
                drone_radius = 0.1
        except:
            drone_radius = 1.2
        print(f"[INFO] Menggunakan drone radius = {drone_radius} m")

        eng = PlanningEngine3D(self.data['start'], self.data['goal'], self.data['obs'], drone_radius=drone_radius)

        if self.algo_var.get() == "rrtstar":
            algo_name = "RRT*"
            raw_path = eng.solve_rrt_star(max_iter)
        else:
            algo_name = "Multi-Bias"
            raw_path = eng.solve_multibias(max_iter)

        if raw_path:
            path = eng.smooth_path(raw_path)
        else:
            path = None

        dt = time.time() - t0
        tree_edges = []
        if len(eng.node_list) < 2000:
            for n in eng.node_list:
                if n["parent"]:
                    tree_edges.append([(n["x"], n["y"], n["z"]), (n["parent"]["x"], n["parent"]["y"], n["parent"]["z"])])

        self.root.after(0, self.planning_done, path, tree_edges, dt, len(eng.node_list), algo_name, path is not None)

    def planning_done(self, path, tree_edges, dt, nodes, algo_name, success):
        self.running = False
        self.btn_run.config(state="normal")

        if not success:
            self.lbl_metrics.config(text=f"Status: {algo_name} FAILED\nTime: {dt:.3f}s\nNodes: {nodes}")
            messagebox.showwarning("Fail", "Path not found!")
            self.draw_world(tree_edges)
            return

        self.data['path'] = path
        self.draw_world(tree_edges)
        cost = calculate_path_cost_3d(path)
        self.lbl_metrics.config(text=f"[{algo_name} RESULT]\nTime: {dt:.3f}s\nCost: {cost:.2f}m\nNodes: {nodes}\nStatus: SUCCESS")
        self.btn_fly.config(state="normal")

    def fly(self):
        if self.data['path'] and self.sock:
            try:
                self.sock.sendall(json.dumps({"command": "START_SIM", "path": self.data['path']}).encode())
            except Exception as e:
                messagebox.showerror("Socket Error", f"Gagal mengirim perintah terbang: {e}")

    def reset_sim(self):
        if self.sock:
            try:
                self.sock.sendall(json.dumps({"command": "RESET"}).encode())
            except:
                pass
        self.data['path'] = []
        self.draw_world()

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneApp3D(root)
    root.mainloop()