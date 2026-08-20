import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import socket, json, threading, math, random, time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import openpyxl  # Untuk menyimpan hasil benchmark ke Excel

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
#  PLANNING ENGINE 3D (RRT + RRT*) - DIPERBAIKI
# ============================================================

class PlanningEngine3D:
    def __init__(self, start, goal, obstacles, drone_radius=0.6,
                 bounds_xy=(-10, 10), bounds_z=(0, 10),
                 goal_bias=0.1, uniform_bias=0.4, gaussian_bias=0.5):
        if len(start) == 2:
            start = [start[0], start[1], 0.0]
        if len(goal) == 2:
            goal = [goal[0], goal[1], 0.0]
        self.start = {"x": start[0], "y": start[1], "z": start[2], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1], "z": goal[2]}
        self.obstacles = obstacles
        self.bounds_xy = bounds_xy
        self.bounds_z = bounds_z
        self.drone_radius = drone_radius
        self.node_list = []

        self.expand_dis = max(0.8, drone_radius * 1.2)
        self.search_radius = max(2.0, drone_radius * 2.5)
        self.max_gaussian_attempts = 50
        self.goal_bias = goal_bias          # proporsi 0-1
        self.gaussian_bias = gaussian_bias  # proporsi 0-1
        self.uniform_bias = uniform_bias    # proporsi 0-1

    def check_collision(self, x, y, z):
        bx1, bx2 = self.bounds_xy
        bz1, bz2 = self.bounds_z
        if (x < bx1 + self.drone_radius or x > bx2 - self.drone_radius or
            y < bx1 + self.drone_radius or y > bx2 - self.drone_radius):
            return True
        if z < bz1:   # bawah = 0
            return True
        if z > bz2 - self.drone_radius:
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
        bx1, bx2 = self.bounds_xy
        bz1, bz2 = self.bounds_z
        for _ in range(self.max_gaussian_attempts):
            x1 = random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius)
            y1 = random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius)
            z1 = random.uniform(bz1, bz2 - self.drone_radius)
            if self.check_collision(x1, y1, z1):
                continue
            sigma = random.uniform(0.5, 2.0)
            x2 = max(bx1 + self.drone_radius, min(bx2 - self.drone_radius, random.gauss(x1, sigma)))
            y2 = max(bx1 + self.drone_radius, min(bx2 - self.drone_radius, random.gauss(y1, sigma)))
            z2 = max(bz1, min(bz2 - self.drone_radius, random.gauss(z1, sigma)))
            if self.check_collision(x2, y2, z2):
                mx, my, mz = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
                if not self.check_collision(mx, my, mz):
                    return [mx, my, mz]
        return [random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius),
                random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius),
                random.uniform(bz1, bz2 - self.drone_radius)]

    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"], node["z"]],
                         [self.goal["x"], self.goal["y"], self.goal["z"]])
        if dist <= self.expand_dis * 1.5:
            if self.is_line_safe([node["x"], node["y"], node["z"]],
                                 [self.goal["x"], self.goal["y"], self.goal["z"]]):
                return True
        return False

    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]
        bx1, bx2 = self.bounds_xy
        bz1, bz2 = self.bounds_z
        for _ in range(max_iter):
            p = random.random()
            if p < self.goal_bias:
                rnd = [self.goal["x"], self.goal["y"], self.goal["z"]]
            elif p < self.goal_bias + self.gaussian_bias:
                rnd = self.get_gaussian_sample()
            else:
                rnd = [random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius),
                       random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius),
                       random.uniform(bz1, bz2 - self.drone_radius)]

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
            if not self.is_line_safe([nearest["x"], nearest["y"], nearest["z"]],
                                     [new_node["x"], new_node["y"], new_node["z"]]):
                continue
            self.node_list.append(new_node)
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)
        return None

    def solve_rrt_star(self, max_iter=500):
        self.node_list = [self.start]
        bx1, bx2 = self.bounds_xy
        bz1, bz2 = self.bounds_z
        for _ in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"], self.goal["z"]]
            else:
                rnd = [random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius),
                       random.uniform(bx1 + self.drone_radius, bx2 - self.drone_radius),
                       random.uniform(bz1, bz2 - self.drone_radius)]

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
            if not self.is_line_safe([nearest["x"], nearest["y"], nearest["z"]],
                                     [new_node["x"], new_node["y"], new_node["z"]]):
                continue

            near_nodes = [n for n in self.node_list if (n["x"]-new_node["x"])**2 +
                          (n["y"]-new_node["y"])**2 + (n["z"]-new_node["z"])**2 <= self.search_radius**2]
            for near in near_nodes:
                d = math.dist([near["x"], near["y"], near["z"]],
                              [new_node["x"], new_node["y"], new_node["z"]])
                if near["cost"] + d < new_node["cost"]:
                    if self.is_line_safe([near["x"], near["y"], near["z"]],
                                         [new_node["x"], new_node["y"], new_node["z"]]):
                        new_node["cost"] = near["cost"] + d
                        new_node["parent"] = near

            self.node_list.append(new_node)
            for near in near_nodes:
                if near == new_node["parent"]:
                    continue
                d = math.dist([new_node["x"], new_node["y"], new_node["z"]],
                              [near["x"], near["y"], near["z"]])
                if new_node["cost"] + d < near["cost"]:
                    if self.is_line_safe([new_node["x"], new_node["y"], new_node["z"]],
                                         [near["x"], near["y"], near["z"]]):
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
        self.listening = False

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
        self.radius_var = tk.StringVar(value="0.6")
        tk.Entry(radius_frame, textvariable=self.radius_var, width=10).pack(pady=2)
        tk.Label(radius_frame, text="(clearance from obstacles)", bg="#2c3e50", fg="#bdc3c7", font=("Arial", 7)).pack()

        # === TAMBAHAN: Input untuk bias ===
        bias_frame = tk.Frame(side, bg="#2c3e50")
        bias_frame.pack(fill='x', padx=20, pady=10)
        tk.Label(bias_frame, text="BIAS PROBABILITY (%)", bg="#2c3e50", fg="white", font=("Arial", 9, "bold")).pack(anchor="w")

        tk.Label(bias_frame, text="Goal Bias", bg="#2c3e50", fg="#bdc3c7", font=("Arial", 8)).pack(anchor="w")
        self.goal_bias_var = tk.StringVar(value="10")
        tk.Entry(bias_frame, textvariable=self.goal_bias_var, width=10).pack(pady=2)

        tk.Label(bias_frame, text="Uniform Bias", bg="#2c3e50", fg="#bdc3c7", font=("Arial", 8)).pack(anchor="w")
        self.uniform_bias_var = tk.StringVar(value="40")
        tk.Entry(bias_frame, textvariable=self.uniform_bias_var, width=10).pack(pady=2)

        tk.Label(bias_frame, text="Gaussian Bias", bg="#2c3e50", fg="#bdc3c7", font=("Arial", 8)).pack(anchor="w")
        self.gaussian_bias_var = tk.StringVar(value="50")
        tk.Entry(bias_frame, textvariable=self.gaussian_bias_var, width=10).pack(pady=2)

        # === AKHIR TAMBAHAN BIAS ===

        self.btn_map = ttk.Button(side, text="1. LOAD MAP", command=self.get_map)
        self.btn_map.pack(fill='x', padx=20, pady=5)
        self.btn_run = ttk.Button(side, text="2. RUN PLANNING", command=self.start_thread, state="disabled")
        self.btn_run.pack(fill='x', padx=20, pady=5)
        self.btn_fly = ttk.Button(side, text="3. EXECUTE (WEBOTS)", command=self.fly, state="disabled")
        self.btn_fly.pack(fill='x', padx=20, pady=5)
        self.btn_reset = tk.Button(side, text="⚠ RESET DRONE", command=self.reset_sim, bg="#e74c3c", fg="white")
        self.btn_reset.pack(fill='x', padx=20, pady=15)

        # === TAMBAHAN: Tombol Benchmark ===
        self.btn_benchmark = ttk.Button(side, text="BENCHMARK", command=self.open_benchmark_dialog)
        self.btn_benchmark.pack(fill='x', padx=20, pady=5)

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
        x = float(obs.get("x", 0))
        y = float(obs.get("y", 0))
        z = float(obs.get("z", 0))
        rot = float(obs.get("rot", 0))

        width = 1.0; depth = 1.0; height = 1.0

        if "width" in obs and "depth" in obs and "height" in obs:
            width = float(obs["width"]); depth = float(obs["depth"]); height = float(obs["height"])
        elif "size" in obs and isinstance(obs["size"], list) and len(obs["size"]) == 3:
            width = float(obs["size"][0]); height = float(obs["size"][1]); depth = float(obs["size"][2])
        elif "dimensions" in obs and isinstance(obs["dimensions"], list) and len(obs["dimensions"]) == 3:
            width = float(obs["dimensions"][0]); height = float(obs["dimensions"][1]); depth = float(obs["dimensions"][2])
        else:
            if "w" in obs: width = float(obs["w"])
            if "d" in obs: depth = float(obs["d"])
            if "h" in obs: height = float(obs["h"])

        print(f"[DEBUG] Obstacle: x={x}, y={y}, z={z}, w={width}, d={depth}, h={height}, rot={rot}")
        return {"x": x, "y": y, "z": z, "width": width, "depth": depth, "height": height, "rot": rot}

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
            if 'obstacles' in data and isinstance(data['obstacles'], list):
                self.data['obs'] = [self.normalize_obstacle(o) for o in data['obstacles']]
            else:
                self.data['obs'] = []

            self.draw_world()
            self.btn_run.config(state="normal")
            self.lbl_metrics.config(text="Status: Map loaded (3D)")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal load map:\n{str(e)}")
            import traceback; traceback.print_exc()

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
                self.ax.plot3D(*zip(*e), color='orange', linewidth=1.5, alpha=0.7)
            nodes = set()
            for e in tree_edges:
                nodes.add(e[0]); nodes.add(e[1])
            if nodes:
                xs, ys, zs = zip(*nodes)
                self.ax.scatter(xs, ys, zs, c='gray', s=8, alpha=0.5)
        
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
        
        self.ax.set_xlim(-10, 10); self.ax.set_ylim(-10, 10); self.ax.set_zlim(0, 10)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title(f"3D Environment ({len(self.data.get('obs', []))} Obstacles)")
        self.ax.legend(loc='upper right')
        self.canvas.draw_idle()

    def validate_bias(self):
        """Mengambil nilai bias dari input, memvalidasi total 100%.
        Mengembalikan tuple (goal_bias, uniform_bias, gaussian_bias) dalam proporsi 0-1,
        atau None jika tidak valid.
        """
        try:
            goal = float(self.goal_bias_var.get())
            uniform = float(self.uniform_bias_var.get())
            gaussian = float(self.gaussian_bias_var.get())
        except ValueError:
            messagebox.showerror("Error", "Nilai bias harus berupa angka.")
            return None

        total = goal + uniform + gaussian
        if abs(total - 100.0) > 1e-6:
            messagebox.showerror("Error", f"Total bias harus 100%.\nSaat ini: {total:.1f}%")
            return None

        if goal < 0 or uniform < 0 or gaussian < 0:
            messagebox.showerror("Error", "Bias tidak boleh negatif.")
            return None

        # Konversi ke proporsi
        return (goal/100.0, uniform/100.0, gaussian/100.0)

    def start_thread(self):
        if self.running: return

        bias = self.validate_bias()
        if bias is None:
            return

        try:
            max_iter = int(self.iter_var.get())
            if max_iter <= 0:
                messagebox.showerror("Error", "Max iterations harus positif.")
                return
        except ValueError:
            messagebox.showerror("Error", "Max iterations harus angka.")
            return

        self.running = True
        self.btn_run.config(state="disabled")
        # Pass bias ke thread
        threading.Thread(target=self.solve, args=(max_iter, bias), daemon=True).start()

    def solve(self, max_iter, bias):
        t0 = time.time()
        try:
            drone_radius = float(self.radius_var.get())
            if drone_radius < 0.1: drone_radius = 0.1
        except:
            drone_radius = 0.6
        print(f"[INFO] Menggunakan drone radius = {drone_radius} m")

        goal_bias, uniform_bias, gaussian_bias = bias

        eng = PlanningEngine3D(self.data['start'], self.data['goal'], self.data['obs'],
                               drone_radius=drone_radius,
                               bounds_xy=(-10, 10), bounds_z=(0, 10),
                               goal_bias=goal_bias,
                               uniform_bias=uniform_bias,
                               gaussian_bias=gaussian_bias)

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
        if len(eng.node_list) < 5000:
            for n in eng.node_list:
                if n["parent"]:
                    tree_edges.append(((n["x"], n["y"], n["z"]), (n["parent"]["x"], n["parent"]["y"], n["parent"]["z"])))

        self.root.after(0, self.planning_done, path, tree_edges, dt, len(eng.node_list), algo_name, path is not None)

    def planning_done(self, path, tree_edges, dt, nodes, algo_name, success):
        self.running = False
        self.btn_run.config(state="normal")

        if not success:
            self.lbl_metrics.config(text=f"Status: {algo_name} FAILED\nTime: {dt:.3f}s\nNodes: {nodes}")
            messagebox.showwarning("Fail", "Path not found! Periksa start/goal dan radius.")
            self.draw_world(tree_edges)
            return

        self.data['path'] = path
        self.draw_world(tree_edges)
        cost = calculate_path_cost_3d(path)
        self.lbl_metrics.config(text=f"[{algo_name} RESULT]\nTime: {dt:.3f}s\nCost: {cost:.2f}m\nNodes: {nodes}\nStatus: SUCCESS")
        self.btn_fly.config(state="normal")

    def interpolate_path(self, path, step_size=0.5):
        if not path or len(path) < 2:
            return path
        new_path = [path[0]]
        for i in range(len(path)-1):
            p1 = np.array(path[i]); p2 = np.array(path[i+1])
            dist = np.linalg.norm(p2 - p1)
            if dist <= step_size:
                new_path.append(path[i+1])
            else:
                num_steps = int(np.ceil(dist / step_size))
                for j in range(1, num_steps):
                    t = j / num_steps
                    interp = p1 + t * (p2 - p1)
                    new_path.append(interp.tolist())
                new_path.append(path[i+1])
        return new_path

    def listen_for_status(self):
        self.listening = True
        while self.listening:
            try:
                if not self.sock:
                    break
                self.sock.settimeout(0.5)
                raw = self.sock.recv(1024).decode()
                if raw:
                    data = json.loads(raw)
                    if data.get("status") == "LANDED":
                        self.root.after(0, self.landing_done)
                        break
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Listener error: {e}")
                break
        self.listening = False

    def landing_done(self):
        self.lbl_metrics.config(text="Status: LANDED successfully")
        self.btn_fly.config(state="disabled")
        messagebox.showinfo("Info", "Drone telah mendarat di goal.")

    def fly(self):
        if self.data['path'] and self.sock:
            try:
                drone_radius = float(self.radius_var.get())
                smooth_path = self.interpolate_path(self.data['path'], step_size=0.5)
                offset_z = 0.2
                path_with_offset = [[p[0], p[1], p[2] + offset_z] for p in smooth_path]

                msg = {"command": "START_SIM", "path": path_with_offset, "drone_radius": drone_radius}
                self.sock.sendall(json.dumps(msg).encode())
                self.lbl_metrics.config(text="Executing path... (waiting for landing)")
                self.btn_fly.config(state="disabled")

                if not self.listening:
                    threading.Thread(target=self.listen_for_status, daemon=True).start()

            except Exception as e:
                messagebox.showerror("Socket Error", f"Gagal mengirim perintah terbang: {e}")

    def reset_sim(self):
        if self.sock:
            try:
                self.sock.sendall(json.dumps({"command": "RESET"}).encode())
            except:
                pass
        self.listening = False
        self.data['path'] = []
        self.btn_fly.config(state="normal" if self.data['path'] else "disabled")
        self.lbl_metrics.config(text="Status: Reset")
        self.draw_world()

    # ============================================================
    #  METODE BENCHMARK (TAMBAHAN)
    # ============================================================
    def open_benchmark_dialog(self):
        if not self.data.get('start') or not self.data.get('goal') or not self.data.get('obs'):
            messagebox.showwarning("Benchmark", "Silakan load map terlebih dahulu.")
            return

        # Validasi bias sebelum membuka dialog? Atau saat jalankan? Lebih baik saat jalankan.
        # Tapi kita perlu memastikan bias valid saat benchmark dijalankan.
        # Kita akan validasi di dalam on_run.

        dialog = tk.Toplevel(self.root)
        dialog.title("Benchmark Settings")
        dialog.geometry("380x280")
        dialog.resizable(False, False)
        dialog.grab_set()  # Modal

        # Variabel input
        iter_var_dialog = tk.StringVar(value=self.iter_var.get())
        trials_var = tk.StringVar(value="5")
        file_var = tk.StringVar(value="benchmark.xlsx")

        # Form
        tk.Label(dialog, text="Iterasi:").pack(anchor='w', padx=20, pady=(20,5))
        tk.Entry(dialog, textvariable=iter_var_dialog).pack(fill='x', padx=20)

        tk.Label(dialog, text="Jumlah Percobaan:").pack(anchor='w', padx=20, pady=(10,5))
        tk.Entry(dialog, textvariable=trials_var).pack(fill='x', padx=20)

        tk.Label(dialog, text="Nama File (.xlsx):").pack(anchor='w', padx=20, pady=(10,5))
        file_frame = tk.Frame(dialog)
        file_frame.pack(fill='x', padx=20)
        tk.Entry(file_frame, textvariable=file_var).pack(side='left', fill='x', expand=True)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_benchmark_file(file_var)).pack(side='left', padx=5)

        # Progress bar (indeterminate)
        progress = ttk.Progressbar(dialog, mode='indeterminate', length=250)
        # Awalnya tidak ditampilkan
        status_label = tk.Label(dialog, text="")
        status_label.pack(pady=5)

        def on_run():
            # Validasi bias terlebih dahulu
            bias = self.validate_bias()
            if bias is None:
                return

            try:
                iterations = int(iter_var_dialog.get())
                if iterations <= 0:
                    raise ValueError
                trials = int(trials_var.get())
                if trials <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Iterasi dan jumlah percobaan harus angka positif.")
                return
            filename = file_var.get().strip()
            if not filename:
                messagebox.showerror("Error", "Nama file tidak boleh kosong.")
                return
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'

            # Disable tombol dan tampilkan progress
            run_btn.config(state='disabled')
            cancel_btn.config(state='disabled')
            progress.pack(pady=10)
            progress.start(10)
            status_label.config(text="Benchmarking...")

            # Jalankan di thread terpisah
            threading.Thread(
                target=self.run_benchmark_thread,
                args=(iterations, trials, filename, progress, status_label, dialog, bias),
                daemon=True
            ).start()

        def on_cancel():
            dialog.destroy()

        run_btn = ttk.Button(dialog, text="Jalankan", command=on_run)
        run_btn.pack(side='left', padx=20, pady=20)
        cancel_btn = ttk.Button(dialog, text="Cancel", command=on_cancel)
        cancel_btn.pack(side='right', padx=20, pady=20)

        # Simpan referensi untuk akses dari thread
        dialog.iter_var = iter_var_dialog
        dialog.trials_var = trials_var
        dialog.file_var = file_var
        dialog.progress = progress
        dialog.status_label = status_label
        dialog.run_btn = run_btn
        dialog.cancel_btn = cancel_btn

    def browse_benchmark_file(self, file_var):
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            file_var.set(filename)

    def run_benchmark_thread(self, iterations, trials, filename, progress, status_label, dialog, bias):
        results = []
        try:
            drone_radius = float(self.radius_var.get())
        except:
            drone_radius = 0.6

        goal_bias, uniform_bias, gaussian_bias = bias

        for i in range(1, trials + 1):
            # Update status label via root.after agar thread-safe
            self.root.after(0, lambda idx=i: status_label.config(text=f"Trial {idx}/{trials}..."))
            start_time = time.time()

            # Buat engine baru setiap trial
            eng = PlanningEngine3D(
                self.data['start'], self.data['goal'], self.data['obs'],
                drone_radius=drone_radius,
                bounds_xy=(-10, 10), bounds_z=(0, 10),
                goal_bias=goal_bias,
                uniform_bias=uniform_bias,
                gaussian_bias=gaussian_bias
            )

            if self.algo_var.get() == "rrtstar":
                raw_path = eng.solve_rrt_star(iterations)
            else:
                raw_path = eng.solve_multibias(iterations)

            elapsed = time.time() - start_time
            if raw_path:
                path = eng.smooth_path(raw_path)
                cost = calculate_path_cost_3d(path)
            else:
                path = None
                cost = None  # None berarti gagal

            nodes = len(eng.node_list)
            results.append({
                'nomor': i,
                'cost': cost,
                'time': elapsed,
                'nodes': nodes
            })

        # Simpan ke Excel
        try:
            self.save_benchmark_to_excel(results, filename)
            self.root.after(0, lambda: self.benchmark_done(dialog, filename))
        except Exception as e:
            self.root.after(0, lambda: self.benchmark_error(dialog, str(e)))

    def save_benchmark_to_excel(self, results, filename):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Benchmark"
        ws.append(["Nomor", "Cost (m)", "Time (s)", "Node"])
        for r in results:
            cost_val = r['cost'] if r['cost'] is not None else "FAILED"
            ws.append([r['nomor'], cost_val, round(r['time'], 4), r['nodes']])
        wb.save(filename)

    def benchmark_done(self, dialog, filename):
        dialog.destroy()
        messagebox.showinfo("Benchmark", f"Benchmark selesai!\nHasil disimpan di:\n{filename}")

    def benchmark_error(self, dialog, error_msg):
        dialog.destroy()
        messagebox.showerror("Error", f"Benchmark gagal:\n{error_msg}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DroneApp3D(root)
    root.mainloop()