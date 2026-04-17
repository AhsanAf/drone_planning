import tkinter as tk
from tkinter import messagebox, ttk
import socket, json, threading, math, random, time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np

# --- Helper Math ---
def is_point_in_rect(px, py, obs, margin=0.2):
    """Mengecek apakah titik berada di dalam rintangan dengan margin keamanan."""
    tx, ty = px - obs['x'], py - obs['y']
    c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
    lx, ly = tx * c - ty * s, tx * s + ty * c
    return - (obs['w']/2 + margin) <= lx <= (obs['w']/2 + margin) and \
           - (obs['h']/2 + margin) <= ly <= (obs['h']/2 + margin)

def get_corners(obs):
    """Mendapatkan koordinat sudut rintangan untuk visualisasi."""
    c, s = math.cos(obs['rot']), math.sin(obs['rot'])
    hw, hh = obs['w']/2, obs['h']/2
    pts = [(-hw,-hh), (hw,-hh), (hw,hh), (-hw,hh)]
    return [[obs['x'] + (x*c - y*s), obs['y'] + (x*s + y*c)] for x, y in pts]

def calculate_path_cost(path):
    if not path or len(path) < 2: return 0
    return sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))

# --- Planning Engine ---
class PlanningEngine:
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles
        self.node_list = []
        
        # PARAMETER 
        self.expand_dis = 0.5        # Langkah ekspansi
        self.search_radius = 2.0     # Radius pencarian tetangga (Khusus RRT*)
        self.max_gaussian_attempts = 50
        
        # Probabilitas Bias
        self.goal_bias = 0.1         # 10% Goal 
        self.gaussian_bias = 0.5     # 50% Gaussian 
        self.uniform_bias = 0.4      # 40% Uniform

    def check_collision(self, x, y):
        for o in self.obstacles:
            if is_point_in_rect(x, y, o): return True
        return False

    def is_line_safe(self, p1, p2):
        """Validasi garis agar tidak menembus tembok."""
        dist = math.dist(p1, p2)
        # Resolusi pengecekan: 10 cm
        steps = max(2, int(dist / 0.1)) 
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t*(p2[0]-p1[0])
            y = p1[1] + t*(p2[1]-p1[1])
            if self.check_collision(x, y):
                return False
        return True

    def get_gaussian_sample(self):
        """
        [ALGORITMA GAUSSIAN SAMPLING]
        Mencari titik tengah antara (Free Space) dan (Obstacle Space).
        """
        for _ in range(self.max_gaussian_attempts):
            # 1. Ambil titik acak (Free) - q1
            x1, y1 = random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)
            if self.check_collision(x1, y1): continue 
            
            # 2. Ambil titik kedua di sekitarnya (Distribusi Normal) - q2 (Rumus 4)
            sigma = random.uniform(0.5, 2.0)
            x2 = max(-7.5, min(7.5, random.gauss(x1, sigma)))
            y2 = max(-7.5, min(7.5, random.gauss(y1, sigma)))
            
            # 3. Jika titik kedua TABRAKAN (Obstacle), ambil tengahnya
            if self.check_collision(x2, y2):
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                # Pastikan titik tengahnya aman
                if not self.check_collision(mx, my):
                    return [mx, my] # Berhasil menemukan sampel di boundary
                    
        # Fallback jika gagal: Uniform random
        return [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]])
        # Jika dekat dan garis aman, goal tercapai
        if dist <= self.expand_dis * 1.5:
            if self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]):
                return True
        return False

    # =========================================================
    # ALGORITMA 1: Multi-Bias RRT (STANDARD / TANPA REWIRING)
    # =========================================================
    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]
        
        for _ in range(max_iter):
            # --- 1. SAMPLING MULTI-BIAS (Sesuai Proporsi Draft) ---
            p = random.random()
            if p < self.goal_bias:
                # Goal Bias (10%)
                rnd = [self.goal["x"], self.goal["y"]]
            elif p < self.goal_bias + self.gaussian_bias:
                # Gaussian Bias (50%)
                rnd = self.get_gaussian_sample()
            else:
                # Uniform Bias (40%)
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
            
            # --- 2. NEAREST NEIGHBOR ---
            nearest = min(self.node_list, 
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            
            # --- 3. STEER (Extend) ---
            dist = math.dist([nearest["x"], nearest["y"]], rnd)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            step = min(self.expand_dis, dist)
            
            new_node = {
                "x": nearest["x"] + step * math.cos(theta),
                "y": nearest["y"] + step * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + step
            }
            
            # --- 4. COLLISION CHECK ---
            if self.check_collision(new_node["x"], new_node["y"]): continue
            
            # Validasi garis (Edge Check) agar tidak tembus tembok
            if not self.is_line_safe([nearest["x"], nearest["y"]], [new_node["x"], new_node["y"]]):
                continue
            
            # --- 5. ADD TO TREE (Langsung tambah, tanpa optimasi/rewiring) ---
            self.node_list.append(new_node)
            
            # Cek Goal
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)
                
        # [PERBAIKAN LOGIKA]
        # Jika iterasi habis dan goal tidak ketemu, kembalikan None (GAGAL)
        # Jangan kembalikan path terdekat agar tidak menyesatkan.
        return None

    # =========================================================
    # ALGORITMA 2: RRT* (DENGAN REWIRING & OPTIMASI)
    # [PEMBANDING / BASELINE]
    # =========================================================
    def solve_rrt_star(self, max_iter=500):
        self.node_list = [self.start]
        
        for _ in range(max_iter):
            # Sampling: Uniform (95%) + Goal Bias Dikit (5%)
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
                
            # Nearest
            nearest = min(self.node_list, 
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            
            # Steer
            dist = math.dist([nearest["x"], nearest["y"]], rnd)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            step = min(self.expand_dis, dist)
            
            new_node = {
                "x": nearest["x"] + step * math.cos(theta),
                "y": nearest["y"] + step * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + step
            }
            
            if self.check_collision(new_node["x"], new_node["y"]): continue
            if not self.is_line_safe([nearest["x"], nearest["y"]], [new_node["x"], new_node["y"]]): continue

            # --- FITUR KHUSUS RRT*: CHOOSE BEST PARENT ---
            near_nodes = [n for n in self.node_list 
                         if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= self.search_radius**2]
            
            # Cari parent yang memberikan total cost terendah
            for near in near_nodes:
                d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                if near["cost"] + d < new_node["cost"]:
                    if self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"] = near["cost"] + d
                        new_node["parent"] = near
            
            self.node_list.append(new_node)
            
            # --- FITUR KHUSUS RRT*: REWIRING ---
            # Cek apakah tetangga bisa lewat node baru agar lebih hemat
            for near in near_nodes:
                if near == new_node["parent"]: continue
                d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                if new_node["cost"] + d < near["cost"]:
                    if self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"] = new_node
                        near["cost"] = new_node["cost"] + d

        # RRT*
        # Cek apakah ada yang reach goal di akhir
        reachable = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable:
            best = min(reachable, key=lambda n: n["cost"])
            return self.extract_path(best)
        
        # [PERBAIKAN LOGIKA] Jika tidak ada yang sampai goal -> Gagal
        return None

    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=30):
        """Menghaluskan jalur zigzag (Path Smoothing)."""
        if not path or len(path) < 3: return path
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
                if curr < len(path): smoothed.append(path[curr])
        return smoothed

# --- GUI Class ---
class DroneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Bias RRT vs RRT* (Comparison)")
        self.sock = None
        self.data = {"start": None, "goal": None, "obs": [], "path": []}
        self.running = False
        
        # UI Layout
        side = tk.Frame(root, width=280, bg="#2c3e50")
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)

        tk.Label(side, text="ALGORITHM", bg="#2c3e50", fg="white", font=("Arial", 11, "bold")).pack(pady=15)
        self.algo_var = tk.StringVar(value="multibias")
        
        # Opsi Algoritma yang Jelas Bedanya
        tk.Radiobutton(side, text="RRT* ", variable=self.algo_var, value="rrtstar", 
                      bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Arial", 9)).pack(anchor="w", padx=20, pady=2)
        tk.Radiobutton(side, text="Multi-Bias RRT", variable=self.algo_var, value="multibias", 
                      bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Arial", 9)).pack(anchor="w", padx=20, pady=2)

        # Iterasi
        iter_frame = tk.Frame(side, bg="#2c3e50")
        iter_frame.pack(fill='x', padx=20, pady=15)
        tk.Label(iter_frame, text="MAX ITERATIONS", bg="#2c3e50", fg="white", font=("Arial", 9, "bold")).pack(anchor="w")
        self.iter_var = tk.StringVar(value="500")
        tk.Entry(iter_frame, textvariable=self.iter_var, width=15).pack(pady=5)

        # Buttons
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

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
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

    def get_map(self):
        if not self.sock: return
        try:
            self.sock.sendall(json.dumps({"command": "GET_MAP"}).encode())
            data = self.sock.recv(65536) # Buffer besar
            if data:
                r = json.loads(data.decode())
                self.data.update({"start": r['start'], "goal": r['goal'], "obs": r['obstacles']})
                self.draw_world()
                self.btn_run.config(state="normal")
        except Exception as e: messagebox.showerror("Error", str(e))

    def reset_sim(self):
        if self.sock:
            self.sock.sendall(json.dumps({"command": "RESET"}).encode())
            self.data['path'] = []
            self.draw_world()

    def draw_world(self, tree_edges=[]):
        self.ax.clear()
        # Gambar Rintangan
        for o in self.data['obs']: 
            self.ax.add_patch(patches.Polygon(get_corners(o), color='#34495e', alpha=0.9))
        
        # Gambar Tree (Visualisasi eksplorasi)
        if tree_edges:
            lc = LineCollection(tree_edges, colors='#bdc3c7', linewidths=0.5, alpha=0.5)
            self.ax.add_collection(lc)

        # Gambar Start/Goal
        if self.data['start']: self.ax.plot(*self.data['start'], 'go', markersize=10, label='Start')
        if self.data['goal']: self.ax.plot(*self.data['goal'], 'r*', markersize=14, label='Goal')
        
        # Gambar Path Akhir
        if self.data['path'] and len(self.data['path']) > 1:
            p_np = np.array(self.data['path'])
            # Warna beda untuk algo beda
            c = 'cyan' if self.algo_var.get() == "multibias" else 'magenta'
            lbl = 'Multi-Bias Path' if self.algo_var.get() == "multibias" else 'RRT* Path'
            self.ax.plot(p_np[:,0], p_np[:,1], color=c, linewidth=3, label=lbl)

        self.ax.set_xlim(-8, 8); self.ax.set_ylim(-8, 8)
        self.ax.set_title(f"Environment Map ({len(self.data['obs'])} Obstacles)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.2)
        self.canvas.draw_idle()

    def start_thread(self):
        if self.running: return
        self.running = True
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.solve, args=(int(self.iter_var.get()),), daemon=True).start()

    def solve(self, max_iter):
        t0 = time.time()
        eng = PlanningEngine(self.data['start'], self.data['goal'], self.data['obs'])
        
        if self.algo_var.get() == "rrtstar":
            algo_name = "RRT*"
            raw_path = eng.solve_rrt_star(max_iter)
        else:
            algo_name = "Multi-Bias"
            raw_path = eng.solve_multibias(max_iter) # Standard RRT
            
        # --- PENTING UNTUK DATA ---
        # Hanya smooth jika path ditemukan (raw_path is not None)
        if raw_path:
            path = eng.smooth_path(raw_path) 
        else:
            path = None
            
        dt = time.time() - t0
        
        # Siapkan visualisasi tree
        tree_edges = []
        if len(eng.node_list) < 2000: # Jangan gambar kalau terlalu banyak node (berat)
            for n in eng.node_list:
                if n["parent"]:
                    tree_edges.append([(n["x"], n["y"]), (n["parent"]["x"], n["parent"]["y"])])
        
        self.root.after(0, self.planning_done, path, tree_edges, dt, len(eng.node_list), algo_name, path is not None)

    def planning_done(self, path, tree_edges, dt, nodes, algo_name, success):
        self.running = False
        self.btn_run.config(state="normal")
        
        if not success:
            self.lbl_metrics.config(text=f"Status: {algo_name} FAILED\nTime: {dt:.3f}s\nNodes: {nodes}")
            messagebox.showwarning("Fail", "Path not found!")
            self.draw_world(tree_edges) # Tetap gambar tree biar kelihatan mentok dimana
            return
        
        self.data['path'] = path
        self.draw_world(tree_edges)
        
        cost = calculate_path_cost(path)
        self.lbl_metrics.config(text=f"[{algo_name} RESULT]\nTime: {dt:.3f}s (Fast?)\nCost: {cost:.2f}m\nNodes: {nodes}\nStatus: SUCCESS")
        self.btn_fly.config(state="normal")

    def fly(self):
        if self.data['path'] and self.sock:
            self.sock.sendall(json.dumps({"command": "START_SIM", "path": self.data['path']}).encode())

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneApp(root)
    root.mainloop()