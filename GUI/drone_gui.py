import tkinter as tk
from tkinter import messagebox, ttk
import socket, json, threading, math, random, time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np

# --- Helper Math ---
def is_point_in_rect(px, py, obs, margin=0.45):
    """Mengecek apakah titik berada di dalam rintangan yang terotasi."""
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

        # ---------- BATAS SAMPLING DINAMIS ----------
        self.bounds = self._compute_bounds()
        self.min_x, self.max_x = self.bounds[0], self.bounds[1]
        self.min_y, self.max_y = self.bounds[2], self.bounds[3]
        # --------------------------------------------

        # ---------- PARAMETER UNTUK CELAH SEMPIT ----------
        self.margin = 0.2                     # safety margin obstacle
        self.expand_dis = 0.45               # langkah ekspansi
        self.search_radius = 2.0            # radius pencarian tetangga (RRT*)
        self.goal_bias = 0.1                # 10% goal
        self.gaussian_bias = 0.5            # 50% Gaussian
        self.uniform_bias = 0.4             # 40% uniform
        self.max_gaussian_attempts = 200    # percobaan Gaussian
        self.gaussian_sigma_range = (0.8, 2.5)  # sigma bervariasi
        self.check_edge = True              # selalu validasi garis
        # ---------------------------------------------------

        self.node_list = []

    # ================== HITUNG BATAS SAMPLING ==================
    def _compute_bounds(self):
        """Batas dinamis dari start, goal, dan semua obstacle + padding 2m."""
        xs = [self.start["x"], self.goal["x"]]
        ys = [self.start["y"], self.goal["y"]]
        for o in self.obstacles:
            corners = get_corners(o)  # dari drone_gui.py
            for (x, y) in corners:
                xs.append(x)
                ys.append(y)
        return (min(xs) - 2.0, max(xs) + 2.0,
                min(ys) - 2.0, max(ys) + 2.0)

    # ================== COLLISION DENGAN MARGIN KECIL ==================
    def is_point_in_rect(self, px, py, obs):
        tx, ty = px - obs['x'], py - obs['y']
        c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
        lx, ly = tx * c - ty * s, tx * s + ty * c
        return (- (obs['w']/2 + self.margin) <= lx <= (obs['w']/2 + self.margin) and
                - (obs['h']/2 + self.margin) <= ly <= (obs['h']/2 + self.margin))

    def check_collision(self, x, y):
        for o in self.obstacles:
            if self.is_point_in_rect(x, y, o):
                return True
        return False

    # ================== CEK GARIS DENGAN RESOLUSI TINGGI ==================
    def is_line_safe(self, p1, p2, resolution=0.04):
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / resolution))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.check_collision(x, y):
                return False
        return True

    # ================== GAUSSIAN SAMPLING (EKSPLORATIF) ==================
    def get_gaussian_sample(self):
        for _ in range(self.max_gaussian_attempts):
            x1 = random.uniform(self.min_x, self.max_x)
            y1 = random.uniform(self.min_y, self.max_y)
            if self.check_collision(x1, y1):
                continue
            sigma = random.uniform(*self.gaussian_sigma_range)
            x2 = random.gauss(x1, sigma)
            y2 = random.gauss(y1, sigma)
            x2 = max(self.min_x, min(self.max_x, x2))
            y2 = max(self.min_y, min(self.max_y, y2))
            if self.check_collision(x2, y2):
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                if not self.check_collision(mx, my):
                    return [mx, my]
        return [random.uniform(self.min_x, self.max_x),
                random.uniform(self.min_y, self.max_y)]

    # ================== CEK APAKAH GOAL TERJANGKAU ==================
    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]])
        return (dist <= self.expand_dis * 1.2 and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    # ================== MULTI-BIAS RRT* (3 BIAS + REWIRING) ==================
    def solve_multibias(self, max_iter=500):
        """
        RRT* dengan sampling Multi-Bias (Goal, Gaussian, Uniform).
        Tetap 3 bias, tetapi menggunakan pemilihan parent terbaik dan rewiring.
        """
        self.node_list = [self.start]

        for i in range(max_iter):
            # ---------- SAMPLING (3 BIAS MURNI) ----------
            p = random.random()
            if p < self.goal_bias:
                rnd = [self.goal["x"], self.goal["y"]]
            elif p < self.goal_bias + self.gaussian_bias:
                rnd = self.get_gaussian_sample()
            else:
                rnd = [random.uniform(self.min_x, self.max_x),
                       random.uniform(self.min_y, self.max_y)]

            # ---------- NODE TERDEKAT ----------
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)

            # ---------- STEERING ----------
            dx = rnd[0] - nearest["x"]
            dy = rnd[1] - nearest["y"]
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            theta = math.atan2(dy, dx)
            new_x = nearest["x"] + min(self.expand_dis, dist) * math.cos(theta)
            new_y = nearest["y"] + min(self.expand_dis, dist) * math.sin(theta)

            # ---------- VALIDASI POSISI ----------
            if self.check_collision(new_x, new_y):
                continue

            # ---------- VALIDASI GARIS (PARENT -> NEW) ----------
            if not self.is_line_safe([nearest["x"], nearest["y"]], [new_x, new_y]):
                continue

            # ---------- BUAT NODE BARU DENGAN BIAYA SEMENTARA ----------
            new_node = {
                "x": new_x,
                "y": new_y,
                "parent": nearest,
                "cost": nearest["cost"] + math.dist([nearest["x"], nearest["y"]], [new_x, new_y])
            }

            # ---------- RRT*: CARI PARENT TERBAIK DI SEKITAR ----------
            near_nodes = [n for n in self.node_list
                         if (n["x"]-new_x)**2 + (n["y"]-new_y)**2 <= self.search_radius**2]
            for near in near_nodes:
                if self.is_line_safe([near["x"], near["y"]], [new_x, new_y]):
                    d = math.dist([near["x"], near["y"]], [new_x, new_y])
                    if near["cost"] + d < new_node["cost"]:
                        new_node["cost"] = near["cost"] + d
                        new_node["parent"] = near

            self.node_list.append(new_node)

            # ---------- RRT*: REWIRING TETANGGA ----------
            for near in near_nodes:
                if near == new_node["parent"]:
                    continue
                if self.is_line_safe([new_x, new_y], [near["x"], near["y"]]):
                    d = math.dist([new_x, new_y], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"]:
                        near["parent"] = new_node
                        near["cost"] = new_node["cost"] + d

            # ---------- CEK GOAL ----------
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)

            # Heuristik jarak dekat
            dist_to_goal = math.dist([new_x, new_y],
                                     [self.goal["x"], self.goal["y"]])
            if dist_to_goal < self.expand_dis * 2:
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)

        # ---------- FINAL CHECK ----------
        reachable = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable:
            best = min(reachable, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    # ================== RRT* (PEMBANDING, UNIFORM SAMPLING) ==================
    def solve_rrt_star(self, max_iter=500):
        """RRT* standar dengan uniform sampling (5% goal bias)."""
        self.node_list = [self.start]
        for iteration in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(self.min_x, self.max_x),
                       random.uniform(self.min_y, self.max_y)]
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            new_node = {
                "x": nearest["x"] + self.expand_dis * math.cos(theta),
                "y": nearest["y"] + self.expand_dis * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + self.expand_dis
            }
            if not self.check_collision(new_node["x"], new_node["y"]):
                near_nodes = [n for n in self.node_list
                            if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= self.search_radius**2]
                for near in near_nodes:
                    d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                    if near["cost"] + d < new_node["cost"] and self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"], new_node["parent"] = near["cost"] + d, near
                self.node_list.append(new_node)
                for near in near_nodes:
                    d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"] and self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"], near["cost"] = new_node, new_node["cost"] + d
        reachable = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable:
            best = min(reachable, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    # ================== EKSTRAK PATH & SMOOTHING ==================
    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        curr = 0
        iterations = 0
        while curr < len(path) - 1 and iterations < max_smooth_iter:
            iterations += 1
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
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles

        # ---------- BATAS SAMPLING DINAMIS ----------
        self.bounds = self._compute_bounds()
        self.min_x, self.max_x = self.bounds[0], self.bounds[1]
        self.min_y, self.max_y = self.bounds[2], self.bounds[3]
        # --------------------------------------------

        # ---------- PARAMETER UNTUK CELAH SEMPIT ----------
        self.margin = 0.2                     # obstacle safety margin
        self.expand_dis = 0.45               # langkah ekspansi (0.4~0.5)
        self.goal_bias = 0.1                # 10% goal
        self.gaussian_bias = 0.5            # 50% Gaussian
        self.uniform_bias = 0.4             # 40% uniform
        self.max_gaussian_attempts = 200    # percobaan Gaussian
        self.gaussian_sigma_range = (0.8, 2.5)  # sigma bervariasi
        self.check_edge = True              # selalu validasi edge
        # ---------------------------------------------------

        self.node_list = []

    # ================== HITUNG BATAS SAMPLING ==================
    def _compute_bounds(self):
        """Menentukan batas sampling dari start, goal, dan semua obstacle."""
        xs = [self.start["x"], self.goal["x"]]
        ys = [self.start["y"], self.goal["y"]]
        for o in self.obstacles:
            # ambil keempat sudut obstacle
            corners = get_corners(o)  # fungsi dari drone_gui.py
            for (x, y) in corners:
                xs.append(x)
                ys.append(y)
        min_x = min(xs) - 2.0   # padding 2 meter
        max_x = max(xs) + 2.0
        min_y = min(ys) - 2.0
        max_y = max(ys) + 2.0
        return (min_x, max_x, min_y, max_y)

    # ================== COLLISION DENGAN MARGIN KECIL ==================
    def is_point_in_rect(self, px, py, obs):
        tx, ty = px - obs['x'], py - obs['y']
        c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
        lx, ly = tx * c - ty * s, tx * s + ty * c
        return (- (obs['w']/2 + self.margin) <= lx <= (obs['w']/2 + self.margin) and
                - (obs['h']/2 + self.margin) <= ly <= (obs['h']/2 + self.margin))

    def check_collision(self, x, y):
        for o in self.obstacles:
            if self.is_point_in_rect(x, y, o):
                return True
        return False

    # ================== CEK GARIS DENGAN RESOLUSI TINGGI ==================
    def is_line_safe(self, p1, p2, resolution=0.04):
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / resolution))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.check_collision(x, y):
                return False
        return True

    # ================== GAUSSIAN SAMPLING (DENGAN BOUNDS DINAMIS) ==================
    def get_gaussian_sample(self):
        for _ in range(self.max_gaussian_attempts):
            # 1. Titik FREE seragam di seluruh arena
            x1 = random.uniform(self.min_x, self.max_x)
            y1 = random.uniform(self.min_y, self.max_y)
            if self.check_collision(x1, y1):
                continue

            # 2. Sigma bervariasi
            sigma = random.uniform(*self.gaussian_sigma_range)

            # 3. Titik kedua dari Gaussian
            x2 = random.gauss(x1, sigma)
            y2 = random.gauss(y1, sigma)
            x2 = max(self.min_x, min(self.max_x, x2))
            y2 = max(self.min_y, min(self.max_y, y2))

            # 4. Syarat: titik kedua harus di DALAM obstacle
            if self.check_collision(x2, y2):
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                # 5. Titik tengah harus FREE
                if not self.check_collision(mx, my):
                    return [mx, my]

        # Fallback: uniform random dalam bounds
        return [random.uniform(self.min_x, self.max_x),
                random.uniform(self.min_y, self.max_y)]

    # ================== CEK APAKAH GOAL TERJANGKAU ==================
    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]])
        return (dist <= self.expand_dis * 1.2 and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    # ================== MULTI-BIAS DENGAN STEERING YANG BENAR ==================
    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]

        for i in range(max_iter):
            # ---------- SAMPLING (3 BIAS MURNI) ----------
            p = random.random()
            if p < self.goal_bias:
                rnd = [self.goal["x"], self.goal["y"]]
            elif p < self.goal_bias + self.gaussian_bias:
                rnd = self.get_gaussian_sample()
            else:
                rnd = [random.uniform(self.min_x, self.max_x),
                       random.uniform(self.min_y, self.max_y)]

            # ---------- NODE TERDEKAT ----------
            nearest = min(self.node_list,
                         key=lambda n: (n["x"] - rnd[0])**2 + (n["y"] - rnd[1])**2)

            # ---------- STEERING (STANDAR RRT) ----------
            dx = rnd[0] - nearest["x"]
            dy = rnd[1] - nearest["y"]
            dist_to_rnd = math.hypot(dx, dy)

            if dist_to_rnd < self.expand_dis:
                new_node = {
                    "x": rnd[0],
                    "y": rnd[1],
                    "parent": nearest,
                    "cost": nearest["cost"] + dist_to_rnd
                }
            else:
                theta = math.atan2(dy, dx)
                new_node = {
                    "x": nearest["x"] + self.expand_dis * math.cos(theta),
                    "y": nearest["y"] + self.expand_dis * math.sin(theta),
                    "parent": nearest,
                    "cost": nearest["cost"] + self.expand_dis
                }

            # ---------- VALIDASI NODE BARU ----------
            if self.check_collision(new_node["x"], new_node["y"]):
                continue
            if self.check_edge:
                if not self.is_line_safe([nearest["x"], nearest["y"]],
                                         [new_node["x"], new_node["y"]]):
                    continue

            self.node_list.append(new_node)

            # ---------- CEK GOAL ----------
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)

            # Heuristik jarak dekat
            dist_to_goal = math.dist([new_node["x"], new_node["y"]],
                                     [self.goal["x"], self.goal["y"]])
            if dist_to_goal < self.expand_dis * 2:
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)

        # Final check
        if self.node_list:
            last = min(self.node_list,
                      key=lambda n: math.dist([n["x"], n["y"]],
                                             [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(last):
                return self.extract_path(last)
        return None

    # ================== RRT* (TIDAK DIUBAH) ==================
    def solve_rrt_star(self, max_iter=500):
        # ... (kode asli, tidak diubah, namun pastikan menggunakan self.bounds jika perlu)
        # (Disarankan juga mengganti random uniform dengan self.bounds pada RRT*)
        # Namun untuk konsistensi, berikut versi yang sudah disesuaikan bounds-nya:
        self.node_list = [self.start]
        for iteration in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(self.min_x, self.max_x),
                       random.uniform(self.min_y, self.max_y)]
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            new_node = {
                "x": nearest["x"] + 0.8 * math.cos(theta),
                "y": nearest["y"] + 0.8 * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + 0.8
            }
            if not self.check_collision(new_node["x"], new_node["y"]):
                near_nodes = [n for n in self.node_list
                            if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= 2.2**2]
                for near in near_nodes:
                    d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                    if near["cost"] + d < new_node["cost"] and self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"], new_node["parent"] = near["cost"] + d, near
                self.node_list.append(new_node)
                for near in near_nodes:
                    d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"] and self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"], near["cost"] = new_node, new_node["cost"] + d
        reachable_nodes = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable_nodes:
            best = min(reachable_nodes, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    # ================== EKSTRAK PATH & SMOOTHING ==================
    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        curr = 0
        iterations = 0
        while curr < len(path) - 1 and iterations < max_smooth_iter:
            iterations += 1
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
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles

        # ---------- PARAMETER YANG DIPERBAIKI UNTUK CELAH SEMPIT ----------
        self.margin = 0.2                     # dari 0.45 → 0.2
        self.expand_dis = 0.4                # dari 0.8 → 0.4
        self.goal_bias = 0.1                # 10% (tetap)
        self.gaussian_bias = 0.5            # 50% (tetap)
        self.uniform_bias = 0.4             # 40% (tetap)
        self.max_gaussian_attempts = 200    # dari 3 → 200
        self.gaussian_sigma_range = (0.8, 2.5)  # sigma bervariasi
        self.check_edge = True              # selalu cek garis parent → new_node
        # ----------------------------------------------------------------

        self.node_list = []

    # ================== COLLISION DENGAN MARGIN KECIL ==================
    def is_point_in_rect(self, px, py, obs):
        tx, ty = px - obs['x'], py - obs['y']
        c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
        lx, ly = tx * c - ty * s, tx * s + ty * c
        return (- (obs['w']/2 + self.margin) <= lx <= (obs['w']/2 + self.margin) and
                - (obs['h']/2 + self.margin) <= ly <= (obs['h']/2 + self.margin))

    def check_collision(self, x, y):
        for o in self.obstacles:
            if self.is_point_in_rect(x, y, o):
                return True
        return False

    # ================== CEK GARIS DENGAN RESOLUSI TINGGI ==================
    def is_line_safe(self, p1, p2, resolution=0.04):   # dari 0.3 → 0.04
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / resolution))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.check_collision(x, y):
                return False
        return True

    # ================== GAUSSIAN SAMPLING EKSPLORATIF ==================
    def get_gaussian_sample(self):
        """
        - Titik FREE dipilih SERAGAM di seluruh arena.
        - Sigma bervariasi (kadang kecil, kadang besar).
        - Hanya pakai jika titik kedua di dalam obstacle.
        - Titik tengah harus FREE.
        """
        for _ in range(self.max_gaussian_attempts):
            # 1. Titik pertama: uniform random, HARUS FREE
            x1 = random.uniform(-7.5, 7.5)
            y1 = random.uniform(-7.5, 7.5)
            if self.check_collision(x1, y1):
                continue

            # 2. Sigma bervariasi
            sigma = random.uniform(*self.gaussian_sigma_range)

            # 3. Titik kedua dari Gaussian
            x2 = random.gauss(x1, sigma)
            y2 = random.gauss(y1, sigma)
            x2 = max(-7.5, min(7.5, x2))
            y2 = max(-7.5, min(7.5, y2))

            # 4. Syarat: titik kedua harus di DALAM obstacle
            if self.check_collision(x2, y2):
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                # 5. Titik tengah harus FREE
                if not self.check_collision(mx, my):
                    return [mx, my]

        # Fallback: uniform random
        return [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

    # ================== CEK APAKAH GOAL TERJANGKAU ==================
    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]])
        return (dist <= self.expand_dis * 1.2 and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    # ================== MULTI-BIAS DENGAN STEERING YANG BENAR ==================
    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]

        for i in range(max_iter):
            # ---------- SAMPLING (3 BIAS MURNI) ----------
            p = random.random()
            if p < self.goal_bias:
                rnd = [self.goal["x"], self.goal["y"]]
            elif p < self.goal_bias + self.gaussian_bias:
                rnd = self.get_gaussian_sample()
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

            # ---------- NODE TERDEKAT ----------
            nearest = min(self.node_list,
                         key=lambda n: (n["x"] - rnd[0])**2 + (n["y"] - rnd[1])**2)

            # ---------- STEERING (STANDAR RRT) ----------
            dx = rnd[0] - nearest["x"]
            dy = rnd[1] - nearest["y"]
            dist_to_rnd = math.hypot(dx, dy)

            if dist_to_rnd < self.expand_dis:
                # Sampel lebih dekat dari langkah maksimum → langsung ke sampel
                new_node = {
                    "x": rnd[0],
                    "y": rnd[1],
                    "parent": nearest,
                    "cost": nearest["cost"] + dist_to_rnd
                }
            else:
                # Langkah sejauh expand_dis ke arah sampel
                theta = math.atan2(dy, dx)
                new_node = {
                    "x": nearest["x"] + self.expand_dis * math.cos(theta),
                    "y": nearest["y"] + self.expand_dis * math.sin(theta),
                    "parent": nearest,
                    "cost": nearest["cost"] + self.expand_dis
                }

            # ---------- VALIDASI NODE BARU ----------
            # 1. Posisi node harus free
            if self.check_collision(new_node["x"], new_node["y"]):
                continue

            # 2. Pastikan garis dari parent ke node baru aman
            if self.check_edge:
                if not self.is_line_safe([nearest["x"], nearest["y"]],
                                         [new_node["x"], new_node["y"]]):
                    continue

            # Tambahkan ke pohon
            self.node_list.append(new_node)

            # 3. Cek apakah goal sudah terjangkau dari node baru
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)

            # (Opsional) Jika sudah sangat dekat, coba langsung
            dist_to_goal = math.dist([new_node["x"], new_node["y"]],
                                     [self.goal["x"], self.goal["y"]])
            if dist_to_goal < self.expand_dis * 2:
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)

        # Final check: node terdekat dengan goal
        if self.node_list:
            last = min(self.node_list,
                      key=lambda n: math.dist([n["x"], n["y"]],
                                             [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(last):
                return self.extract_path(last)
        return None

    # ================== RRT* (TIDAK DIUBAH, HANYA UNTUK KOMPATIBILITAS) ==================
    def solve_rrt_star(self, max_iter=500):
        self.node_list = [self.start]
        for iteration in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            new_node = {
                "x": nearest["x"] + 0.8 * math.cos(theta),
                "y": nearest["y"] + 0.8 * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + 0.8
            }
            if not self.check_collision(new_node["x"], new_node["y"]):
                near_nodes = [n for n in self.node_list
                            if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= 2.2**2]
                for near in near_nodes:
                    d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                    if near["cost"] + d < new_node["cost"] and self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"], new_node["parent"] = near["cost"] + d, near
                self.node_list.append(new_node)
                for near in near_nodes:
                    d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"] and self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"], near["cost"] = new_node, new_node["cost"] + d
        reachable_nodes = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable_nodes:
            best = min(reachable_nodes, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    # ================== EKSTRAK PATH & SMOOTHING ==================
    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        curr = 0
        iterations = 0
        while curr < len(path) - 1 and iterations < max_smooth_iter:
            iterations += 1
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
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles

        # ---------- PARAMETER UNTUK CELAH SEMPIT ----------
        self.margin = 0.2                     # lebih kecil dari 0.45
        self.expand_dis = 0.35               # langkah tetap, cukup kecil
        self.goal_bias = 0.1                # 10% goal
        self.gaussian_bias = 0.5            # 50% Gaussian
        self.uniform_bias = 0.4             # 40% uniform
        # ------------------------------------------------

        self.max_gaussian_attempts = 300     # cukup besar
        self.node_list = []

    # ================== COLLISION DENGAN MARGIN KECIL ==================
    def is_point_in_rect(self, px, py, obs):
        tx, ty = px - obs['x'], py - obs['y']
        c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
        lx, ly = tx * c - ty * s, tx * s + ty * c
        return (- (obs['w']/2 + self.margin) <= lx <= (obs['w']/2 + self.margin) and
                - (obs['h']/2 + self.margin) <= ly <= (obs['h']/2 + self.margin))

    def check_collision(self, x, y):
        for o in self.obstacles:
            if self.is_point_in_rect(x, y, o):
                return True
        return False

    # ================== CEK GARIS DENGAN RESOLUSI TINGGI ==================
    def is_line_safe(self, p1, p2, resolution=0.04):
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / resolution))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.check_collision(x, y):
                return False
        return True

    # ================== GAUSSIAN SAMPLING EKSPLORATIF (TETAP SAFE-UNSAFE) ==================
    def get_gaussian_sample(self):
        """
        - Pilih titik FREE secara SERAGAM di seluruh arena (bukan lokal).
        - Titik kedua dari distribusi Gaussian dengan sigma bervariasi.
        - Hanya gunakan jika titik kedua di dalam obstacle.
        - Ambil titik tengah, pastikan free.
        """
        for _ in range(self.max_gaussian_attempts):
            # 1. Titik pertama: uniform random, HARUS FREE
            x1 = random.uniform(-7.5, 7.5)
            y1 = random.uniform(-7.5, 7.5)
            if self.check_collision(x1, y1):
                continue

            # 2. Sigma bervariasi: kadang lokal, kadang eksploratif
            sigma = random.uniform(0.8, 2.5)

            # 3. Titik kedua dari Gaussian
            x2 = random.gauss(x1, sigma)
            y2 = random.gauss(y1, sigma)
            x2 = max(-7.5, min(7.5, x2))
            y2 = max(-7.5, min(7.5, y2))

            # 4. Syarat: titik kedua harus di DALAM obstacle
            if self.check_collision(x2, y2):
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                # 5. Titik tengah harus FREE
                if not self.check_collision(mx, my):
                    return [mx, my]

        # Fallback: uniform random (jika gagal terus)
        return [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

    # ================== CEK APAKAH GOAL TERJANGKAU ==================
    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]])
        return (dist <= self.expand_dis * 1.2 and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    # ================== MULTI-BIAS DENGAN STEERING YANG BENAR ==================
    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]

        for i in range(max_iter):
            # ---------- SAMPLING (3 BIAS MURNI) ----------
            p = random.random()
            if p < self.goal_bias:
                rnd = [self.goal["x"], self.goal["y"]]
            elif p < self.goal_bias + self.gaussian_bias:
                rnd = self.get_gaussian_sample()
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

            # ---------- NEAREST NODE ----------
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)

            # ---------- STEERING (STANDAR RRT) ----------
            dist_to_rnd = math.dist([nearest["x"], nearest["y"]], rnd)
            if dist_to_rnd < self.expand_dis:
                # Sampel lebih dekat dari langkah maksimum -> langsung ke sampel
                new_node = {
                    "x": rnd[0],
                    "y": rnd[1],
                    "parent": nearest,
                    "cost": nearest["cost"] + dist_to_rnd
                }
            else:
                # Langkah sejauh expand_dis ke arah sampel
                theta = math.atan2(rnd[1] - nearest["y"], rnd[0] - nearest["x"])
                new_node = {
                    "x": nearest["x"] + self.expand_dis * math.cos(theta),
                    "y": nearest["y"] + self.expand_dis * math.sin(theta),
                    "parent": nearest,
                    "cost": nearest["cost"] + self.expand_dis
                }

            # ---------- VALIDASI NODE BARU ----------
            # 1. Posisi harus free
            if self.check_collision(new_node["x"], new_node["y"]):
                continue

            # 2. Pastikan garis dari parent ke node baru aman
            #    (ini opsional, tapi sangat direkomendasikan untuk pohon yang valid)
            if not self.is_line_safe([nearest["x"], nearest["y"]],
                                     [new_node["x"], new_node["y"]]):
                continue

            # Tambahkan ke pohon
            self.node_list.append(new_node)

            # 3. Cek apakah goal sudah terjangkau
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)

            # Heuristik: jika sudah sangat dekat dengan goal, coba langsung
            dist_to_goal = math.dist([new_node["x"], new_node["y"]],
                                     [self.goal["x"], self.goal["y"]])
            if dist_to_goal < self.expand_dis * 2:
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)

        # Final check: node terdekat dengan goal
        if self.node_list:
            last = min(self.node_list,
                      key=lambda n: math.dist([n["x"], n["y"]],
                                             [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(last):
                return self.extract_path(last)
        return None

    # ================== RRT* (TIDAK DIUBAH, UNTUK LENGKAP) ==================
    def solve_rrt_star(self, max_iter=500):
        self.node_list = [self.start]
        for iteration in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            new_node = {
                "x": nearest["x"] + 0.8 * math.cos(theta),
                "y": nearest["y"] + 0.8 * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + 0.8
            }
            if not self.check_collision(new_node["x"], new_node["y"]):
                near_nodes = [n for n in self.node_list
                            if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= 2.2**2]
                for near in near_nodes:
                    d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                    if near["cost"] + d < new_node["cost"] and self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"], new_node["parent"] = near["cost"] + d, near
                self.node_list.append(new_node)
                for near in near_nodes:
                    d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"] and self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"], near["cost"] = new_node, new_node["cost"] + d
        reachable_nodes = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable_nodes:
            best = min(reachable_nodes, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    # ================== EKSTRAK PATH & SMOOTHING ==================
    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        curr = 0
        iterations = 0
        while curr < len(path) - 1 and iterations < max_smooth_iter:
            iterations += 1
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
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles

        # ---------- PARAMETER UNTUK CELAH SEMPIT ----------
        self.margin = 0.2                     # lebih kecil dari 0.45
        self.expand_dis = 0.35               # langkah tetap, cukup kecil
        self.search_radius = 2.0             # untuk RRT* (tidak dipakai di Multi‑Bias)
        self.goal_bias = 0.1                # 10% goal
        self.gaussian_bias = 0.5            # 50% Gaussian
        self.uniform_bias = 0.4             # 40% uniform
        # ------------------------------------------------

        # Gaussian sampling yang efektif
        self.max_gaussian_attempts = 150     # cukup besar
        self.gaussian_sigma = 0.9            # kecil agar sampel dekat obstacle
        # Opsi: cek garis lurus parent → new node (sangat direkomendasikan)
        self.check_edge = True              # set False jika ingin persis seperti asli

        self.node_list = []

    # ================== COLLISION DENGAN MARGIN KECIL ==================
    def is_point_in_rect(self, px, py, obs):
        tx, ty = px - obs['x'], py - obs['y']
        c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
        lx, ly = tx * c - ty * s, tx * s + ty * c
        return (- (obs['w']/2 + self.margin) <= lx <= (obs['w']/2 + self.margin) and
                - (obs['h']/2 + self.margin) <= ly <= (obs['h']/2 + self.margin))

    def check_collision(self, x, y):
        for o in self.obstacles:
            if self.is_point_in_rect(x, y, o):
                return True
        return False

    # ================== CEK GARIS DENGAN RESOLUSI TINGGI ==================
    def is_line_safe(self, p1, p2, resolution=0.04):
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / resolution))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.check_collision(x, y):
                return False
        return True

    # ================== GAUSSIAN SAMPLING (HANYA SAFE-UNSAFE) ==================
    def get_gaussian_sample(self):
        for _ in range(self.max_gaussian_attempts):
            # Cari titik free
            free_found = False
            for _ in range(10):
                x1 = random.uniform(-7.5, 7.5)
                y1 = random.uniform(-7.5, 7.5)
                if not self.check_collision(x1, y1):
                    free_found = True
                    break
            if not free_found:
                continue

            # Titik kedua dari Gaussian di sekitar (x1,y1) dengan sigma kecil
            x2 = random.gauss(x1, self.gaussian_sigma)
            y2 = random.gauss(y1, self.gaussian_sigma)
            x2 = max(-7.5, min(7.5, x2))
            y2 = max(-7.5, min(7.5, y2))

            # Jika titik kedua di dalam obstacle, ambil titik tengah
            if self.check_collision(x2, y2):
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                if not self.check_collision(mx, my):
                    return [mx, my]

        # Fallback uniform
        return [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

    # ================== CEK JANGKAUAN KE GOAL ==================
    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]])
        return (dist <= self.expand_dis * 1.2 and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    # ================== MULTI-BIAS DENGAN STEERING YANG BENAR ==================
    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]

        for i in range(max_iter):
            # Sampling dengan 3 bias murni
            p = random.random()
            if p < self.goal_bias:
                rnd = [self.goal["x"], self.goal["y"]]
            elif p < self.goal_bias + self.gaussian_bias:
                rnd = self.get_gaussian_sample()
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

            # Node terdekat
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)

            # ---- STEERING YANG SESUAI STANDAR RRT ----
            dist_to_rnd = math.dist([nearest["x"], nearest["y"]], rnd)
            if dist_to_rnd < self.expand_dis:
                # Jika sampel lebih dekat dari langkah, langsung ambil sampel
                new_node = {
                    "x": rnd[0],
                    "y": rnd[1],
                    "parent": nearest,
                    "cost": nearest["cost"] + dist_to_rnd
                }
            else:
                # Langkah sejauh expand_dis ke arah sampel
                theta = math.atan2(rnd[1] - nearest["y"], rnd[0] - nearest["x"])
                new_node = {
                    "x": nearest["x"] + self.expand_dis * math.cos(theta),
                    "y": nearest["y"] + self.expand_dis * math.sin(theta),
                    "parent": nearest,
                    "cost": nearest["cost"] + self.expand_dis
                }

            # ---- VALIDASI NODE BARU ----
            # 1. Posisi node harus free
            if self.check_collision(new_node["x"], new_node["y"]):
                continue

            # 2. (Opsional) Pastikan garis dari parent ke node baru aman
            if self.check_edge:
                if not self.is_line_safe([nearest["x"], nearest["y"]],
                                         [new_node["x"], new_node["y"]]):
                    continue

            # Tambahkan ke pohon
            self.node_list.append(new_node)

            # Cek apakah goal sudah terjangkau dari node baru
            if self.is_goal_reachable(new_node):
                return self.extract_path(new_node)

            # (Opsional) heuristik: jika sudah dekat goal, coba langsung
            dist_to_goal = math.dist([new_node["x"], new_node["y"]],
                                     [self.goal["x"], self.goal["y"]])
            if dist_to_goal < self.expand_dis * 2:
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)

        # Final check: node terdekat dengan goal
        if self.node_list:
            last = min(self.node_list,
                      key=lambda n: math.dist([n["x"], n["y"]],
                                             [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(last):
                return self.extract_path(last)
        return None

    # ================== RRT* (TIDAK DIUBAH, HANYA UNTUK LENGKAP) ==================
    def solve_rrt_star(self, max_iter=500):
        self.node_list = [self.start]
        for iteration in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            new_node = {
                "x": nearest["x"] + 0.8 * math.cos(theta),
                "y": nearest["y"] + 0.8 * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + 0.8
            }
            if not self.check_collision(new_node["x"], new_node["y"]):
                near_nodes = [n for n in self.node_list
                            if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= 2.2**2]
                for near in near_nodes:
                    d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                    if near["cost"] + d < new_node["cost"] and self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"], new_node["parent"] = near["cost"] + d, near
                self.node_list.append(new_node)
                for near in near_nodes:
                    d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"] and self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"], near["cost"] = new_node, new_node["cost"] + d
        reachable_nodes = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable_nodes:
            best = min(reachable_nodes, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    # ================== EKSTRAK PATH & SMOOTHING ==================
    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        curr = 0
        iterations = 0
        while curr < len(path) - 1 and iterations < max_smooth_iter:
            iterations += 1
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
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles

        # ---------- PARAMETER YANG DISESUAIKAN UNTUK CELAH SEMPIT ----------
        self.margin = 0.2                     # <-- perkecil margin (dari 0.45)
        self.expand_dis = 0.35               # <-- langkah lebih kecil (dari 0.8)
        self.search_radius = 2.0             # untuk RRT* (tidak diubah)
        self.goal_bias = 0.1                # 10% goal
        self.gaussian_bias = 0.5            # 50% Gaussian (diperbaiki)
        self.uniform_bias = 0.4             # 40% uniform
        # ----------------------------------------------------------------

        # Gaussian sampling yang agresif
        self.max_gaussian_attempts = 150     # naikkan drastis (dari 3)
        self.gaussian_sigma = 0.9            # lebih kecil agar dekat obstacle
        self.bridge_mode = False            # tidak menggunakan bridge test (hanya safe-unsafe)

        self.node_list = []
        # untuk adaptasi langkah
        self.progress_stall = 0
        self.last_best_dist = math.dist([start[0], start[1]], [goal[0], goal[1]])

    # ================== COLLISION DENGAN MARGIN KECIL ==================
    def is_point_in_rect(self, px, py, obs):
        """margin sudah diperkecil melalui self.margin"""
        tx, ty = px - obs['x'], py - obs['y']
        c, s = math.cos(-obs['rot']), math.sin(-obs['rot'])
        lx, ly = tx * c - ty * s, tx * s + ty * c
        return (- (obs['w']/2 + self.margin) <= lx <= (obs['w']/2 + self.margin) and
                - (obs['h']/2 + self.margin) <= ly <= (obs['h']/2 + self.margin))

    def check_collision(self, x, y):
        for o in self.obstacles:
            if self.is_point_in_rect(x, y, o):
                return True
        return False

    # ================== CEK GARIS DENGAN RESOLUSI TINGGI ==================
    def is_line_safe(self, p1, p2, resolution=0.04):   # 4 cm, lebih rapat
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / resolution))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.check_collision(x, y):
                return False
        return True

    # ================== GAUSSIAN SAMPLING YANG SANGAT EFEKTIF ==================
    def get_gaussian_sample(self):
        """
        Mencari pasangan titik (free, obstacle) dengan Gaussian di sekitar titik free.
        Dilakukan berkali-kali sampai mendapatkan titik tengah yang benar-benar free.
        Hanya menggunakan konsep safe-unsafe, tidak ada bias tambahan.
        """
        for _ in range(self.max_gaussian_attempts):
            # 1. Dapatkan titik FREE dengan mencoba maksimal 10 kali
            free_found = False
            for _ in range(10):
                x1 = random.uniform(-7.5, 7.5)
                y1 = random.uniform(-7.5, 7.5)
                if not self.check_collision(x1, y1):
                    free_found = True
                    break
            if not free_found:
                continue

            # 2. Titik kedua dari distribusi Gaussian di sekitar (x1,y1)
            #    sigma kecil agar cenderung dekat dengan titik pertama
            sigma = self.gaussian_sigma
            x2 = random.gauss(x1, sigma)
            y2 = random.gauss(y1, sigma)
            # jepit dalam batas arena
            x2 = max(-7.5, min(7.5, x2))
            y2 = max(-7.5, min(7.5, y2))

            # 3. Syarat: titik kedua harus berada di DALAM obstacle
            if self.check_collision(x2, y2):
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                # 4. Validasi titik tengah harus FREE
                if not self.check_collision(mx, my):
                    return [mx, my]

        # Fallback: jika gagal, kembalikan uniform random
        return [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

    # ================== ADAPTASI LANGKAH SAAT MAMPET ==================
    def adapt_expand_dis(self):
        """Jika jarak ke goal tidak membaik dalam 20 iterasi, perkecil langkah."""
        if self.progress_stall > 20:
            self.expand_dis = max(0.2, 0.35 * 0.7)   # turunkan jadi 0.25
        else:
            self.expand_dis = 0.35                  # kembalikan ke normal
        # (nilai hardcode disesuaikan, bisa di-tuning)

    # ================== CEK APAKAH GOAL SUDAH TERJANGKAU ==================
    def is_goal_reachable(self, node):
        dist = math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]])
        return (dist <= self.expand_dis * 1.2 and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    # ================== MULTI-BIAS (HANYA 3 BIAS: GOAL, GAUSSIAN, UNIFORM) ==================
    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]
        self.progress_stall = 0
        self.last_best_dist = math.dist([self.start["x"], self.start["y"]],
                                        [self.goal["x"], self.goal["y"]])

        for i in range(max_iter):
            # Adaptasi langkah dinamis
            self.adapt_expand_dis()

            # Evaluasi node terbaik saat ini
            nearest_to_goal = min(self.node_list,
                                 key=lambda n: math.dist([n["x"], n["y"]],
                                                        [self.goal["x"], self.goal["y"]]))
            current_best = math.dist([nearest_to_goal["x"], nearest_to_goal["y"]],
                                     [self.goal["x"], self.goal["y"]])

            # Update progress stall
            if current_best < self.last_best_dist - 0.1:
                self.last_best_dist = current_best
                self.progress_stall = 0
            else:
                self.progress_stall += 1

            # Early termination jika goal sudah terjangkau
            if self.is_goal_reachable(nearest_to_goal):
                return self.extract_path(nearest_to_goal)

            # ---------- SAMPLING DENGAN 3 BIAS MURNI ----------
            p = random.random()
            if p < self.goal_bias:                     # Goal bias 10%
                rnd = [self.goal["x"], self.goal["y"]]
            elif p < self.goal_bias + self.gaussian_bias:  # Gaussian bias 50%
                rnd = self.get_gaussian_sample()
            else:                                      # Uniform bias 40%
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

            # ---------- EKSPANSI ----------
            nearest = min(self.node_list,
                         key=lambda n: (n["x"] - rnd[0])**2 + (n["y"] - rnd[1])**2)

            dist_to_rnd = math.dist([nearest["x"], nearest["y"]], rnd)
            if dist_to_rnd < 0.1:
                continue

            theta = math.atan2(rnd[1] - nearest["y"], rnd[0] - nearest["x"])
            step = min(self.expand_dis, dist_to_rnd)

            new_node = {
                "x": nearest["x"] + step * math.cos(theta),
                "y": nearest["y"] + step * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + step
            }

            if not self.check_collision(new_node["x"], new_node["y"]):
                self.node_list.append(new_node)
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)

        # Final check: coba node terdekat dengan goal
        if self.node_list:
            last = min(self.node_list,
                      key=lambda n: math.dist([n["x"], n["y"]],
                                             [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(last):
                return self.extract_path(last)
        return None

    # ================== RRT* (TIDAK DIUBAH, HANYA UNTUK LENGKAP) ==================
    def solve_rrt_star(self, max_iter=500):
        """Kode asli, tidak diubah – hanya disertakan agar tidak error."""
        self.node_list = [self.start]
        for iteration in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            new_node = {
                "x": nearest["x"] + 0.8 * math.cos(theta),
                "y": nearest["y"] + 0.8 * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + 0.8
            }
            if not self.check_collision(new_node["x"], new_node["y"]):
                near_nodes = [n for n in self.node_list
                            if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= 2.2**2]
                for near in near_nodes:
                    d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                    if near["cost"] + d < new_node["cost"] and self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"], new_node["parent"] = near["cost"] + d, near
                self.node_list.append(new_node)
                for near in near_nodes:
                    d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"] and self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"], near["cost"] = new_node, new_node["cost"] + d
        reachable_nodes = [n for n in self.node_list if self.is_goal_reachable(n)]
        if reachable_nodes:
            best = min(reachable_nodes, key=lambda n: n["cost"])
            return self.extract_path(best)
        return None

    # ================== EKSTRAK PATH & SMOOTHING ==================
    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        curr = 0
        iterations = 0
        while curr < len(path) - 1 and iterations < max_smooth_iter:
            iterations += 1
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
    
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles
        self.node_list = []
        # Parameter adaptif – akan diubah dinamis jika perlu
        self.base_expand_dis = 0.8
        self.expand_dis = self.base_expand_dis
        self.search_radius = 2.2
        # Sampling parameters
        self.max_sampling_attempts = 80          # ↑ lebih banyak kesempatan
        self.bridge_radius = 1.2                # radius untuk bridge test
        self.use_bridge_prob = 0.6              # probabilitas bridge test vs safe-unsafe
        self.progress_stall_counter = 0
        self.last_best_dist = float('inf')

    def check_collision(self, x, y):
        for o in self.obstacles:
            if is_point_in_rect(x, y, o):
                return True
        return False

    def is_line_safe(self, p1, p2):
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / 0.3))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t*(p2[0]-p1[0])
            y = p1[1] + t*(p2[1]-p1[1])
            if self.check_collision(x, y):
                return False
        return True

    # ================== PERBAIKAN UTAMA ==================
    def get_gaussian_sample(self):
        """
        Improved Gaussian / narrow passage sampling.
        Menggabungkan dua strategi:
        1. Bridge test : dua titik di dalam obstacle, titik tengah di free space → kandidat celah.
        2. Boundary test : satu titik free, satu titik obstacle → titik tengah (validasi free).
        Jika gagal setelah max attempts → uniform random fallback.
        """
        for _ in range(self.max_sampling_attempts):
            # Pilih strategi berdasarkan probabilitas
            if random.random() < self.use_bridge_prob:
                # --- Bridge test (dua titik di dalam obstacle) ---
                p1 = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
                if not self.check_collision(*p1):
                    continue
                # cari titik kedua dalam radius tertentu dari p1
                p2 = [
                    p1[0] + random.uniform(-self.bridge_radius, self.bridge_radius),
                    p1[1] + random.uniform(-self.bridge_radius, self.bridge_radius)
                ]
                p2[0] = max(-7.5, min(7.5, p2[0]))
                p2[1] = max(-7.5, min(7.5, p2[1]))
                if not self.check_collision(*p2):
                    continue
                # titik tengah
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                if not self.check_collision(mx, my):
                    # Validasi tambahan: pastikan titik tengah tidak terlalu dekat dengan obstacle lain?
                    # Opsional: cek jarak ke obstacle terdekat > threshold kecil
                    return [mx, my]
            else:
                # --- Boundary test (satu free, satu obstacle) ---
                # Sample titik pertama (free)
                for _ in range(10):  # coba dapatkan titik free
                    x1 = random.uniform(-7.5, 7.5)
                    y1 = random.uniform(-7.5, 7.5)
                    if not self.check_collision(x1, y1):
                        break
                else:
                    continue
                # Titik kedua dari distribusi Gaussian di sekitar titik pertama
                sigma = 1.8
                x2 = max(-7.5, min(7.5, random.gauss(x1, sigma)))
                y2 = max(-7.5, min(7.5, random.gauss(y1, sigma)))
                if self.check_collision(x2, y2):   # pastikan titik kedua di obstacle
                    mx = (x1 + x2) / 2
                    my = (y1 + y2) / 2
                    if not self.check_collision(mx, my):
                        return [mx, my]
        # Fallback: uniform random
        return [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

    # ================== ADAPTIVE STEP SIZE ==================
    def adapt_expand_dis(self):
        """Kurangi langkah jika eksplorasi macet di dekat celah."""
        # Sederhana: jika progress terhenti, perkecil expand_dis
        if self.progress_stall_counter > 20:
            self.expand_dis = max(0.3, self.base_expand_dis * 0.6)
        else:
            self.expand_dis = self.base_expand_dis

    # ================== GOAL REACHABLE CHECK ==================
    def is_goal_reachable(self, node):
        return (math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]) <= self.expand_dis and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    # ================== MULTI‑BIAS DENGAN SAMPLING CERDAS ==================
    def solve_multibias(self, max_iter=500):
        self.node_list = [self.start]
        self.progress_stall_counter = 0
        self.last_best_dist = math.dist([self.start["x"], self.start["y"]],
                                        [self.goal["x"], self.goal["y"]])

        for i in range(max_iter):
            # Adaptasi step size
            self.adapt_expand_dis()

            # Early stopping
            nearest_to_goal = min(self.node_list,
                                 key=lambda n: math.dist([n["x"], n["y"]],
                                                        [self.goal["x"], self.goal["y"]]))
            current_best_dist = math.dist([nearest_to_goal["x"], nearest_to_goal["y"]],
                                          [self.goal["x"], self.goal["y"]])

            # Update progress stall counter
            if current_best_dist < self.last_best_dist - 0.1:
                self.last_best_dist = current_best_dist
                self.progress_stall_counter = 0
            else:
                self.progress_stall_counter += 1

            if self.is_goal_reachable(nearest_to_goal):
                return self.extract_path(nearest_to_goal)

            p = random.random()
            # 1. Goal bias (10%)
            if p < 0.1:
                rnd = [self.goal["x"], self.goal["y"]]
            # 2. Improved Gaussian / narrow bias (70% – dinaikkan!)
            elif p < 0.8:
                rnd = self.get_gaussian_sample()
            # 3. Uniform (20%)
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

            # Nearest node
            nearest = min(self.node_list,
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)

            dist_to_rnd = math.dist([nearest["x"], nearest["y"]], rnd)
            if dist_to_rnd < 0.1:
                continue

            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            step = min(self.expand_dis, dist_to_rnd)

            new_node = {
                "x": nearest["x"] + step * math.cos(theta),
                "y": nearest["y"] + step * math.sin(theta),
                "parent": nearest,
                "cost": nearest["cost"] + step
            }

            if not self.check_collision(new_node["x"], new_node["y"]):
                self.node_list.append(new_node)
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)

        # Final check
        if self.node_list:
            last = min(self.node_list,
                      key=lambda n: math.dist([n["x"], n["y"]],
                                             [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(last):
                return self.extract_path(last)
        return None

    # ================== RRT* (TIDAK DIUBAH, TAPI TETAP DISERTAKAN) ==================
    def solve_rrt_star(self, max_iter=500):
        # ... (kode asli, tidak diubah) ...
        pass

    def extract_path(self, node):
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        # ... (kode asli, tidak diubah) ...
        pass
    def __init__(self, start, goal, obstacles):
        self.start = {"x": start[0], "y": start[1], "parent": None, "cost": 0.0}
        self.goal = {"x": goal[0], "y": goal[1]}
        self.obstacles = obstacles
        self.node_list = []
        self.expand_dis = 0.8
        self.search_radius = 2.2
        self.max_gaussian_attempts = 3

    def check_collision(self, x, y):
        """Mengecek kolisi pada titik tunggal."""
        for o in self.obstacles:
            if is_point_in_rect(x, y, o): return True
        return False

    def is_line_safe(self, p1, p2):
        """Mengecek kolisi di sepanjang garis lintasan."""
        dist = math.dist(p1, p2)
        steps = max(2, int(dist / 0.3))
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t*(p2[0]-p1[0])
            y = p1[1] + t*(p2[1]-p1[1])
            if self.check_collision(x, y):
                return False
        return True

    def get_gaussian_sample(self, sigma=1.8):
        """Algoritma Gaussian Sampling."""
        for _ in range(self.max_gaussian_attempts): 
            x1, y1 = random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)
            safe1 = not self.check_collision(x1, y1)

            sigma_clamped = min(sigma, 3.0)
            x2 = max(-7.5, min(7.5, random.gauss(x1, sigma_clamped)))
            y2 = max(-7.5, min(7.5, random.gauss(y1, sigma_clamped)))
            safe2 = not self.check_collision(x2, y2)

            if safe1 != safe2:
                return [(x1 + x2) / 2, (y1 + y2) / 2]
        
        return [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]

    def is_goal_reachable(self, node):
        """Cek apakah goal bisa dicapai dari node ini."""
        return (math.dist([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]) <= self.expand_dis and
                self.is_line_safe([node["x"], node["y"]], [self.goal["x"], self.goal["y"]]))

    def solve_multibias(self, max_iter=500):
        """
        Implementasi Multi-Bias yang lebih efisien.
        Kembalikan None jika tidak sampai goal.
        """
        self.node_list = [self.start]
        
        for i in range(max_iter):
            # Early stopping jika sudah dekat goal
            nearest_to_goal = min(self.node_list, 
                                 key=lambda n: math.dist([n["x"], n["y"]], 
                                                        [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(nearest_to_goal):
                return self.extract_path(nearest_to_goal)
            
            p = random.random()
            
            # 1. Goal Bias (10%)
            if p < 0.1:
                rnd = [self.goal["x"], self.goal["y"]]
            # 2. Gaussian Bias (50%)
            elif p < 0.6:
                rnd = self.get_gaussian_sample()
            # 3. Uniform (40%)
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
            
            # Find nearest node
            nearest = min(self.node_list, 
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            
            dist_to_rnd = math.dist([nearest["x"], nearest["y"]], rnd)
            if dist_to_rnd < 0.1:
                continue
                
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            
            new_node = {
                "x": nearest["x"] + min(self.expand_dis, dist_to_rnd) * math.cos(theta), 
                "y": nearest["y"] + min(self.expand_dis, dist_to_rnd) * math.sin(theta), 
                "parent": nearest, 
                "cost": nearest["cost"] + min(self.expand_dis, dist_to_rnd)
            }

            if not self.check_collision(new_node["x"], new_node["y"]):
                self.node_list.append(new_node)
                # Cek jika goal bisa dicapai
                if self.is_goal_reachable(new_node):
                    return self.extract_path(new_node)
        
        # Cek lagi node terdekat di akhir
        if self.node_list:
            last = min(self.node_list, 
                      key=lambda n: math.dist([n["x"], n["y"]], 
                                             [self.goal["x"], self.goal["y"]]))
            if self.is_goal_reachable(last):
                return self.extract_path(last)
        return None

    def solve_rrt_star(self, max_iter=500):
        """Algoritma RRT* yang dioptimalkan."""
        self.node_list = [self.start]
        
        for iteration in range(max_iter):
            if random.random() < 0.05:
                rnd = [self.goal["x"], self.goal["y"]]
            else:
                rnd = [random.uniform(-7.5, 7.5), random.uniform(-7.5, 7.5)]
            
            nearest = min(self.node_list, 
                         key=lambda n: (n["x"]-rnd[0])**2 + (n["y"]-rnd[1])**2)
            
            theta = math.atan2(rnd[1]-nearest["y"], rnd[0]-nearest["x"])
            new_node = {
                "x": nearest["x"] + self.expand_dis * math.cos(theta), 
                "y": nearest["y"] + self.expand_dis * math.sin(theta), 
                "parent": nearest, 
                "cost": nearest["cost"] + self.expand_dis
            }

            if not self.check_collision(new_node["x"], new_node["y"]):
                near_nodes = [n for n in self.node_list 
                            if (n["x"]-new_node["x"])**2 + (n["y"]-new_node["y"])**2 <= self.search_radius**2]
                
                for near in near_nodes:
                    d = math.dist([near["x"], near["y"]], [new_node["x"], new_node["y"]])
                    if near["cost"] + d < new_node["cost"] and self.is_line_safe([near["x"], near["y"]], [new_node["x"], new_node["y"]]):
                        new_node["cost"], new_node["parent"] = near["cost"] + d, near
                
                self.node_list.append(new_node)
                
                for near in near_nodes:
                    d = math.dist([new_node["x"], new_node["y"]], [near["x"], near["y"]])
                    if new_node["cost"] + d < near["cost"] and self.is_line_safe([new_node["x"], new_node["y"]], [near["x"], near["y"]]):
                        near["parent"], near["cost"] = new_node, new_node["cost"] + d
        
        # Cari node terdekat yang bisa mencapai goal
        if self.node_list:
            reachable_nodes = [n for n in self.node_list if self.is_goal_reachable(n)]
            if reachable_nodes:
                best = min(reachable_nodes, key=lambda n: n["cost"])
                return self.extract_path(best)
        return None

    def extract_path(self, node):
        """Ekstrak path dari node ke start."""
        path = [[self.goal["x"], self.goal["y"]]]
        curr = node
        while curr:
            path.append([curr["x"], curr["y"]])
            curr = curr["parent"]
        return path[::-1]

    def smooth_path(self, path, max_smooth_iter=20):
        """Menyederhanakan path dengan batasan iterasi."""
        if not path or len(path) < 3:
            return path
            
        smoothed = [path[0]]
        curr = 0
        iterations = 0
        
        while curr < len(path) - 1 and iterations < max_smooth_iter:
            iterations += 1
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

# --- GUI Class ---
class DroneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Bias RRT Gaussian Dashboard")
        self.sock = None
        self.data = {"start": None, "goal": None, "obs": [], "path": []}
        self.running = False
        self.raw_path = None  # Untuk menyimpan path sebelum smoothing

        # UI Layout
        side = tk.Frame(root, width=250, bg="#2c3e50")
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)

        tk.Label(side, text="ALGORITHM", bg="#2c3e50", fg="white", font=("Arial", 10, "bold")).pack(pady=15)
        self.algo_var = tk.StringVar(value="multibias")
        tk.Radiobutton(side, text="Pure RRT*", variable=self.algo_var, value="rrtstar", 
                      bg="#2c3e50", fg="white", selectcolor="#34495e").pack(anchor="w", padx=20)
        tk.Radiobutton(side, text="Multi-Bias (Gaussian)", variable=self.algo_var, value="multibias", 
                      bg="#2c3e50", fg="white", selectcolor="#34495e").pack(anchor="w", padx=20)

        # Frame untuk iterasi settings
        iter_frame = tk.Frame(side, bg="#2c3e50")
        iter_frame.pack(fill='x', padx=20, pady=15)
        tk.Label(iter_frame, text="MAX ITERATIONS", bg="#2c3e50", fg="white", 
                font=("Arial", 9, "bold")).pack(anchor="w")
        
        self.iter_frame_content = tk.Frame(iter_frame, bg="#2c3e50")
        self.iter_frame_content.pack(fill='x', pady=5)
        
        self.iter_var = tk.StringVar(value="500")
        self.iter_entry = tk.Entry(self.iter_frame_content, textvariable=self.iter_var, 
                                  width=10, bg="white", fg="black", justify="center")
        self.iter_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_iter_apply = tk.Button(self.iter_frame_content, text="Apply", 
                                       command=self.validate_iterations,
                                       bg="#3498db", fg="white", font=("Arial", 8),
                                       width=6)
        self.btn_iter_apply.pack(side=tk.LEFT)

        # Control buttons
        self.btn_map = ttk.Button(side, text="1. LOAD MAP", command=self.get_map)
        self.btn_map.pack(fill='x', padx=20, pady=10)
        self.btn_run = ttk.Button(side, text="2. RUN PLANNING", command=self.start_thread, state="disabled")
        self.btn_run.pack(fill='x', padx=20, pady=5)
        self.btn_fly = ttk.Button(side, text="3. EXECUTE", command=self.fly, state="disabled")
        self.btn_fly.pack(fill='x', padx=20, pady=5)
        
        self.btn_reset = tk.Button(side, text="⚠ RESET DRONE", command=self.reset_sim, 
                                  bg="#e74c3c", fg="white", font=("Arial", 9, "bold"))
        self.btn_reset.pack(fill='x', padx=20, pady=15)

        # Metrics display
        self.lbl_metrics = tk.Label(side, text="Status: Ready\nTime: -\nCost: -\nNodes: -", 
                                   bg="#2c3e50", fg="#bdc3c7", justify=tk.LEFT)
        self.lbl_metrics.pack(pady=20)

        # Progress indicator
        self.progress_var = tk.StringVar(value="")
        self.lbl_progress = tk.Label(side, textvariable=self.progress_var, 
                                    bg="#2c3e50", fg="#f39c12")
        self.lbl_progress.pack()

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.connect_socket()

    def validate_iterations(self):
        """Validasi input iterasi."""
        try:
            val = int(self.iter_var.get())
            if val < 50:
                self.iter_var.set("50")
                messagebox.showwarning("Warning", "Minimum iterations is 50")
            elif val > 50000:
                self.iter_var.set("5000")
                messagebox.showwarning("Warning", "Maximum iterations is 5000")
        except ValueError:
            self.iter_var.set("500")
            messagebox.showerror("Error", "Please enter a valid number")

    def connect_socket(self):
        """Koneksi ke Webots supervisor."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2)
            self.sock.connect(('127.0.0.1', 65432))
            self.lbl_metrics.config(text="Status: Connected\nTime: -\nCost: -\nNodes: -")
        except Exception as e:
            self.sock = None
            self.root.after(1000, self.connect_socket)

    def get_map(self):
        """Mengambil peta dari Webots."""
        if not self.sock:
            messagebox.showerror("Error", "Not connected to Webots!")
            return
            
        try:
            self.sock.sendall(json.dumps({"command": "GET_MAP"}).encode())
            data = self.sock.recv(16384)
            if data:
                r = json.loads(data.decode())
                self.data.update({"start": r['start'], "goal": r['goal'], "obs": r['obstacles']})
                self.draw_world()
                self.btn_run.config(state="normal")
                self.lbl_metrics.config(text="Status: Map loaded\nTime: -\nCost: -\nNodes: -")
            else:
                messagebox.showerror("Error", "No data received from Webots!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get map: {str(e)}")

    def reset_sim(self):
        """Reset drone ke posisi awal."""
        if not self.sock:
            return
            
        try:
            self.sock.sendall(json.dumps({"command": "RESET"}).encode())
            self.data['path'] = []
            self.raw_path = None
            self.draw_world()
            self.btn_fly.config(state="disabled")
            self.lbl_metrics.config(text="Status: Reset OK\nTime: -\nCost: -\nNodes: -")
        except:
            pass

    def draw_world(self, draw_tree=True):
        """Menggambar peta dan obstacle."""
        self.ax.clear()
        
        # Gambar obstacles
        for o in self.data['obs']: 
            self.ax.add_patch(patches.Polygon(get_corners(o), color='#2c3e50', alpha=0.8))
        
        # Gambar start dan goal
        if self.data['start']:
            self.ax.plot(*self.data['start'], 'go', markersize=8, label='Start')
        if self.data['goal']:
            self.ax.plot(*self.data['goal'], 'r*', markersize=12, label='Goal')
        
        # Gambar raw path (hanya untuk Multi-Bias, sebelum smoothing)
        if self.raw_path and len(self.raw_path) > 1 and self.algo_var.get() == "multibias":
            raw_np = np.array(self.raw_path)
            self.ax.plot(raw_np[:,0], raw_np[:,1], 'b--', linewidth=1.5, alpha=0.7, label='Raw Path')
        
        # Gambar smoothed/final path
        if self.data['path'] and len(self.data['path']) > 1:
            p_np = np.array(self.data['path'])
            if self.algo_var.get() == "multibias":
                self.ax.plot(p_np[:,0], p_np[:,1], 'g-', linewidth=2.5, label='Smoothed Path')
            else:
                self.ax.plot(p_np[:,0], p_np[:,1], '#9b59b6', linewidth=2.5, label='RRT* Path')
            
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Drone Path Planning")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def start_thread(self):
        """Mulai planning di thread terpisah."""
        if self.running:
            return
            
        self.running = True
        self.btn_run.config(state="disabled")
        self.root.config(cursor="watch")
        self.progress_var.set("Planning...")
        
        # Validasi iterasi
        self.validate_iterations()
        max_iter = int(self.iter_var.get())
        
        threading.Thread(target=self.solve, args=(max_iter,), daemon=True).start()

    def solve(self, max_iter):
        """Fungsi planning utama."""
        t0 = time.time()
        
        try:
            # Cek data yang diperlukan
            if not self.data['start'] or not self.data['goal']:
                self.root.after(0, self.planning_done, None, None, [], 0, 0, "Error: No start/goal", False)
                return
                
            eng = PlanningEngine(self.data['start'], self.data['goal'], self.data['obs'])
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(f"Running ({max_iter} iter)..."))
            
            success = False
            if self.algo_var.get() == "rrtstar":
                path = eng.solve_rrt_star(max_iter=max_iter)
                self.raw_path = path  # Untuk RRT*, raw_path sama dengan final path
                algo_name = "RRT*"
                success = path is not None
            else:
                raw_path = eng.solve_multibias(max_iter=max_iter)
                self.raw_path = raw_path  # Simpan path sebelum smoothing
                if raw_path:
                    path = eng.smooth_path(raw_path)
                    success = path is not None
                else:
                    path = None
                algo_name = "Multi-Bias"
            
            dt = time.time() - t0
            
            # Siapkan tree untuk visualisasi
            tree = []
            if eng.node_list:
                for n in eng.node_list:
                    if n["parent"]:
                        tree.append([(n["x"], n["y"]), (n["parent"]["x"], n["parent"]["y"])])
                
                # Batasi jumlah garis tree untuk performa
                if len(tree) > 1000:
                    tree = tree[::2]
            
            self.root.after(0, self.planning_done, path, tree, dt, len(eng.node_list), algo_name, success)
            
        except Exception as e:
            self.root.after(0, self.planning_done, None, [], 0, 0, f"Error: {str(e)}", False)

    def planning_done(self, path, tree, dt, nodes, algo_name, success):
        """Callback setelah planning selesai."""
        self.running = False
        self.root.config(cursor="")
        self.btn_run.config(state="normal")
        self.progress_var.set("")
        
        # Reset path data jika gagal
        if not success:
            self.data['path'] = []
            self.lbl_metrics.config(text=f"Status: Gagal menemukan Path\nTime: {dt:.3f}s\nNodes: {nodes}")
            messagebox.showwarning("Planning Failed", f"{algo_name}: Gagal menemukan Path!")
            
            # Gambar ulang dunia tanpa path
            self.draw_world(draw_tree=False)
            
            # Gambar tree eksplorasi saja
            if tree:
                self.ax.add_collection(LineCollection(tree, colors='#2ecc71', 
                                                     linewidths=0.3, alpha=0.2))
                self.canvas.draw_idle()
            return

        # Jika sukses, simpan path
        self.data['path'] = path
        
        # Gambar ulang dunia dengan path
        self.draw_world()
        
        # Tambahkan tree ke plot
        if tree:
            self.ax.add_collection(LineCollection(tree, colors='#2ecc71', 
                                                 linewidths=0.3, alpha=0.2))
        
        # Update metrics
        cost = calculate_path_cost(path)
        self.lbl_metrics.config(
            text=f"Status: {algo_name} success\n"
                 f"Time: {dt:.3f}s\n"
                 f"Cost: {cost:.2f}m\n"
                 f"Nodes: {nodes}"
        )
        
        self.canvas.draw_idle()
        self.btn_fly.config(state="normal")

    def fly(self):
        """Kirim path ke Webots untuk dieksekusi."""
        if self.data['path'] and self.sock:
            try:
                self.sock.sendall(json.dumps({
                    "command": "START_SIM", 
                    "path": self.data['path']
                }).encode())
                self.lbl_metrics.config(text="Status: Flying...\n" + 
                                              self.lbl_metrics.cget("text").split('\n', 1)[1])
            except:
                messagebox.showerror("Error", "Failed to send path to Webots!")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x700")
    root.minsize(800, 600)
    app = DroneApp(root)
    
    # Handle window close
    def on_closing():
        if app.sock:
            app.sock.close()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
