from controller import Supervisor
import socket, json, math

robot = Supervisor()
ts = int(robot.getBasicTimeStep())
drone, target = robot.getSelf(), robot.getFromDef("TARGET")
child_field = robot.getRoot().getField("children")

init_pos = drone.getPosition()
init_rot = drone.getField("rotation").getSFRotation()

def fly_to(target_p):
    curr = drone.getPosition()
    dx, dy = target_p[0]-curr[0], target_p[1]-curr[1]
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 0.15: return True
    spd = 0.15
    drone.getField("translation").setSFVec3f([curr[0]+(dx/dist)*spd, curr[1]+(dy/dist)*spd, 1.0])
    drone.getField("rotation").setSFRotation([0,0,1, math.atan2(dy, dx)])
    return False

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('127.0.0.1', 65432)); srv.listen(1); srv.setblocking(False)

conn, waypoints, wp_idx, flying = None, [], 0, False

while robot.step(ts) != -1:
    if not conn:
        try: conn, _ = srv.accept(); conn.setblocking(False)
        except: pass
    else:
        try:
            data = conn.recv(16384).decode()
            if data:
                msg = json.loads(data)
                cmd = msg.get("command")
                if cmd == "GET_MAP":
                    obs = []
                    for i in range(child_field.getCount()):
                        n = child_field.getMFNode(i)
                        if n.getDef() and "OBSTACLE" in n.getDef():
                            p, s, r = n.getPosition(), n.getField("size").getSFVec3f(), n.getField("rotation").getSFRotation()
                            obs.append({"x":p[0], "y":p[1], "w":s[0], "h":s[1], "rot": r[3] if r[2]>0 else -r[3]})
                    conn.sendall(json.dumps({"start":drone.getPosition()[:2], "goal":target.getPosition()[:2], "obstacles":obs}).encode())
                elif cmd == "START_SIM":
                    waypoints, wp_idx, flying = msg['path'], 0, True
                elif cmd == "RESET":
                    flying = False; wp_idx = 0
                    drone.getField("translation").setSFVec3f(init_pos)
                    drone.getField("rotation").setSFRotation(init_rot); drone.resetPhysics()
            else: conn.close(); conn = None
        except: pass

    if flying and wp_idx < len(waypoints):
        if fly_to(waypoints[wp_idx]): wp_idx += 1
    elif wp_idx >= len(waypoints): flying = False