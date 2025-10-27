# -*- coding: utf-8 -*-
# Creator/scripts/sdl_to_layout.py
import os, json, math, random, time, sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "..", "output")
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 小工具 ----------
def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
HINT = {"immediate":2.0, "close":5.0, "medium":12.0, "far":25.0}

def hint_to_range(h):
    if isinstance(h, list) and len(h) == 2: return float(h[0]), float(h[1])
    if isinstance(h, (int, float)): return float(h), float(h)
    if isinstance(h, str) and h in HINT: return HINT[h], HINT[h]
    return 8.0, 12.0

def forward_of(rot_deg_y: float) -> Tuple[float,float]:
    a = math.radians(rot_deg_y)
    return (-math.sin(a), -math.cos(a))  # 初始 forward=(0,-1)

def left_right_dirs(rot_deg_y: float):
    fx, fz = forward_of(rot_deg_y)
    return (-fz, fx), (fz, -fx)

def radius_from_dimensions(category:str, dim:Dict)->float:
    t = dim.get("type","height"); v = float(dim.get("value",1.8))
    base = v*0.5 if t not in ("diameter","radius") else (v*0.5 if t=="diameter" else v)
    k = {"building":0.35, "terrain":0.30, "flora":0.18, "water":0.50, "prop":0.22}.get(category,0.2)
    return max(0.4, base*k)

# ---------- SDL IO ----------
def load_sdl(path)->Dict:
    with open(path, "r", encoding="utf-8") as f:
        sdl = json.load(f)
    for o in sdl.get("objects", []):
        o.setdefault("category","prop")
        o.setdefault("dimensions", {"type":"height","value":1.8})
        o.setdefault("count", 1)
        if isinstance(o["count"], list) and len(o["count"])==2:
            o["count"] = random.randint(int(o["count"][0]), int(o["count"][1]))
    sdl.setdefault("relations", [])
    return sdl

def find(objects, oid):
    for o in objects:
        if o["id"] == oid: return o
    return None

# ---------- 核心：生成布局（你要求的完整替换版） ----------
def compile_layout(sdl:Dict)->Dict:
    # 基准：地面 80x80 米，原点在中心，Y-up
    W = H = 80.0

    # 1) 初始状态
    base = {}
    for o in sdl["objects"]:
        default_count = int(o.get("count", 1))
        name = (o.get("element_name") or "").lower()
        _id = o.get("id","").lower()
        if "grove" in _id or "grove" in name or ("林" in o.get("element_name","")):
            if "count" not in o:
                default_count = 24
        base[o["id"]] = {
            "pos2": (0.0, 0.0),
            "rotY": 0.0,
            "scale": 1.0,
            "radius": radius_from_dimensions(o["category"], o["dimensions"]),
            "count": default_count,
        }

    # 2) 主对象（第一个 building；没有则第一个）
    buildings = [o for o in sdl["objects"] if o["category"]=="building"]
    main_id = buildings[0]["id"] if buildings else sdl["objects"][0]["id"]
    base[main_id]["pos2"] = (0.0, 0.0)
    base[main_id]["rotY"] = 0.0

    # 3) 先处理“在前方”与 leads_to
    for r in sdl["relations"]:
        v, a, b = r.get("verb",""), r.get("subject"), r.get("object")
        if a not in base or b not in base: continue
        if v == "is_in_front_of":
            d,_ = hint_to_range(r.get("distance_hint","medium"))
            rot = base[b]["rotY"]
            fx, fz = forward_of(rot); bx, bz = base[b]["pos2"]
            base[a]["pos2"] = (bx + fx*d, bz + fz*d)
        if v == "leads_to":
            d,_ = hint_to_range(r.get("distance_hint","medium"))
            rot = base[b]["rotY"]
            fx, fz = forward_of(rot); bx, bz = base[b]["pos2"]
            base[a]["pos2"] = (bx + fx*d, bz + fz*d)

    # 4) 收集其它语义
    flank_pairs = []   # (a,b,d)
    surrounds = []     # (a,b,d,count)
    paths = []         # (from,to,width)
    for r in sdl["relations"]:
        v, a, b = r.get("verb",""), r.get("subject"), r.get("object")
        if a not in base or b not in base: continue
        if v == "flanks":
            d,_ = hint_to_range(r.get("distance_hint","immediate"))
            flank_pairs.append((a,b,d))
        elif v == "surrounds":
            d,_ = hint_to_range(r.get("distance_hint","near"))
            cnt_obj = find(sdl["objects"], a)
            cnt = cnt_obj.get("count", base[a]["count"] if a in base else 18)
            surrounds.append((a,b,d,int(cnt)))
        elif v == "connects":
            subj = find(sdl["objects"], a)
            w = max(1.0, radius_from_dimensions(subj["category"], subj["dimensions"])*0.8) if subj else 1.2
            tgt = r.get("target")
            if tgt and tgt in base:
                paths.append((b, tgt, w))
            else:
                paths.append((b, main_id, w))
        elif v == "leads_to":
            subj = find(sdl["objects"], a)
            w = 1.2
            if subj: w = max(1.0, radius_from_dimensions(subj["category"], subj["dimensions"])*0.8)
            paths.append((a, b, w))

    # 5) 环带散布
    instances = []
    for a,b,d,cnt in surrounds:
        ox, oz = base[b]["pos2"]
        for i in range(max(1, cnt)):
            ang = (i / max(1, cnt)) * 2*math.pi
            if -math.pi/6 < ang < math.pi/6:  # 留出前方通道
                ang += math.pi/3
            x = ox + math.cos(ang)*(d + random.uniform(-1.0, 1.0))
            z = oz + math.sin(ang)*(d + random.uniform(-1.0, 1.0))
            instances.append({
                "id": f"{a}_inst_{i+1}",
                "src": a,
                "pos2": (x,z),
                "rotY": random.uniform(-10,10),
                "scale": 1.0
            })

    # 6) 两侧对称
    for a,b,d in flank_pairs:
        ox, oz = base[b]["pos2"]; rot = base[b]["rotY"]
        left, right = left_right_dirs(rot)
        lx, lz = ox + left[0]*d,  oz + left[1]*d
        rx, rz = ox + right[0]*d, oz + right[1]*d
        instances += [
            {"id": f"{a}_L", "src": a, "pos2":(lx,lz), "rotY":rot, "scale":1.0},
            {"id": f"{a}_R", "src": a, "pos2":(rx,rz), "rotY":rot, "scale":1.0},
        ]

    # 7) 其它未实例化对象 → 各放一个
    already = set(i["src"] for i in instances)
    for o in sdl["objects"]:
        if o["id"] in already: continue
        x,z = base[o["id"]]["pos2"]
        instances.append({"id": o["id"], "src": o["id"], "pos2":(x,z), "rotY":0.0, "scale":1.0})

    # 8) 小径 polyline
    def cubic_bezier(p0, p1, p2, p3, t):
        u = 1-t
        x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
        y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
        return (x,y)

    polylines = []
    seen_pairs = set()
    for fr, to, w in paths:
        if fr not in base or to not in base: continue
        key = (fr, to)
        if key in seen_pairs: continue
        seen_pairs.add(key)

        ax, az = base[fr]["pos2"]; bx, bz = base[to]["pos2"]
        mid = ((ax+bx)/2, (az+bz)/2)
        dx, dz = bx-ax, bz-az
        L = max(1e-4, math.hypot(dx,dz))
        nx, nz = -dz/L, dx/L
        off = min(6.0, 0.25*L)
        p0, p3 = (ax,az), (bx,bz)
        p1 = (mid[0]+nx*off, mid[1]+nz*off)
        p2 = (mid[0]+nx*off, mid[1]+nz*off)
        pts = [cubic_bezier(p0,p1,p2,p3, i/19.0) for i in range(20)]
        polylines.append({"id": f"path_{fr}_to_{to}", "width": w, "points": pts})

    # 9) 池塘（圆盘）
    discs = []
    for o in sdl["objects"]:
        if o["category"] == "water":
            r = radius_from_dimensions("water", o["dimensions"])
            x,z = base[o["id"]]["pos2"]
            discs.append({"id": o["id"], "radius": r, "pos2": [x,z]})

    # 10) 导出 dict
    out = {
        "units":"meters",
        "y_up": True,
        "ground_size":[W,H],
        "models":[
            {"id":i["id"], "src_object":i["src"],
             "position":[ i["pos2"][0], 0.0, i["pos2"][1] ],
             "rotation_euler_deg":[0.0, i["rotY"], 0.0],
             "scale": i["scale"]}
            for i in instances
        ],
        "procedural":{
            "polylines": polylines,
            "discs": discs
        }
    }
    return out

def save_layout(out_dict: Dict)->str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.abspath(os.path.join(OUTDIR, f"placements_{ts}.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
    return out_path

# 命令行入口：打印生成文件的绝对路径（供 C++ 捕获）
if __name__=="__main__":
    try:
        if len(sys.argv) < 2:
            print("usage: python scripts/sdl_to_layout.py <path_to_sdl.json>")
            sys.exit(1)
        sdl_path = sys.argv[1]
        if not os.path.exists(sdl_path):
            print(f"ERROR: SDL file not found: {sdl_path}", file=sys.stderr)
            sys.exit(2)
        sdl = load_sdl(sdl_path)
        out_dict = compile_layout(sdl)
        out_path = save_layout(out_dict)
        print(out_path)  # ★ 仅打印路径，供外部捕获
    except Exception as e:
        import traceback
        print("ERROR:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)
