# -*- coding: utf-8 -*-
"""
批量按 JSON 场景数据对模型做“轴心归底 + 等比缩放”，并输出统一尺度的 GLB。
优先使用：pygltflib（仅修改节点矩阵，最大保真）
失败回退：trimesh.Scene（场景级矩阵变换后导出 GLB，通常会内嵌贴图）

依赖：
    pip install numpy trimesh
    （可选，为最佳保真）pip install pygltflib

用法：
    python calculate_bbox.py <scene_json> <assets_dir> [<output_dir>] [--write-registry] [--registry-path <file>] [--min-confidence 40]

示例：
    python calculate_bbox.py D:\Creator\Creator\SDL\scene_xxx.json D:\Creator\Creator\ProjectTest ^
        D:\Creator\Creator\assets\resized_output --write-registry

作者：你可信赖的 OpenGL 小伙伴
"""

import argparse
import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

# ---- 可选依赖：pygltflib（如果没有则自动回退）----
try:
    from pygltflib import GLTF2
    _HAS_PYGLTFLIB = True
except Exception:
    _HAS_PYGLTFLIB = False


# =========================
# 参数解析
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="按 SDL 场景对资产目录中的模型做：轴心归底 + 等比缩放（智能匹配资产文件）"
    )
    ap.add_argument("json_file", help="SDL/scene json 文件路径")
    ap.add_argument("assets_dir", help="资产目录（自动递归扫描 .glb/.gltf/.fbx/.obj/.ply/.dae）")
    ap.add_argument("output_dir", nargs="?", default="",
                    help="输出目录（默认 <assets_dir>/resized_output）")
    ap.add_argument("--write-registry", action="store_true",
                    help="处理完成后在 output_dir 写出 registry_autogen.json（id->输出GLB绝对路径）")
    ap.add_argument("--registry-path", default="",
                    help="显式指定 registry 输出路径（默认 output_dir/registry_autogen.json）")
    ap.add_argument("--min-confidence", type=float, default=40.0,
                    help="匹配最低置信分阈值（默认 40.0，低于该分数的匹配将被跳过）")
    return ap.parse_args()


# =========================
# 常量与工具
# =========================

ALLOWED_EXTS = [".glb", ".gltf", ".fbx", ".obj", ".ply", ".dae"]
EXT_BONUS = {".glb": 3.0, ".gltf": 2.5, ".fbx": 2.0, ".obj": 1.5, ".ply": 1.0, ".dae": 0.8}
EPS = 1e-6


def slugify(s: str) -> str:
    """仅保留字母数字，统一小写，用于名字/文件名的近似匹配。"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def safe_filename(s: str) -> str:
    """将任意 id/name 转为可作文件名的形态。"""
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = s.strip()
    return s or "unknown"


def build_assets_index(assets_dir: Path) -> List[Dict]:
    """
    递归扫描资产目录，收集候选文件。
    返回列表：[{stem_lower, slug, path, ext}, ...]
    """
    items = []
    for root, _, files in os.walk(assets_dir):
        for fn in files:
            p = Path(root) / fn
            ext = p.suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            stem = p.stem
            items.append({
                "stem_lower": stem.lower(),
                "slug": slugify(stem),
                "path": p,
                "ext": ext
            })
    return items


def find_best_match_for_object(obj_id: str,
                               element_name: str,
                               index: List[Dict],
                               min_conf: float) -> Tuple[Optional[Path], str, float]:
    """
    在 index 中为 (obj_id, element_name) 找到最合适的模型文件。
    评分策略（越高越好）：
      1) 文件名（stem）与 id 全等：+100
      2) 文件名包含 id 作为子串：+80
      3) 文件名 slug 与 element_name slug 全等：+70
      4) 文件名 slug 包含 name_slug：+60
      5) 模糊匹配：+ (difflib ratio * 50)
      6) 扩展名偏好加权 EXT_BONUS[ext]
    返回 (path, reason, score)；若 score < min_conf 则返回 (None, reason, score)
    """
    id_key = str(obj_id or "").lower()
    name_slug = slugify(element_name or "")
    best_path: Optional[Path] = None
    best_score = -1e9
    best_reason = "n/a"
    for it in index:
        score = 0.0
        stem = it["stem_lower"]
        slug = it["slug"]
        ext = it["ext"]

        if id_key:
            if stem == id_key:
                score += 100.0
            elif id_key in stem:
                score += 80.0

        if name_slug:
            if slug == name_slug:
                score += 70.0
            elif name_slug in slug:
                score += 60.0
            r_name = difflib.SequenceMatcher(None, name_slug, slug).ratio()
            r_id = difflib.SequenceMatcher(None, id_key, stem).ratio() if id_key else 0.0
            score += max(r_name, r_id) * 50.0

        score += EXT_BONUS.get(ext, 0.0)

        if score > best_score:
            best_score = score
            best_path = it["path"]
            best_reason = f"stem='{stem}', ext={ext}, score={score:.2f}"

    if best_score < min_conf:
        return None, f"no confident match (<{min_conf:.1f})", best_score
    return best_path, best_reason, best_score


def compute_bounds_with_trimesh_scene(model_path: Path):
    """
    用 trimesh.Scene 读取并返回：scene, min_point, max_point
    （Scene 的 bounds 是“应用节点变换后”的全局包围盒）
    """
    scene = trimesh.load(model_path, force='scene')
    if scene is None or len(scene.geometry) == 0:
        return None, None, None
    min_point, max_point = scene.bounds
    return scene, min_point, max_point


def make_transform(bottom_center_point: np.ndarray, scale: float) -> np.ndarray:
    """
    生成 4x4 行优先矩阵 M = S @ T
    左乘到节点：N' = M @ N
    语义：先将模型底部中心平移到原点，再做等比缩放（围绕原点缩放，底部仍贴地）。
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -bottom_center_point
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = float(scale)
    return S @ T


# ---- 以下几个函数仅在使用 pygltflib 路径时用到 ----
def mat4_from_trs(translation=None, rotation=None, scale=None):
    T = np.eye(4, dtype=np.float64)
    if translation is not None:
        T[:3, 3] = translation

    R = np.eye(4, dtype=np.float64)
    if rotation is not None:
        x, y, z, w = rotation  # glTF: [x,y,z,w]
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R[:3, :3] = np.array([
            [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
            [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
        ], dtype=np.float64)

    S = np.eye(4, dtype=np.float64)
    if scale is not None:
        S[0, 0], S[1, 1], S[2, 2] = scale

    return T @ R @ S


def node_matrix_to_np(node):
    # 读 glTF node 的 matrix（列优先）→ 转为 numpy 行优先
    if getattr(node, "matrix", None) and len(node.matrix) == 16:
        M = np.array(node.matrix, dtype=np.float64).reshape((4, 4))
        return M.T.copy()
    else:
        t = node.translation if node.translation else [0.0, 0.0, 0.0]
        r = node.rotation if node.rotation else [0.0, 0.0, 0.0, 1.0]
        s = node.scale if node.scale else [1.0, 1.0, 1.0]
        return mat4_from_trs(t, r, s)


def set_node_matrix_from_np(node, M_np: np.ndarray):
    # numpy 行优先 → glTF 列优先
    node.matrix = (M_np.T.reshape(-1).astype(float)).tolist()
    node.translation = None
    node.rotation = None
    node.scale = None


def left_multiply_on_tree(gltf, node_index: int, M: np.ndarray):
    node = gltf.nodes[node_index]
    N = node_matrix_to_np(node)
    set_node_matrix_from_np(node, M @ N)
    if node.children:
        for c in node.children:
            left_multiply_on_tree(gltf, c, M)


def decide_scale_factor(dim_type: str, target_value: float,
                        width: float, height: float, depth: float) -> Optional[float]:
    """
    根据 JSON 的尺寸定义选择缩放因子：
    - height: 使用当前高度作为分母
    - diameter: 使用当前水平最大跨度 max(width, depth) 作为分母
    """
    if dim_type == 'height':
        if height > EPS:
            return float(target_value) / float(height)
        return None
    elif dim_type == 'diameter':
        horizontal_extent = max(width, depth)
        if horizontal_extent > EPS:
            return float(target_value) / float(horizontal_extent)
        return None
    else:
        return None


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def read_scene_objects(json_path: Path) -> List[Dict]:
    """读取 SDL/scene json，返回 objects 列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    # 兼容可能的字段名
    objects = scene_data.get("objects")
    if objects is None:
        objects = scene_data.get("models")
    if not isinstance(objects, list):
        objects = []

    return objects


def get_object_display_name(obj: Dict) -> str:
    """尽力获取对象的可读名称"""
    return (obj.get("element_name")
            or obj.get("name")
            or obj.get("label")
            or obj.get("model_name")
            or "")


def get_object_id(obj: Dict) -> str:
    """尽力获取对象 id（字符串化）"""
    return str(obj.get("id") if "id" in obj else obj.get("uuid") if "uuid" in obj else "")


def get_object_dimension(obj: Dict) -> Tuple[Optional[str], Optional[float]]:
    """
    从对象里解析尺寸约束，返回 (dim_type, target_value)
    主通道：obj["dimensions"] = {"type": "height"/"diameter", "value": number}
    兜底：若存在 obj["height"] 是数值，也当作 height
    """
    dims = obj.get("dimensions", {}) or {}
    dim_type = str(dims.get("type") or "").strip().lower()
    target_value = safe_float(dims.get("value"))
    if not dim_type or target_value is None:
        # 兜底：若对象直接有数值 height
        if "height" in obj:
            hv = safe_float(obj.get("height"))
            if hv is not None:
                return "height", hv
        return None, None
    return dim_type, target_value


# =========================
# 主流程
# =========================

def resize_models_from_scene_data(json_file: Path,
                                  assets_dir: Path,
                                  output_dir: Path,
                                  write_registry: bool,
                                  registry_path: Optional[Path],
                                  min_conf: float):
    print("--- 场景模型尺寸调整脚本启动 ---")

    if not json_file.is_file():
        print(f"错误: JSON 文件不存在 -> {json_file}")
        return

    if not assets_dir.is_dir():
        print(f"错误: 资产目录不存在 -> {assets_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"资产目录: {assets_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 64)

    # 读取 JSON
    try:
        objects = read_scene_objects(json_file)
        print(f"成功从 {json_file.name} 中加载了 {len(objects)} 个对象定义。")
    except Exception as e:
        print(f"错误: 解析 JSON 失败 -> {e}")
        return

    # 建索引（一次）
    print(f"扫描资产目录以建立索引: {assets_dir}")
    assets_index = build_assets_index(assets_dir)
    print(f"共发现 {len(assets_index)} 个候选模型文件（包含子目录，允许后缀：{', '.join(ALLOWED_EXTS)}）")

    if not _HAS_PYGLTFLIB:
        print("提示：未安装 pygltflib，将直接使用 trimesh 回退方案；"
              "为最大保真建议执行： pip install pygltflib")

    start_time = time.perf_counter()
    processed = 0

    # 写 registry 所需
    saved_map: Dict[str, str] = {}

    # 遍历对象
    for idx, obj in enumerate(objects, 1):
        object_id = get_object_id(obj)
        element_name = get_object_display_name(obj)
        dim_type, target_value = get_object_dimension(obj)

        print(f"\n[{idx}/{len(objects)}] 对象 id='{object_id}' name='{element_name}'")

        # 匹配模型文件
        model_path, reason, score = find_best_match_for_object(object_id, element_name,
                                                               assets_index, min_conf)
        if not model_path:
            print(f"  └─ [跳过] 未找到足够置信的匹配（score={score:.2f}）：{reason}")
            continue
        print(f"  ├─ 匹配文件: {model_path}  [{reason}]")

        # 尺寸约束检查
        if not dim_type or target_value is None:
            print("  └─ [跳过] 未定义目标尺寸（dimensions.type/value 或 height），不缩放。")
            continue
        print(f"  ├─ 目标尺寸: type='{dim_type}', value={target_value}")

        try:
            # 1) 读取 trimesh.Scene 以获得包围盒
            scene, min_point, max_point = compute_bounds_with_trimesh_scene(model_path)
            if scene is None:
                print("  └─ 错误: 模型为空或无法作为 Scene 加载。")
                continue

            current_dims = (max_point - min_point).astype(np.float64)
            width, height, depth = current_dims
            print(f"  ├─ 原始尺寸 (W,H,D): [{width:.4f}, {height:.4f}, {depth:.4f}]")

            center_point = (min_point + max_point) / 2.0
            bottom_center_point = np.array(
                [center_point[0], min_point[1], center_point[2]], dtype=np.float64
            )

            # 2) 计算缩放因子
            scale_factor = decide_scale_factor(dim_type, target_value, width, height, depth)
            if scale_factor is None:
                print(f"  └─ 警告: 无法根据 {dim_type} 计算缩放因子（可能当前尺寸接近 0）。")
                continue
            print(f"  ├─ 计算出的缩放因子: {scale_factor:.6f}")

            # 3) 变换矩阵（行优先）
            M = make_transform(bottom_center_point, scale_factor)

            # 4) 导出 —— 输出文件命名用对象 id，避免同源文件名冲突
            out_name = safe_filename(object_id or model_path.stem) + ".glb"
            output_path = (output_dir / out_name)

            ok = False
            if model_path.suffix.lower() in [".glb", ".gltf"] and _HAS_PYGLTFLIB:
                try:
                    gltf = GLTF2().load(str(model_path))
                    scene_index = gltf.scene if gltf.scene is not None else 0
                    if (scene_index is not None
                            and scene_index < len(gltf.scenes)
                            and gltf.scenes[scene_index].nodes):
                        for root in gltf.scenes[scene_index].nodes:
                            left_multiply_on_tree(gltf, root, M)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        gltf.save_binary(str(output_path))
                        ok = True
                        # 校验
                        try:
                            chk_scene, chk_min, chk_max = compute_bounds_with_trimesh_scene(output_path)
                            if chk_scene is not None:
                                new_dims = (chk_max - chk_min)
                                print(f"  ├─ (校验) 写回后尺寸 (W,H,D): "
                                      f"[{new_dims[0]:.4f}, {new_dims[1]:.4f}, {new_dims[2]:.4f}]")
                        except Exception:
                            pass
                    else:
                        ok = False
                except Exception as e:
                    print(f"  ├─ [pygltflib] 失败，回退 trimesh：{e}")
                    ok = False

            if not ok:
                try:
                    sc2 = scene.copy()
                    sc2.apply_transform(M)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    sc2.export(str(output_path), file_type="glb")
                    ok = True
                except Exception as e:
                    print(f"  └─ [trimesh] 导出失败：{e}")
                    ok = False

            if ok:
                processed += 1
                print(f"  └─ ✔ 成功保存至: {output_path}")
                saved_map[str(object_id or model_path.stem)] = str(output_path.resolve())
            else:
                print(f"  └─ ❌ 导出失败。")

        except Exception as e:
            print(f"  └─ ❌ 处理时发生错误: {e}")

    # 统计与 registry
    end_time = time.perf_counter()
    print("\n" + "=" * 64)
    print("--- 批量处理完成 ---")
    print(f"成功处理并保存了 {processed} 个模型。")
    print(f"总耗时: {end_time - start_time:.4f} 秒。")
    print("=" * 64)

    if write_registry:
        reg_path = registry_path if registry_path else (output_dir / "registry_autogen.json")
        try:
            reg = {"models": [{"id": k, "path": v} for k, v in saved_map.items()]}
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(reg_path, "w", encoding="utf-8") as f:
                json.dump(reg, f, ensure_ascii=False, indent=2)
            print(f"[registry] 写出成功: {reg_path}")
        except Exception as e:
            print(f"[registry] 写出失败: {e}")


# =========================
# 入口
# =========================

if __name__ == "__main__":
    args = parse_args()
    json_file = Path(args.json_file)
    assets_dir = Path(args.assets_dir)
    output_dir = Path(args.output_dir or (assets_dir / "resized_output"))
    registry_path = Path(args.registry_path) if args.registry_path else None

    resize_models_from_scene_data(
        json_file=json_file,
        assets_dir=assets_dir,
        output_dir=output_dir,
        write_registry=args.write_registry,
        registry_path=registry_path,
        min_conf=float(args.min_confidence),
    )
