# -*- coding: utf-8 -*-
# D:/Creator/scripts/tripo_batch_processor.py
#
# 使用 Tripo 官方 Python SDK，从 SDL 批量提交 text_to_model 任务，并把生成模型下载到本地。
# 与 C++ 侧的通信协议保持一致：
#   - 进度：  PROGRESS:x/total
#   - 状态：  STATUS:...
#   - 文件：  FILE:abs_path_to_model

import os
import sys
import json
import argparse
from pathlib import Path

# ===== 环境与输出 =====
try:
    from dotenv import load_dotenv
    load_dotenv()  # 允许同目录 .env 里放 TRIPO_API_KEY=tsk_xxx
except Exception:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import asyncio

# --- 安装： pip install tripo3d python-dotenv ---
from tripo3d import TripoClient, TaskStatus


def out(line: str):
    """打印到 stdout 并立即 flush（供 C++ 侧管道实时读取）。"""
    print(line)
    try:
        sys.stdout.flush()
    except Exception:
        pass


# 扩展名优先级（用于选择最合适的文件）
_EXT_PRIORITY = ["glb", "gltf", "obj", "fbx", "usdz", "stl", "zip"]


def _sort_paths_by_priority(paths):
    def key_func(p: Path):
        ext = p.suffix.lower().lstrip(".")
        try:
            return _EXT_PRIORITY.index(ext)
        except ValueError:
            return len(_EXT_PRIORITY) + 1
    return sorted(paths, key=key_func)


async def process_object(client: TripoClient, obj: dict, index: int, total: int, output_base: Path):
    """提交单个对象并下载模型，按协议输出 PROGRESS/STATUS/FILE。"""
    obj_id = obj.get("id", f"unknown_object_{index+1}")
    prompt = obj.get("tripo_prompt")

    # 进度（从 1 开始）
    out(f"PROGRESS:{index+1}/{total}")

    if not prompt:
        out(f"STATUS:Skipping object {index+1}/{total} ({obj_id}): no 'tripo_prompt'.")
        return

    out(f"STATUS:开始处理对象 {index+1}/{total}: {obj_id}")
    out(f"STATUS:[{obj_id}] 提交生成请求...")

    try:
        # 1) 提交任务
        task_id = await client.text_to_model(prompt=prompt)
        out(f"STATUS:[{obj_id}] 任务已创建 (ID: {task_id[:8]}...)，等待完成...")

        # 2) 等待结果（SDK 内部轮询；不使用 verbose，避免非协议输出）
        task = await client.wait_for_task(task_id, verbose=False)

        if task.status == TaskStatus.SUCCESS:
            out(f"STATUS:[{obj_id}] 成功! 准备下载模型...")

            # 3) 下载全部可用模型文件到指定目录
            obj_dir = output_base / obj_id
            obj_dir.mkdir(parents=True, exist_ok=True)

            files_map = await client.download_task_models(task, str(obj_dir))
            # files_map: dict[str, str] 例如 {"pbr_model": ".../pbr_model.glb", "base_model": ".../base_model.obj"}
            if not files_map:
                out(f"STATUS:[{obj_id}] 错误: 未从 SDK 获取到可下载文件。")
                return

            # 4) 根据扩展名优先级排序后，逐条回报 FILE
            paths = [Path(p) for p in files_map.values() if isinstance(p, str)]
            if not paths:
                out(f"STATUS:[{obj_id}] 错误: 下载结果为空或无效。")
                return

            for p in _sort_paths_by_priority(paths):
                out(f"FILE:{str(p.resolve())}")

            out(f"STATUS:[{obj_id}] 下载完成。")

        elif task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            msg = getattr(task, "error_message", None) or "未知错误"
            out(f"STATUS:[{obj_id}] 任务失败: {msg}")
        else:
            # 其它状态（极少见）
            out(f"STATUS:[{obj_id}] 任务状态: {getattr(task.status, 'name', str(task.status))}")

    except Exception as e:
        out(f"STATUS:[{obj_id}] 发生异常: {e}")


async def main_async(sdl_path: str):
    # 1) API Key
    api_key = os.environ.get("TRIPO_API_KEY")
    if not api_key:
        out("ERROR: Environment variable TRIPO_API_KEY not set.")
        sys.exit(1)

    # 2) 读取 SDL
    try:
        with open(sdl_path, "r", encoding="utf-8") as f:
            sdl = json.load(f)
        objects = sdl.get("objects", [])
        if not objects:
            out("ERROR: No 'objects' found in the SDL file.")
            sys.exit(1)
    except Exception as e:
        out(f"ERROR: Failed to read or parse SDL file: {e}")
        sys.exit(1)

    total = len(objects)
    output_base = Path("D:/Creator/assets").resolve()

    # 3) 使用 SDK 客户端
    async with TripoClient(api_key=api_key) as client:
        for i, obj in enumerate(objects):
            await process_object(client, obj, i, total, output_base)

    # 4) 收尾
    out(f"PROGRESS:{total}/{total}")
    out("STATUS:所有任务处理完毕。")


def main():
    parser = argparse.ArgumentParser(description="Batch generate 3D models from an SDL JSON file (Tripo SDK).")
    parser.add_argument("sdl_file_path", help="Path to the SDL JSON file.")
    args = parser.parse_args()

    # Windows / 常规环境均可直接跑
    asyncio.run(main_async(args.sdl_file_path))


if __name__ == "__main__":
    main()
