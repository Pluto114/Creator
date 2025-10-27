# -*- coding: utf-8 -*-
# D:/Creator/Creator/scripts/deepseek_chat.py

import os
import sys
import re
import json
from datetime import datetime
import argparse

# 允许同目录 .env 中设置 DEEPSEEK_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ====== 你的原有提示词（保持不变/略）======
SDL_GENERATION_INSTRUCTIONS_V3 =  """---
**系统指令：**

你是一位顶级的世界构建AI（World-Building AI）。你的任务是将用户模糊的场景概念，转化为一份结构化的、机器可读的**场景描述语言 (SDL)** 文件。这份文件将用于程序化地生成并布局一个完整的3D场景。

**基准参考 (Baseline Reference):**
默认一个成年人（`human-scale`）的高度为 **1.8** 个单位。所有其他对象的尺寸都应以此为参考进行常识性推断。

**核心任务：**

1.  **场景解构 (Scene Deconstruction):**
    * 从用户需求中识别出构成场景的核心**对象 (Objects)**。
    * 推断并定义这些对象之间的**空间关系 (Relations)**。

2.  **结构化输出 (Structured Output):**
    * 你的输出**必须**是一个结构严谨的JSON对象，**绝不能**包含任何Markdown代码块（即` ``` `）或其他解释性文字。你的整个回复必须是纯粹的、原始的JSON文本，以 `{` 开头，以 `}` 结尾。
    * JSON的根对象必须包含两个键：`objects` 和 `relations`。

3.  **对象定义 (`objects` Array):**
    * 这是一个JSON数组，每个元素代表一个场景中的独立对象。
    * 每个对象必须包含以下键：
        * `id`: 一个简短、唯一的英文ID（例如："main_cabin_1"）。
        * `element_name`: 中文名称（例如：“主木屋”）。
        * `tripo_prompt`: 用于文本到3D模型服务的、细节丰富的**英文**提示词。
        * `category`: 对象的类别。可选值："building", "prop", "flora", "fauna", "terrain", "water"。
        * `dimensions`: 一个描述其最终目标尺寸的JSON对象。**必须**包含 `type` (类型，对于绝大多数三维物体，请使用 `'height'`) 和 `value` (一个代表单位的浮点数值) 两个键。例如：`"dimensions": {"type": "height", "value": 1.8}`。

4.  **关系定义 (`relations` Array):**
    * (此部分规则与V2版本相同，保持不变)
    * ...

**示例：**

# 用户输入：“我想要一个简单的日式乡村场景”
# 你的JSON输出应该是：
# {
#   "objects": [
#     {
#       "id": "main_house_1",
#       "element_name": "日式传统民居",
#       "tripo_prompt": "traditional japanese farmhouse (minka), thatched roof, dark wooden beams, sliding paper doors (shoji), engawa porch, serene and rustic",
#       "category": "building",
#       "dimensions": {
#         "type": "height",
#         "value": 7.5
#       }
#     },
#     {
#       "id": "stone_path_1",
#       "element_name": "石板路",
#       "tripo_prompt": "a winding path of flat, irregular grey stones (ishidatami), with moss growing between the cracks, leading through a garden",
#       "category": "terrain",
#       "dimensions": {
#         "type": "height",
#         "value": 0.1
#       }
#     },
#     {
#       "id": "pond_1",
#       "element_name": "小池塘",
#       "tripo_prompt": "small, calm pond with clear water, a few koi fish visible, water lilies on the surface, surrounded by smooth rocks",
#       "category": "water",
#       "dimensions": {
#         "type": "diameter", 
#         "value": 15.0
#       }
#     },
#     {
#       "id": "stone_lantern_1",
#       "element_name": "石灯笼",
#       "tripo_prompt": "a classic japanese stone lantern (toro), kasuga-style, covered in light moss, granite texture, placed in a garden setting",
#       "category": "prop",
#       "dimensions": {
#         "type": "height",
#         "value": 1.8
#       }
#     },
#     {
#       "id": "bamboo_1",
#       "element_name": "竹子",
#       "tripo_prompt": "a single tall, thin bamboo stalk, green leaves, segmented body, realistic",
#       "category": "flora",
#       "dimensions": {
#         "type": "height",
#         "value": 12.0
#       }
#     }
#   ],
#   "relations": [
#     {
#       "subject": "stone_path_1",
#       "verb": "connects",
#       "object": "main_house_1",
#       "target": "pond_1"
#     },
#     {
#       "subject": "stone_lantern_1",
#       "verb": "is_next_to",
#       "object": "pond_1",
#       "distance_hint": "close"
#     },
#     {
#       "subject": "bamboo_1",
#       "verb": "is_in_area",
#       "object": "main_house_1",
#       "distance_hint": "far"
#     }
#   ]
# }

**现在，请严格按照上述指令，处理用户在“---”分隔符之上的需求。**
"""

# -------------------- 配置 --------------------
# 脚本的根目录是 Creator/scripts/
# 我们希望SDL文件保存在 Creator/SDL/
# 所以路径需要从当前文件位置(scripts)向上退一级(Creator)再进入SDL
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDL_SAVE_DIRECTORY = os.path.join(SCRIPT_DIR, "..", "SDL")
SDL_SAVE_DIRECTORY = os.path.abspath(SDL_SAVE_DIRECTORY)

BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

def get_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY 未设置（可在系统环境或本目录 .env 中设置）")
    return key

def extract_json_from_response(raw_text: str) -> str:
    m = re.search(r'```(json)?\s*(\{.*\})\s*```', raw_text, re.DOTALL)
    if m:
        return m.group(2).strip()
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end+1].strip()
    if raw_text.strip().startswith('{') and raw_text.strip().endswith('}'):
        return raw_text.strip()
    return ""

def get_scene_sdl(client, user_description: str) -> dict:
    messages = [
        {"role": "system", "content": SDL_GENERATION_INSTRUCTIONS_V3},
        {"role": "user", "content": user_description}
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            temperature=0.2,
        )
    except Exception as e:
        raise RuntimeError(f"调用 DeepSeek API 失败: {e}")

    if not (resp.choices and resp.choices[0].message and resp.choices[0].message.content):
        raise RuntimeError("模型未能返回有效内容。")

    raw = resp.choices[0].message.content
    cleaned = extract_json_from_response(raw)
    if not cleaned:
        raise RuntimeError(f"未能从模型返回中提取有效 JSON。模型返回: {raw}")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 解析失败: {e}\n----\n{cleaned}")

def process_single_prompt(user_prompt: str) -> str:
    # 延迟导入，避免没有 openai 包时难定位
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"缺少 openai 包，请在 venv 中安装：pip install openai。详细: {e}")

    api_key = get_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    os.makedirs(SDL_SAVE_DIRECTORY, exist_ok=True)

    scene_sdl = get_scene_sdl(client, user_prompt)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scene_{ts}.json"
    full_path = os.path.join(SDL_SAVE_DIRECTORY, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(scene_sdl, f, ensure_ascii=False, indent=2)

    return os.path.abspath(full_path)

def main():
    parser = argparse.ArgumentParser(description="从用户描述生成场景描述语言 (SDL) JSON 文件。")
    parser.add_argument("prompt", type=str, help="用户提供的场景描述。")
    args = parser.parse_args()

    try:
        # 清洗 prompt（防换行/制表影响）
        prompt = args.prompt.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        path = process_single_prompt(prompt)
        # 成功：仅打印路径（C++ 侧用 stdout 捕获）
        print(path)
    except Exception as e:
        # 失败：打印 ERROR: 前缀（C++ 识别）
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()