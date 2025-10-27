# D:/Creator/scripts/tripo_api_handler.py

import requests
import argparse
import time
import json
import os
import sys
import traceback # 用于打印更详细的异常信息

def generate_model(api_key, prompt, output_prefix):
    """
    调用 Tripo API 生成模型，轮询状态，下载文件。
    成功时打印完整文件路径到 stdout，失败时打印 "ERROR: message" 到 stdout。
    调试信息打印到 stderr。
    """
    task_create_url = "https://api.tripo3d.ai/v2/openapi/task"
    task_status_url_template = "https://api.tripo3d.ai/v2/openapi/task/{task_id}"
    final_output_path = "" # 初始化最终输出路径

    # --- 请求头 ---
    post_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    get_headers = {
        "Authorization": f"Bearer {api_key}"
    }

    task_id = None
    session = requests.Session() # 使用 Session

    try:
        # --- 1. 创建任务 ---
        create_payload = {
            "type": "text_to_model",
            "prompt": prompt
        }
        print(f"DEBUG: Sending task creation request for prompt: {prompt}", file=sys.stderr)
        response = session.post(task_create_url, headers=post_headers, json=create_payload, timeout=30)
        print(f"DEBUG: Task creation response status: {response.status_code}", file=sys.stderr)
        response.raise_for_status()
        result_json = response.json()

        if result_json.get("code") == 0 and result_json.get("data", {}).get("task_id"):
            task_id = result_json["data"]["task_id"]
            print(f"DEBUG: Task created successfully. Task ID: {task_id}", file=sys.stderr)
        else:
            print(f"ERROR: API task creation failed or returned unexpected format. Response: {response.text}")
            return

        # --- 2. 轮询任务状态 ---
        status_url = task_status_url_template.format(task_id=task_id)
        retries = 0
        max_retries = 30
        polling_interval = 10

        while retries < max_retries:
            print(f"DEBUG: Polling status for task {task_id} (Attempt {retries+1})...", file=sys.stderr)
            try:
                response = session.get(status_url, headers=get_headers, timeout=30)
                print(f"DEBUG: Status poll response status: {response.status_code}", file=sys.stderr)
                response.raise_for_status()
                status_json = response.json()

                if status_json.get("code") == 0 and status_json.get("data", {}).get("status"):
                    status = status_json["data"]["status"]
                    progress = status_json.get("data", {}).get("progress", 0)
                    print(f"DEBUG: Task status: {status}, Progress: {progress}%", file=sys.stderr)

                    if status == "success":
                        # --- 3. 处理成功状态 (已修正 AttributeError) ---
                        output_info = status_json.get("data", {}).get("output", {})
                        download_url = None
                        model_format = "glb" # 默认格式

                        # 尝试获取 pbr_model 信息
                        pbr_model_data = output_info.get("pbr_model")

                        # --- !! 修正 AttributeError 的逻辑 !! ---
                        if isinstance(pbr_model_data, dict):
                            # 情况一: pbr_model 是一个字典
                            print(f"DEBUG: Found pbr_model as dict: {pbr_model_data}", file=sys.stderr)
                            download_url = pbr_model_data.get("url")
                            model_format = pbr_model_data.get("type", model_format)
                        elif isinstance(pbr_model_data, str):
                            # 情况二: pbr_model 直接就是 URL 字符串
                            print(f"DEBUG: Found pbr_model as string (assuming URL): {pbr_model_data}", file=sys.stderr)
                            download_url = pbr_model_data
                            # 尝试从 URL 推断格式
                            if '.' in download_url.split('?')[0]:
                                deduced_format = download_url.split('?')[0].split('.')[-1].lower()
                                if deduced_format in ['glb', 'obj', 'stl', 'fbx']:
                                    model_format = deduced_format
                        else:
                            print(f"DEBUG: 'pbr_model' not found or has unexpected type. Checking other keys...", file=sys.stderr)
                            # 在这里可以添加查找其他 key 的逻辑, 例如:
                            # model_data = output_info.get("model") # 假设 API 可能用 'model' key
                            # if isinstance(model_data, dict):
                            #     download_url = model_data.get("url")
                            #     model_format = model_data.get("type", model_format)
                            # elif isinstance(model_data, str):
                            #     download_url = model_data
                            #     # ... 推断格式 ...

                        # --- 4. 如果找到了有效的下载链接，则进行下载 ---
                        if download_url:
                            print(f"DEBUG: Download URL confirmed: {download_url}", file=sys.stderr)
                            print(f"DEBUG: Deduced/Assigned format: {model_format}", file=sys.stderr)
                            print(f"DEBUG: Attempting to download model...", file=sys.stderr)
                            try:
                                # 使用 GET headers 或许更安全，但也可能不需要特定头
                                download_response = requests.get(download_url, headers=get_headers, stream=True, timeout=180)
                                download_response.raise_for_status()

                                # 构造最终保存路径
                                file_extension = f".{model_format}"
                                final_output_path = output_prefix + file_extension
                                output_dir = os.path.dirname(final_output_path)
                                if output_dir:
                                    os.makedirs(output_dir, exist_ok=True)

                                print(f"DEBUG: Saving model to: {final_output_path}", file=sys.stderr)
                                with open(final_output_path, "wb") as f:
                                    for chunk in download_response.iter_content(chunk_size=8192):
                                        f.write(chunk)

                                print(f"DEBUG: Model downloaded successfully.", file=sys.stderr)
                                # --- !! 成功: 打印完整路径到 stdout !! ---
                                print(final_output_path) # 打印到 stdout 给 C++
                                return # 函数成功结束

                            except requests.exceptions.RequestException as e:
                                print(f"ERROR: Model download failed: {e}") # 打印到 stdout
                                return
                            except IOError as e:
                                print(f"ERROR: Failed to save model file: {e}") # 打印到 stdout
                                return
                            except Exception as e:
                                print(f"ERROR: Unexpected error during download/save: {e}") # 打印到 stdout
                                print(traceback.format_exc(), file=sys.stderr) # 详细错误到 stderr
                                return
                        else: # 最终没有找到有效的下载链接
                            print(f"ERROR: Success status but no valid download URL could be extracted.") # 打印到 stdout
                            return

                    elif status == "running" or status == "queued":
                        print(f"DEBUG: Waiting {polling_interval} seconds...", file=sys.stderr)
                        time.sleep(polling_interval)
                        retries += 1
                    else: # 任务失败或未知状态
                        error_message = status_json.get("data", {}).get("error_message", f"Unknown status '{status}'")
                        print(f"ERROR: Task failed with status '{status}'. Message: {error_message}") # 打印到 stdout
                        return

                else: # API 返回码非 0 或缺少 data/status
                    print(f"ERROR: Status check returned unexpected format. Response: {response.text}") # 打印到 stdout
                    return

            except requests.exceptions.RequestException as e:
                print(f"ERROR: Status polling request failed: {e}", file=sys.stderr)
                print(f"DEBUG: Network error during polling, retrying in {polling_interval} seconds...", file=sys.stderr)
                time.sleep(polling_interval)
                retries += 1
            except json.JSONDecodeError:
                print(f"ERROR: Failed to parse status JSON response: {response.text}") # 打印到 stdout
                return # JSON 格式错误，停止轮询
            except Exception as e: # 捕获轮询循环中的其他异常
                print(f"ERROR: Unexpected error during polling loop: {e}") # 打印到 stdout
                print(traceback.format_exc(), file=sys.stderr)
                return

        # 如果循环结束是因为达到最大重试次数
        if retries >= max_retries:
            print(f"ERROR: Max retries reached while polling for task status.") # 打印到 stdout

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Initial request failed: {e}") # 打印到 stdout
    except Exception as e:
        # 捕获主流程中的其他意外错误
        print(f"ERROR: An unexpected error occurred in generate_model: {e}") # 打印到 stdout
        print(traceback.format_exc(), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D model using Tripo API.")
    parser.add_argument("--api_key", required=True, help="Your Tripo API key.")
    parser.add_argument("--prompt", required=True, help="Text prompt for model generation.")
    parser.add_argument("--output_prefix", required=True, help="Output file path prefix (without extension), e.g., D:/Creator/assets/mymodel")
    args = parser.parse_args()

    # 调用主函数
    generate_model(args.api_key, args.prompt, args.output_prefix)