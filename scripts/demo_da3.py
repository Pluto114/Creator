"""Small DA3 experiment: photographs -> depth, colored points, GLB and offline HTML."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from smoke_da3 import run, sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, nargs='+')
    parser.add_argument('--model', choices=('base', 'large'), default='base')
    parser.add_argument('--reuse-run', type=Path)
    parser.add_argument('--process-res', type=int, default=504)
    parser.add_argument('--use-ray-pose', action='store_true')
    args = parser.parse_args()
    if args.images is None:
        example = root / '.local/setup/Depth-Anything-3/assets/examples/SOH'
        args.images = [example / '000.png', example / '010.png']
    if not 2 <= len(args.images) <= 5:
        parser.error('This small demo accepts 2-5 overlapping photographs.')
    output = root / '.runtime/demo' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True, exist_ok=False)
    status_file = output / 'report.json'
    status_file.write_text(json.dumps({'ok': False, 'state': 'running'}), encoding='utf-8')
    try:
        if args.reuse_run:
            from environment_paths import require_project_environment
            require_project_environment(root)
            source = args.reuse_run.resolve()
            report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
            if not report.get('ok') or report.get('state') != 'succeeded':
                raise RuntimeError('Only a successful saved inference can be reused.')
            report = dict(report)
            args.images = [Path(item['path']) for item in report['images']]
            report['source_run'] = str(source)
            report['source_prediction_sha256'] = sha256(source / 'prediction.npz')
            shutil.copy2(source / 'prediction.npz', output / 'prediction.npz')
        else:
            report = run(args, root, output)
        # Use the pinned upstream export routines; no custom repair or learned method here.
        import numpy as np
        import plotly.graph_objects as go
        from depth_anything_3.specs import Prediction
        from depth_anything_3.utils.export.glb import (
            _depths_to_world_points_with_colors,
            export_to_glb,
            get_conf_thresh,
        )
        from PIL import Image
        from pointcloud_preview import align_for_display, camera_raster

        with np.load(output / 'prediction.npz', allow_pickle=False) as data:
            prediction = Prediction(is_metric=report.get('is_metric') or 0,
                                    **{name: data[name] for name in data.files})
        export_to_glb(prediction, str(output), num_max_points=100_000)
        threshold = get_conf_thresh(prediction, None, 1.05, conf_thresh_percentile=40.0)
        points, colors = _depths_to_world_points_with_colors(
            prediction.depth, prediction.intrinsics, prediction.extrinsics,
            prediction.processed_images, prediction.conf, threshold,
        )
        if not len(points):
            raise RuntimeError('Confidence filtering retained no points.')
        # Display rigid alignment matches the upstream GLB frame; preserve every retained point.
        displayed, cameras, display_transform = align_for_display(points, prediction.extrinsics)
        scale = float(max(np.abs(displayed).max(), np.abs(cameras).max()) * 1.05)
        indices = np.random.default_rng(0).choice(len(points), min(100_000, len(points)), replace=False)
        shown, rgb = displayed[indices] / scale, colors[indices]
        figure = go.Figure(go.Scatter3d(
            x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode='markers',
            marker={'size': 1.5, 'color': [f'rgb({r},{g},{b})' for r, g, b in rgb]},
            name='重建点云', hoverinfo='skip',
        ))
        normalized_cameras = cameras / scale
        figure.add_trace(go.Scatter3d(
            x=normalized_cameras[:, 0], y=normalized_cameras[:, 1], z=normalized_cameras[:, 2],
            mode='markers+text', text=[f'照片 {i+1}' for i in range(len(cameras))],
            marker={'size': 5, 'color': '#ff795e'}, name='拍摄位置（点击图例开关）',
            visible='legendonly',
        ))
        # Fixed cube ranges map [-1,1] data to [-0.5,0.5] Plotly camera-domain units.
        eye = normalized_cameras[0] / 2
        distance = float(np.median(prediction.depth[0])) / scale / 2
        target = eye + np.array([0., 0., -distance])
        def camera_at(position):
            return {'eye': dict(zip('xyz', position.tolist())),
                    'center': dict(zip('xyz', target.tolist())),
                    'up': {'x': 0, 'y': 1, 'z': 0}, 'projection': {'type': 'perspective'}}
        first_camera = camera_at(eye)
        angle = np.deg2rad(20)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        oblique_camera = camera_at(target + rotation @ (eye - target))
        overview = {'eye': {'x': 1.2, 'y': .6, 'z': 1.2},
                    'center': {'x': 0, 'y': 0, 'z': 0}, 'up': {'x': 0, 'y': 1, 'z': 0}}
        axis = {'range': [-1, 1], 'autorange': False, 'visible': False}
        figure.update_layout(
            template='plotly_dark', margin={'l': 0, 'r': 0, 't': 65, 'b': 0}, height=660,
            scene={'aspectmode': 'cube', 'xaxis': axis, 'yaxis': axis, 'zaxis': axis,
                   'camera': first_camera, 'dragmode': 'orbit', 'bgcolor': '#131b2a'},
            paper_bgcolor='#131b2a', showlegend=True,
            updatemenus=[{'type': 'buttons', 'direction': 'right', 'x': 0, 'y': 1.08,
                         'buttons': [
                             {'label': label, 'method': 'relayout', 'args': [{'scene.camera': camera}]}
                             for label, camera in [('拍摄方向', first_camera), ('偏转 20°', oblique_camera),
                                                   ('全场景概览', overview)]]
                         }],
        )
        for name, yaw in [('camera-view.png', 0), ('orbit-view.png', 15)]:
            raster = camera_raster(points, colors, prediction.intrinsics[0],
                                   prediction.extrinsics[0], prediction.depth[0], yaw)
            Image.fromarray(raster).save(output / name)
        cards = []
        for index, image in enumerate(prediction.processed_images):
            Image.fromarray(image).save(output / f'input-{index}.png')
            cards.append(f'<article><h3>照片 {index+1}</h3><img src="input-{index}.png" '
                         f'alt="处理后的输入照片 {index+1}"><p>对应的深度预测</p>'
                         f'<img src="depth_vis/{index:04d}.jpg" alt="深度预测 {index+1}"></article>')
        plot = figure.to_html(full_html=False, include_plotlyjs=True,
                              config={'responsive': True, 'displaylogo': False})
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Creator · DA3 最小实验</title><style>
body{background:#0c1220;color:#e6edf8;font:16px/1.6 system-ui;margin:24px auto;max-width:1500px;padding:0 24px}
h1{margin-bottom:4px}p{color:#b5c3d8}a{color:#8edbff}main{display:grid;grid-template-columns:320px 1fr;gap:24px}
article{background:#131b2a;padding:14px;margin-bottom:16px;border-radius:12px}img{width:100%;display:block}
section{min-width:0}aside{max-height:800px;overflow:auto}@media(max-width:850px){main{display:block}}
</style><h1>照片 → 深度 → 三维点云</h1>
<p>初始视角匹配第一张照片的拍摄位置和朝向；拖动旋转，滚轮缩放。上方按钮可以复位或查看全场景。</p>
<p>这是 DA3 上游模型的直接结果，尚未进行 Creator 细杆修复。点云允许缺口与重影，坐标单位不代表米。
深度图颜色仅帮助看远近，不是实际材质。</p>
<p>查看器已修正向上轴；仅做统一刚体变换和等比缩放，未裁掉远处背景或修补几何。
网页透视视场与照片不完全相同；下方静态检查图使用真实相机内参。拍摄位置标记可从图例打开。</p>
<details open><summary>同一组点的检查图：拍摄视角 / 偏转 15°（黑色表示没有保留的采样点）</summary>
<div style="display:flex;gap:12px"><img style="width:49%" src="camera-view.png" alt="按相机内参渲染的点云">
<img style="width:49%" src="orbit-view.png" alt="偏转15度的点云"></div>
<p>使用3×3像素点斑显示；没有生成表面。回投到原拍摄位置看起来像照片，不能单独证明深度正确。</p></details>'''
        page += f'<p>模型：{html.escape(report["model"])} · {len(args.images)} 张照片 · '
        page += f'浏览器显示 {len(shown):,} / 过滤后 {len(points):,} 个点</p>'
        page += '<p><a href="scene.glb" download>下载 GLB</a> · '
        page += '<a href="prediction.npz" download>原始深度与相机数组</a> · '
        page += '<a href="report.json">运行记录</a></p>'
        page += '<main><aside>' + ''.join(cards) + '</aside><section>' + plot + '</section></main></html>'
        (output / 'demo.html').write_text(page, encoding='utf-8')
        report.update(ok=True, state='succeeded', filtered_points=len(points),
                      raw_pixels=int(prediction.depth.size), actual_confidence_threshold=float(threshold),
                      display_transform=display_transform.tolist(), display_scale=scale,
                      display_camera=first_camera, geometry_repaired=False,
                      browser_points=len(shown), confidence_percentile=40.0,
                      scope='upstream visual demo; no Creator reconstruction improvements')
    except Exception as error:
        status_file.write_text(json.dumps({'ok': False, 'state': 'failed', 'error': repr(error)}),
                               encoding='utf-8')
        raise
    status_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'DEMO_READY={output / "demo.html"}', flush=True)


if __name__ == '__main__':
    main()