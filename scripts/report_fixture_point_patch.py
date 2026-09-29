"""Offline interactive inspection of full fixture point bundles and finite patches."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction/src"))
from creator_recon.domain.point_patch import compose, open_candidate_view  # noqa: E402
from run_fixture_point_patch import RUN, checked, digest, read, write  # noqa: E402


def build():
    import plotly.graph_objects as go

    _, manifest, _ = checked()
    inference = read(RUN / "inference.json")
    if inference["state"] != "complete" or inference["gt_read"]:
        raise ValueError("Complete normal integration required")
    for name, sha in inference["outputs"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Source output changed: "+name)
    destination = RUN / "preview"
    destination.mkdir(exist_ok=False)
    cards, files = [], []
    for case in manifest["cases"]:
        cid = case["case_id"]
        folder = RUN / cid
        base = compose(folder / "bundle/base")
        candidate = open_candidate_view(folder / "bundle/cylinder_support-enabled.json")
        points, ids = base["points"], base["point_ids"]
        # Fixed source-ID sample only for browser display. All points remain in bundles/evaluation.
        selected = np.linspace(0, len(points)-1, min(50000, len(points)), dtype=int)
        chosen = points[selected]
        with np.load(folder / "prediction.npz", allow_pickle=False) as native:
            source = ids[selected]
            colors = native["processed_images"][source[:, 0], source[:, 1], source[:, 2]]
        figure = go.Figure(go.Scatter3d(x=chosen[:, 0], y=chosen[:, 1], z=chosen[:, 2], mode="markers",
            marker=dict(size=1.1, color=[f"rgb({r},{g},{b})" for r, g, b in colors], opacity=.6), name="基础点云", hoverinfo="skip"))
        coordinates = [[], [], []]
        for segment in candidate["segments"]:
            for axis in range(3):
                coordinates[axis].extend([segment[0, axis], segment[1, axis], None])
        figure.add_trace(go.Scatter3d(x=coordinates[0], y=coordinates[1], z=coordinates[2], mode="lines",
            line=dict(color="#ff604a", width=7), name="新增细杆中心线", hoverinfo="skip"))
        figure.update_layout(template="plotly_dark", height=640, margin=dict(l=0, r=0, t=45, b=0),
            title=f"{cid}: {len(points):,} 个基础点 / {len(candidate['segments'])} 条新增有限线段",
            scene=dict(aspectmode="data", xaxis_title="X / m", yaxis_title="Y / m", zaxis_title="Z / m",
                       camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.3, y=-2.1, z=.7))),
            updatemenus=[dict(type="buttons", direction="right", x=0, y=1.07, buttons=[
                dict(label="基础", method="update", args=[dict(visible=[True, False])]),
                dict(label="基础 + 补丁", method="update", args=[dict(visible=[True, True])]),
                dict(label="仅检查补丁", method="update", args=[dict(visible=[False, True])]),
            ])])
        page = "<!doctype html><meta charset='utf-8'><title>Creator fixture " + html.escape(cid) + "</title>"
        page += "<body style='margin:20px;background:#111827;color:#e5e7eb;font:16px system-ui'>"
        page += "<p>新生成的 DA3 基础点云与可撤回细杆补丁。红线是中心线，线宽仅供显示；未恢复真实杆表面。坐标尺度依赖声明标定物。</p>"
        page += f"<p>浏览器均匀显示 {len(selected):,}/{len(points):,} 个来源点，完整快照与评价保留全部点。切换按钮只控制预览；保存状态见 bundle 的 enabled/withdrawn 文件。</p>"
        page += figure.to_html(full_html=False, include_plotlyjs=True, config=dict(responsive=True, displaylogo=False))
        path = destination / (cid+".html")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(page)
        files.append(dict(case_id=cid, path=path.relative_to(RUN).as_posix(), sha256=digest(path),
                          browser_points=len(selected), full_points=len(points)))
        cards.append(f'<li><a href="{cid}.html">{cid}：基础 / 补丁切换</a></li>')
    with (destination / "index.html").open("x", encoding="utf-8") as stream:
        stream.write("<!doctype html><meta charset='utf-8'><title>Creator 点云补丁</title>"
                     "<h1>Creator：新基础点云与细杆补丁</h1><p>离线预览，三个旧开发对象，非 G1 验收结果。</p><ul>"+"".join(cards)+"</ul>")
    write(destination / "receipt.json", dict(inference_sha256=digest(RUN / "inference.json"), files=files,
          report_source_sha256=digest(Path(__file__)), index_sha256=digest(destination / "index.html"),
          display_only=True, base_modified=False, suppression_applied=False))
    print("FIXTURE_PREVIEW", destination / "index.html", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    build()
