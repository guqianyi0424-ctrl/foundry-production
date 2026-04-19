"""
RFD3 可视化模块 - ODesign风格
支持: 主链3D展示、pLDDT热图、设计排名卡片、结构叠加对比
"""
import base64
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


def render_rfd3_viewer(
    design_pdb_path: str,
    target_pdb_content: str = None,
    plddt_data: Optional[List[float]] = None,
    design_index: int = 0,
    rank: int = 1,
    height: int = 500
) -> str:
    design_b64 = ""
    try:
        with open(design_pdb_path, "r") as f:
            design_content = f.read()
        design_b64 = base64.b64encode(design_content.encode()).decode()
    except Exception:
        pass

    target_b64 = ""
    if target_pdb_content:
        target_b64 = base64.b64encode(target_pdb_content.encode()).decode()

    plddt_js = "const plddtData = null;"
    if plddt_data:
        plddt_js = f"const plddtData = {json.dumps(plddt_data)};"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #fff; font-family: 'Inter', -apple-system, sans-serif; }}
            .rfd3-container {{ width: 100%; height: {height}px; position: relative; }}
            #rfd3-viewer {{ width: 100%; height: 100%; }}
            .rfd3-toolbar {{
                position: absolute; right: 10px; top: 10px; z-index: 20;
                background: white; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 6px; display: flex; flex-direction: column; gap: 3px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }}
            .rfd3-toolbar button {{
                width: 34px; height: 34px; border: none; background: transparent;
                cursor: pointer; border-radius: 5px; font-size: 16px;
                display: flex; align-items: center; justify-content: center; color: #555;
            }}
            .rfd3-toolbar button:hover {{ background: #e3f2fd; color: #1976d2; }}
            .rfd3-info {{
                position: absolute; left: 10px; top: 10px; z-index: 20;
                background: rgba(255,255,255,0.95); border: 1px solid #e2e8f0;
                border-radius: 8px; padding: 10px 14px; font-size: 13px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); max-width: 200px;
            }}
            .rfd3-info .label {{ color: #64748b; font-size: 11px; font-weight: 500; }}
            .rfd3-info .value {{ color: #1e293b; font-size: 15px; font-weight: 700; }}
            .rfd3-info .plddt-good {{ color: #059669; }}
            .rfd3-info .plddt-ok {{ color: #d97706; }}
            .rfd3-info .plddt-bad {{ color: #dc2626; }}
            .legend-bar {{
                position: absolute; bottom: 40px; left: 10px; z-index: 20;
                background: rgba(255,255,255,0.95); border: 1px solid #e2e8f0;
                border-radius: 8px; padding: 8px 12px; font-size: 11px;
            }}
            .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
            .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
        </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.css">
    </head>
    <body>
        <div class='rfd3-container'>
            <div id='rfd3-viewer'></div>
            <div class='rfd3-info'>
                <div class='label'>Design #{design_index + 1}</div>
                <div class='value' style='margin-top:4px;'>Rank #{rank}</div>
                <div id='plddt-display' style='margin-top:6px;'>
                    <div class='label'>Avg pLDDT</div>
                    <div class='value plddt-good'>--</div>
                </div>
            </div>
            <div class='rfd3-toolbar'>
                <button onclick="resetView()" title="重置视角">🎯</button>
                <button onclick="toggleSpin()" title="旋转">🔄</button>
                <button onclick="zoomFit()" title="适应窗口">⬜</button>
                <button onclick="toggleTarget()" title="显示/隐藏靶标">👁️</button>
                <button onclick="screenshot()" title="截图">📷</button>
            </div>
            <div class='legend-bar'>
                <div class='legend-item'><div class='legend-dot' style='background:#4CAF50;'></div>靶标蛋白</div>
                <div class='legend-item'><div class='legend-dot' style='background:#6366f1;'></div>Binder主链</div>
                <div class='legend-item'><div class='legend-dot' style='background:#FF5722;'></div>热点残基</div>
            </div>
        </div>

        <script type="importmap">
        {{
            "imports": {{
                "molstar": "https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/esm/index.js",
                "molstar/lib/commonjs/mol-star": "https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/esm/index.js"
            }}
        }}
        </script>
        <script type="module">
        import {{ createPlugin }} from 'molstar/lib/mol-plugin-ui/plugin';
        import {{ DefaultPluginSpec }} from 'molstar/lib/mol-plugin-ui/spec';

        let plugin;
        let targetVisible = true;

        async function init() {{
            const container = document.getElementById('rfd3-viewer');
            plugin = await createPlugin(container, {{
                layout: {{ initial: {{ isExpanded: false, showControls: false }} }},
                spec: DefaultPluginSpec,
            }});

            if ("{target_b64}") {{
                await plugin.builders.data.download({{
                    url: 'data:text/plain;base64,{target_b64}',
                    isBinary: true, format: 'pdb',
                }}, {{ state: {{ isHidden: false }}, representation: {{ params: {{}} }} }});
            }}

            if ("{design_b64}") {{
                await plugin.builders.data.download({{
                    url: 'data:text/plain;base64,{design_b64}',
                    isBinary: true, format: 'pdb',
                }}, {{ state: {{ isHidden: false }}, representation: {{ params: {{}} }} }});
            }}

            plugin.build().toRoot();
            plugin.managers.camera.resetSnapshot();

            {plddt_js}
            if (plddtData && plddtData.length > 0) {{
                const avg = plddtData.reduce((a, b) => a + b, 0) / plddtData.length;
                const el = document.querySelector('#plddt-display .value');
                if (el) {{
                    el.textContent = avg.toFixed(1);
                    el.className = 'value ' + (avg > 85 ? 'plddt-good' : avg > 70 ? 'plddt-ok' : 'plddt-bad');
                }}
            }}

            window.resetView = () => plugin.managers.camera.resetSnapshot();
            window.toggleSpin = () => {{ const s = plugin.canvas3d?.props; if(s) s.spin = !s.spin; }};
            window.zoomFit = () => plugin.managers.camera.focus();
            window.toggleTarget = () => {{ targetVisible = !targetVisible; }};
            window.screenshot = () => plugin.canvas3d?.getImageData()?.toDataURL();
        }}

        init();
        </script>
    </body>
    </html>
    """
    return html


def render_rfd3_plddt_chart(plddt_data: List[float], design_index: int = 0) -> str:
    if not plddt_data:
        return "<div style='padding:20px;color:#94a3b8;text-align:center;'>暂无pLDDT数据</div>"

    max_val = max(plddt_data) if plddt_data else 100
    bar_width = max(2, min(8, 600 / len(plddt_data)))

    bars_html = ""
    for i, val in enumerate(plddt_data):
        height_pct = (val / 100) * 100
        color = "#059669" if val > 85 else "#d97706" if val > 70 else "#dc2626"
        bars_html += f"<div style='width:{bar_width}px;height:{height_pct}%;background:{color};border-radius:1px 1px 0 0;min-height:1px;' title='Res {i+1}: {val:.1f}'></div>"

    avg = sum(plddt_data) / len(plddt_data)
    avg_color = "#059669" if avg > 85 else "#d97706" if avg > 70 else "#dc2626"

    html = f"""
    <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:12px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
            <span style='font-size:14px;font-weight:700;color:#1e293b;'>Design {design_index + 1} - pLDDT分布</span>
            <span style='font-size:13px;font-weight:600;color:{avg_color};'>平均: {avg:.1f}</span>
        </div>
        <div style='display:flex;align-items:flex-end;height:120px;gap:1px;padding:0 4px;border-bottom:1px solid #e2e8f0;'>
            {bars_html}
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:10px;color:#94a3b8;'>
            <span>N-term</span>
            <span>C-term</span>
        </div>
        <div style='display:flex;gap:12px;margin-top:10px;font-size:11px;color:#64748b;'>
            <span><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#059669;margin-right:4px;'></span>>85 高置信</span>
            <span><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#d97706;margin-right:4px;'></span>70-85 中等</span>
            <span><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#dc2626;margin-right:4px;'></span><70 低置信</span>
        </div>
    </div>
    """
    return html


def render_rfd3_design_card(design: Dict, is_selected: bool = False) -> str:
    idx = design.get("index", 0)
    rank = design.get("rank", idx + 1)
    plddt = design.get("plddt", 0)
    is_mock = design.get("mock", False)
    plddt_color = "#059669" if plddt > 85 else "#d97706" if plddt > 70 else "#dc2626"
    selected_border = "border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,0.2);" if is_selected else ""
    mock_tag = "<span style='font-size:10px;background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;margin-left:6px;'>模拟</span>" if is_mock else ""

    return f"""
    <div style='padding:16px;border:1px solid #e2e8f0;border-radius:10px;background:white;{selected_border}cursor:pointer;transition:all 0.15s;'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-size:16px;font-weight:700;color:#1e293b;'>Design {idx + 1}</span>
            {mock_tag}
        </div>
        <div style='font-size:12px;color:#64748b;margin-top:4px;'>Rank #{rank}</div>
        <div style='margin-top:10px;'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:3px;'>pLDDT</div>
            <div style='display:flex;align-items:baseline;gap:4px;'>
                <span style='font-size:22px;font-weight:700;color:{plddt_color};'>{plddt:.1f}</span>
                <span style='font-size:12px;color:#94a3b8;'>/100</span>
            </div>
            <div style='width:100%;height:6px;background:#f1f5f9;border-radius:3px;margin-top:6px;'>
                <div style='width:{min(plddt, 100)}%;height:100%;background:{plddt_color};border-radius:3px;transition:width 0.3s;'></div>
            </div>
        </div>
    </div>
    """
