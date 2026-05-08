"""
RF3 可视化模块 - DeepBinder风格
支持: pLDDT分布图、PAE热力图、RMSD残基着色、通过/未通过状态卡片
"""
import base64
import json
from typing import Dict, List, Optional


def render_rf3_result_card(result: Dict, index: int = 0) -> str:
    design_idx = result.get("design_idx", 0)
    rmsd = result.get("rmsd", -1.0)
    avg_plddt = result.get("avg_plddt", 0)
    passed = result.get("passed", False)
    is_mock = result.get("mock", False)

    if passed:
        status_bg = "#ecfdf5"
        status_color = "#059669"
        status_icon = "✅"
        status_text = "通过"
    elif rmsd >= 0:
        status_bg = "#fef2f2"
        status_color = "#dc2626"
        status_icon = "❌"
        status_text = "未通过"
    else:
        status_bg = "#f8fafc"
        status_color = "#94a3b8"
        status_icon = "⚠️"
        status_text = "验证失败"

    plddt_color = "#059669" if avg_plddt > 85 else "#d97706" if avg_plddt > 70 else "#dc2626"
    rmsd_color = "#059669" if rmsd < 2.0 else "#d97706" if rmsd < 3.0 else "#dc2626"
    mock_tag = "<span style='font-size:10px;background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;margin-left:6px;'>模拟</span>" if is_mock else ""

    rmsd_display = f"{rmsd:.3f}Å" if rmsd >= 0 else "N/A"
    plddt_display = f"{avg_plddt:.1f}" if avg_plddt else "N/A"

    return f"""
    <div style='padding:16px;border:1px solid #e2e8f0;border-radius:10px;background:white;margin-bottom:10px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
            <div style='display:flex;align-items:center;'>
                <span style='font-size:15px;font-weight:700;color:#1e293b;'>Design {design_idx + 1}</span>
                {mock_tag}
            </div>
            <span style='display:inline-flex;align-items:center;gap:4px;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;background:{status_bg};color:{status_color};'>
                {status_icon} {status_text}
            </span>
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
            <div style='padding:10px;background:#f8fafc;border-radius:8px;'>
                <div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>RMSD</div>
                <div style='font-size:20px;font-weight:700;color:{rmsd_color};'>{rmsd_display}</div>
                <div style='font-size:10px;color:#94a3b8;margin-top:2px;'>阈值: <2.0Å</div>
            </div>
            <div style='padding:10px;background:#f8fafc;border-radius:8px;'>
                <div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>pLDDT</div>
                <div style='font-size:20px;font-weight:700;color:{plddt_color};'>{plddt_display}</div>
                <div style='font-size:10px;color:#94a3b8;margin-top:2px;'>/100</div>
            </div>
        </div>
        <div style='margin-top:10px;'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>RMSD进度</div>
            <div style='width:100%;height:8px;background:#f1f5f9;border-radius:4px;'>
                <div style='width:{min(max(0, (1 - rmsd/5) * 100), 100) if rmsd >= 0 else 0}%;height:100%;background:{rmsd_color};border-radius:4px;transition:width 0.3s;'></div>
            </div>
        </div>
    </div>
    """


def render_rf3_plddt_chart(plddt_data: List[float], design_idx: int = 0) -> str:
    if not plddt_data:
        return "<div style='padding:20px;color:#94a3b8;text-align:center;'>暂无pLDDT数据</div>"

    bar_width = max(2, min(8, 600 / len(plddt_data)))

    bars_html = ""
    for i, val in enumerate(plddt_data):
        height_pct = (val / 100) * 100
        color = "#059669" if val > 85 else "#d97706" if val > 70 else "#dc2626"
        bars_html += f"<div style='width:{bar_width}px;height:{height_pct}%;background:{color};border-radius:1px 1px 0 0;min-height:1px;' title='Res {i+1}: {val:.1f}'></div>"

    avg = sum(plddt_data) / len(plddt_data)
    avg_color = "#059669" if avg > 85 else "#d97706" if avg > 70 else "#dc2626"

    return f"""
    <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:12px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
            <span style='font-size:14px;font-weight:700;color:#1e293b;'>Design {design_idx + 1} - pLDDT置信度</span>
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


def render_rf3_pae_heatmap(pae_data: List[List[float]], design_idx: int = 0) -> str:
    if not pae_data or not pae_data[0]:
        return "<div style='padding:20px;color:#94a3b8;text-align:center;'>暂无PAE数据</div>"

    n = len(pae_data)
    cell_size = max(2, min(6, 300 / n))

    max_pae = max(max(row) for row in pae_data) if pae_data else 30
    max_pae = min(max_pae, 30)

    cells_html = ""
    for i, row in enumerate(pae_data):
        for j, val in enumerate(row):
            norm = min(val / max_pae, 1.0)
            r = int(norm * 220)
            g = int((1 - norm) * 100)
            b = int((1 - norm) * 50)
            color = f"rgb({r},{g},{b})"
            cells_html += f"<div style='width:{cell_size}px;height:{cell_size}px;background:{color};' title='PAE({i+1},{j+1})={val:.1f}'></div>"

    return f"""
    <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:12px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
            <span style='font-size:14px;font-weight:700;color:#1e293b;'>Design {design_idx + 1} - PAE预测误差</span>
            <span style='font-size:11px;color:#94a3b8;'>{n}×{n} 残基</span>
        </div>
        <div style='display:grid;grid-template-columns:repeat({n},{cell_size}px);gap:0px;line-height:0;'>
            {cells_html}
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:8px;font-size:10px;color:#94a3b8;'>
            <span>Scored residue</span>
            <span>Aligned residue</span>
        </div>
        <div style='display:flex;align-items:center;gap:8px;margin-top:10px;'>
            <span style='font-size:10px;color:#94a3b8;'>0Å</span>
            <div style='flex:1;height:8px;background:linear-gradient(to right, rgb(0,100,50), rgb(220,0,0));border-radius:4px;'></div>
            <span style='font-size:10px;color:#94a3b8;'>{max_pae:.0f}Å</span>
        </div>
    </div>
    """


def render_rf3_rmsd_chart(per_res_rmsd: List[float], design_idx: int = 0, threshold: float = 2.0) -> str:
    if not per_res_rmsd:
        return "<div style='padding:20px;color:#94a3b8;text-align:center;'>暂无残基RMSD数据</div>"

    bar_width = max(2, min(8, 600 / len(per_res_rmsd)))

    bars_html = ""
    for i, val in enumerate(per_res_rmsd):
        height_pct = min((val / 5.0) * 100, 100)
        color = "#059669" if val < threshold else "#d97706" if val < 3.0 else "#dc2626"
        bars_html += f"<div style='width:{bar_width}px;height:{height_pct}%;background:{color};border-radius:1px 1px 0 0;min-height:1px;' title='Res {i+1}: {val:.3f}Å'></div>"

    avg_rmsd = sum(per_res_rmsd) / len(per_res_rmsd)
    avg_color = "#059669" if avg_rmsd < threshold else "#d97706" if avg_rmsd < 3.0 else "#dc2626"

    return f"""
    <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:12px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
            <span style='font-size:14px;font-weight:700;color:#1e293b;'>Design {design_idx + 1} - 残基RMSD</span>
            <span style='font-size:13px;font-weight:600;color:{avg_color};'>平均: {avg_rmsd:.3f}Å</span>
        </div>
        <div style='position:relative;'>
            <div style='display:flex;align-items:flex-end;height:120px;gap:1px;padding:0 4px;border-bottom:1px solid #e2e8f0;'>
                {bars_html}
            </div>
            <div style='position:absolute;left:0;right:0;top:{(1 - threshold/5.0) * 100}%;border-top:2px dashed #dc2626;z-index:5;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:10px;color:#94a3b8;'>
            <span>N-term</span>
            <span style='color:#dc2626;'>--- 阈值 {threshold}Å</span>
            <span>C-term</span>
        </div>
    </div>
    """


def render_rf3_summary_metrics(results: List[Dict]) -> str:
    if not results:
        return ""

    total = len(results)
    passed = len([r for r in results if r.get("passed")])
    failed = total - passed
    best_rmsd = min((r["rmsd"] for r in results if r.get("rmsd", -1) >= 0), default=-1)
    avg_plddt_all = sum(r.get("avg_plddt", 0) for r in results) / total if total > 0 else 0

    pass_rate = (passed / total * 100) if total > 0 else 0
    pass_color = "#059669" if pass_rate > 50 else "#d97706" if pass_rate > 20 else "#dc2626"

    return f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;'>
        <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:10px;text-align:center;'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>总验证数</div>
            <div style='font-size:24px;font-weight:700;color:#1e293b;'>{total}</div>
        </div>
        <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:10px;text-align:center;'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>通过率</div>
            <div style='font-size:24px;font-weight:700;color:{pass_color};'>{pass_rate:.0f}%</div>
            <div style='font-size:11px;color:#94a3b8;'>{passed}/{total}</div>
        </div>
        <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:10px;text-align:center;'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>最佳RMSD</div>
            <div style='font-size:24px;font-weight:700;color:#059669;'>{best_rmsd:.3f}Å</div>
        </div>
        <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:10px;text-align:center;'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>平均pLDDT</div>
            <div style='font-size:24px;font-weight:700;color:#6366f1;'>{avg_plddt_all:.1f}</div>
        </div>
    </div>
    """
