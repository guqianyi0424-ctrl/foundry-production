"""
MPNN 可视化模块 - ODesign风格
支持: 序列对比展示、得分分布图、序列logo风格展示
"""
import json
from typing import Dict, List, Optional


AA_COLORS = {
    'G': '#f97316', 'A': '#f97316', 'V': '#f97316', 'L': '#f97316', 'I': '#f97316',
    'F': '#eab308', 'W': '#eab308', 'Y': '#eab308',
    'P': '#22c55e',
    'C': '#ef4444', 'M': '#ef4444',
    'S': '#3b82f6', 'T': '#3b82f6',
    'N': '#06b6d4', 'Q': '#06b6d4',
    'D': '#a855f7', 'E': '#a855f7',
    'K': '#ec4899', 'R': '#ec4899', 'H': '#ec4899',
}

AA_TYPE_LABELS = {
    'hydrophobic': {'aa': 'GAVLIFWY', 'color': '#f97316', 'label': '疏水'},
    'special': {'aa': 'PC', 'color': '#22c55e', 'label': '特殊'},
    'polar': {'aa': 'STNQ', 'color': '#3b82f6', 'label': '极性'},
    'charged_pos': {'aa': 'KRH', 'color': '#ec4899', 'label': '正电'},
    'charged_neg': {'aa': 'DE', 'color': '#a855f7', 'label': '负电'},
    'cysteine': {'aa': 'M', 'color': '#ef4444', 'label': '含硫'},
}


def render_mpnn_sequence_card(
    sequence: str,
    score: float = 0,
    design_idx: int = 0,
    seq_idx: int = 0,
    is_selected: bool = False
) -> str:
    selected_border = "border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,0.2);" if is_selected else ""
    score_color = "#059669" if score > 0.8 else "#d97706" if score > 0.5 else "#dc2626"

    seq_display = sequence[:80]
    if len(sequence) > 80:
        seq_display = sequence[:77] + "..."

    colored_seq = ""
    for i, aa in enumerate(seq_display):
        color = AA_COLORS.get(aa, '#64748b')
        colored_seq += f"<span style='color:{color};font-weight:600;font-size:13px;font-family:monospace;'>{aa}</span>"

    return f"""
    <div style='padding:14px;border:1px solid #e2e8f0;border-radius:10px;background:white;{selected_border}margin-bottom:8px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
            <span style='font-size:14px;font-weight:700;color:#1e293b;'>Design {design_idx + 1} - Seq {seq_idx + 1}</span>
            <span style='font-size:13px;font-weight:600;color:{score_color};background:{score_color}11;padding:3px 10px;border-radius:12px;'>得分: {score:.3f}</span>
        </div>
        <div style='line-height:1.8;letter-spacing:0.5px;word-break:break-all;'>{colored_seq}</div>
        <div style='font-size:11px;color:#94a3b8;margin-top:6px;'>长度: {len(sequence)} aa</div>
    </div>
    """


def render_mpnn_sequence_comparison(
    sequences: List[Dict],
    max_display: int = 5
) -> str:
    if not sequences:
        return "<div style='padding:20px;color:#94a3b8;text-align:center;'>暂无序列数据</div>"

    cards_html = ""
    for i, seq_data in enumerate(sequences[:max_display]):
        design_idx = seq_data.get("design_idx", 0)
        sequence = seq_data.get("sequence", "")
        score = seq_data.get("score", 0)
        cards_html += render_mpnn_sequence_card(sequence, score, design_idx, i)

    return f"""
    <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:12px;'>
        <div style='font-size:15px;font-weight:700;color:#1e293b;margin-bottom:12px;'>序列设计结果</div>
        {cards_html}
    </div>
    """


def render_mpnn_score_chart(sequences: List[Dict]) -> str:
    if not sequences:
        return ""

    scores = [s.get("score", 0) for s in sequences]
    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0

    bars_html = ""
    for i, score in enumerate(scores):
        height_pct = ((score - min_score + 0.01) / (max_score - min_score + 0.01)) * 100
        color = "#6366f1" if i == 0 else "#8b5cf6" if i < 3 else "#a78bfa"
        bars_html += f"""
        <div style='display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;max-width:60px;'>
            <span style='font-size:10px;color:#64748b;'>{score:.2f}</span>
            <div style='width:100%;height:{height_pct}%;background:{color};border-radius:4px 4px 0 0;min-height:4px;transition:height 0.3s;'></div>
            <span style='font-size:10px;color:#94a3b8;'>S{i+1}</span>
        </div>
        """

    return f"""
    <div style='padding:16px;background:white;border:1px solid #e2e8f0;border-radius:12px;margin-top:12px;'>
        <div style='font-size:14px;font-weight:700;color:#1e293b;margin-bottom:12px;'>得分分布</div>
        <div style='display:flex;align-items:flex-end;height:100px;gap:6px;padding:0 4px;border-bottom:1px solid #e2e8f0;'>
            {bars_html}
        </div>
    </div>
    """


def render_mpnn_legend() -> str:
    legend_items = ""
    for key, val in AA_TYPE_LABELS.items():
        legend_items += f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:11px;color:#64748b;'><span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{val['color']};'></span>{val['label']}({val['aa']})</span>"

    return f"""
    <div style='padding:10px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;margin-top:8px;'>
        <div style='font-size:11px;font-weight:600;color:#64748b;margin-bottom:6px;'>氨基酸类型图例</div>
        <div style='display:flex;flex-wrap:wrap;gap:4px;'>{legend_items}</div>
    </div>
    """
