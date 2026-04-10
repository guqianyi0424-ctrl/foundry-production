"""
Mol* 3D可视化模块
支持热点残基高亮、Binder叠加、RMSD着色、序列-3D联动
"""
import base64
import json
from typing import Dict, List, Optional


def render_molstar(
    pdb_content: str,
    hotspot_residues: Optional[List[Dict]] = None,
    rfd3_pdb: Optional[str] = None,
    rf3_pdb: Optional[str] = None,
    rmsd_data: Optional[List[float]] = None,
    height: int = 650,
    target_color: str = "#4CAF50",
    binder_color: str = "#FF5722",
    rf3_color: str = "#2196F3"
) -> str:
    pdb_b64 = base64.b64encode(pdb_content.encode()).decode()

    hotspot_js = "const hotspotResidues = [];"
    if hotspot_residues:
        entries = []
        for h in hotspot_residues:
            chain = h.get("chain", "A")
            res_id = h.get("residue_id", 0)
            score = h.get("score", 0)
            try:
                res_id_int = int(str(res_id).strip())
            except (ValueError, TypeError):
                res_id_int = 0
            entries.append(f'{{chain: "{chain}", resId: {res_id_int}, score: {float(score):.3f}}}')
        hotspot_js = f"const hotspotResidues = [{', '.join(entries)}];"

    rfd3_b64 = ""
    if rfd3_pdb:
        rfd3_b64 = base64.b64encode(rfd3_pdb.encode()).decode()

    rf3_b64 = ""
    if rf3_pdb:
        rf3_b64 = base64.b64encode(rf3_pdb.encode()).decode()

    rmsd_js = "const rmsdData = null;"
    if rmsd_data:
        rmsd_js = f"const rmsdData = {json.dumps(rmsd_data)};"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            #viewer {{ width: 100%; height: {height}px; position: relative; }}
            #statusBar {{
                position: absolute; bottom: 0; left: 0; right: 0;
                background: rgba(0,0,0,0.85); color: #fff;
                padding: 6px 12px; font-size: 13px; font-family: monospace;
                display: flex; gap: 16px; align-items: center; z-index: 10;
            }}
            .status-item {{ display: flex; align-items: center; gap: 4px; }}
            .legend {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; }}
        </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.css">
    </head>
    <body>
        <div id="viewer">
            <div id="statusBar">
                <span class="status-item"><span class="legend" style="background:{target_color}"></span>Target</span>
                {"<span class='status-item'><span class='legend' style='background:" + binder_color + "'></span>RFD3 Binder</span>" if rfd3_pdb else ""}
                {"<span class='status-item'><span class='legend' style='background:" + rf3_color + "'></span>RF3 Prediction</span>" if rf3_pdb else ""}
                <span class="status-item"><span class="legend" style="background:#FF0000"></span>Hotspot</span>
                <span id="resInfo" style="margin-left:auto;"></span>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script>
        <script>
            {hotspot_js}
            {rmsd_js}
            const rfd3B64 = "{rfd3_b64}";
            const rf3B64 = "{rf3_b64}";

            molstar.Viewer.create("viewer", {{
                layoutIsExpanded: false,
                layoutShowControls: false,
                layoutShowRemoteState: false,
                layoutShowSequence: true,
                layoutShowLog: false,
                layoutShowLeftPanel: false,
            }}).then(async viewer => {{
                const plugin = viewer;

                try {{
                    const targetData = atob("{pdb_b64}");
                    const targetTraj = await plugin.builders.structure.readTrajectory({{
                        model: {{ type: 'pdb', data: targetData }}
                    }});
                    const targetPreset = await plugin.builders.structure.hierarchy.applyPreset(
                        {{ structure: targetTraj }},
                        'default'
                    );

                    const targetRepr = targetPreset.structure.representations[0];
                    if (targetRepr) {{
                        await plugin.managers.structure.component.updateRepresentationsOptions(
                            targetRepr,
                            {{ color: {{ name: 'uniform', params: {{ value: '{target_color}' }} }} }}
                        );
                    }}

                    if (rfd3B64) {{
                        try {{
                            const rfd3Data = atob(rfd3B64);
                            const rfd3Traj = await plugin.builders.structure.readTrajectory({{
                                model: {{ type: 'pdb', data: rfd3Data }}
                            }});
                            const rfd3Preset = await plugin.builders.structure.hierarchy.applyPreset(
                                {{ structure: rfd3Traj }},
                                'default'
                            );
                            const rfd3Repr = rfd3Preset.structure.representations[0];
                            if (rfd3Repr) {{
                                await plugin.managers.structure.component.updateRepresentationsOptions(
                                    rfd3Repr,
                                    {{ color: {{ name: 'uniform', params: {{ value: '{binder_color}' }} }} }}
                                );
                            }}
                        }} catch(e) {{ console.error("RFD3 load error:", e); }}
                    }}

                    if (rf3B64) {{
                        try {{
                            const rf3Data = atob(rf3B64);
                            const rf3Traj = await plugin.builders.structure.readTrajectory({{
                                model: {{ type: 'pdb', data: rf3Data }}
                            }});
                            const rf3Preset = await plugin.builders.structure.hierarchy.applyPreset(
                                {{ structure: rf3Traj }},
                                'default'
                            );
                            const rf3Repr = rf3Preset.structure.representations[0];
                            if (rf3Repr) {{
                                await plugin.managers.structure.component.updateRepresentationsOptions(
                                    rf3Repr,
                                    {{
                                        color: {{ name: 'uniform', params: {{ value: '{rf3_color}' }} }},
                                        alpha: 0.6
                                    }}
                                );
                            }}
                        }} catch(e) {{ console.error("RF3 load error:", e); }}
                    }}

                    if (hotspotResidues.length > 0) {{
                        try {{
                            const structures = plugin.managers.structure.hierarchy.current.structures;
                            if (structures.length > 0) {{
                                const targetStruct = structures[0];
                                const sel = plugin.managers.structure.selection;
                                const comp = plugin.managers.structure.component;

                                for (const h of hotspotResidues) {{
                                    try {{
                                        const script = molstar.Script(
                                            "sel.atom: " +
                                            "(chain.authAsymId = " + JSON.stringify(h.chain) + " or chain.labelAsymId = " + JSON.stringify(h.chain) + ") and " +
                                            "(residue.authSeqNumber = " + h.resId + " or residue.labelSeqNumber = " + h.resId + ")"
                                        );
                                        const selData = await plugin.managers.structure.selection.fromScript(targetStruct, script);
                                        if (selData) {{
                                            await comp.addRepresentation(targetStruct, 'ball-and-stick', {{
                                                color: {{ name: 'uniform', params: {{ value: '#FF0000' }} }},
                                                sizeFactor: 0.3
                                            }}, selData);
                                        }}
                                    }} catch(e2) {{ console.warn("Hotspot highlight error:", e2); }}
                                }}
                            }}
                        }} catch(e) {{ console.warn("Hotspot section error:", e); }}
                    }}

                    plugin.managers.camera.resetSnapshot();

                }} catch(e) {{
                    console.error("Mol* error:", e);
                    document.getElementById("resInfo").textContent = "Error: " + e.message;
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html


def render_rmsd_chart(rmsd_values: List[float], threshold: float = 2.0) -> str:
    rmsd_json = json.dumps(rmsd_values)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            .chart-container {{ width: 100%; height: 300px; }}
        </style>
    </head>
    <body>
        <div class="chart-container">
            <canvas id="rmsdChart"></canvas>
        </div>
        <script>
            const rmsdData = {rmsd_json};
            const threshold = {threshold};

            const colors = rmsdData.map(v => v < threshold ? 'rgba(76,175,80,0.7)' : 'rgba(244,67,54,0.7)');

            new Chart(document.getElementById('rmsdChart'), {{
                type: 'bar',
                data: {{
                    labels: rmsdData.map((_, i) => i + 1),
                    datasets: [{{
                        label: 'RMSD (Å)',
                        data: rmsdData,
                        backgroundColor: colors,
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Per-Residue RMSD Distribution',
                            font: {{ size: 14 }}
                        }},
                        annotation: {{
                            annotations: {{
                                thresholdLine: {{
                                    type: 'line',
                                    yMin: threshold,
                                    yMax: threshold,
                                    borderColor: 'rgba(255,0,0,0.6)',
                                    borderWidth: 2,
                                    borderDash: [6, 6],
                                    label: {{
                                        display: true,
                                        content: 'Threshold ({threshold} Å)',
                                        position: 'end'
                                    }}
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            title: {{ display: true, text: 'RMSD (Å)' }},
                            min: 0,
                            suggestedMax: Math.max(threshold * 2, ...rmsdData) * 1.1
                        }},
                        x: {{
                            title: {{ display: true, text: 'Residue Index' }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html


def render_plddt_chart(plddt_values: List[float]) -> str:
    plddt_json = json.dumps(plddt_values)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            .chart-container {{ width: 100%; height: 250px; }}
        </style>
    </head>
    <body>
        <div class="chart-container">
            <canvas id="plddtChart"></canvas>
        </div>
        <script>
            const plddtData = {plddt_json};

            const colors = plddtData.map(v => {{
                if (v >= 90) return 'rgba(0,100,255,0.7)';
                if (v >= 70) return 'rgba(0,200,150,0.7)';
                if (v >= 50) return 'rgba(255,200,0,0.7)';
                return 'rgba(255,50,50,0.7)';
            }});

            new Chart(document.getElementById('plddtChart'), {{
                type: 'bar',
                data: {{
                    labels: plddtData.map((_, i) => i + 1),
                    datasets: [{{
                        label: 'pLDDT',
                        data: plddtData,
                        backgroundColor: colors,
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'pLDDT Confidence Score',
                            font: {{ size: 14 }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            title: {{ display: true, text: 'pLDDT' }},
                            min: 0,
                            max: 100
                        }},
                        x: {{
                            title: {{ display: true, text: 'Residue Index' }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html


def render_pae_heatmap(pae_data: List[List[float]], n_res: int = 0) -> str:
    pae_json = json.dumps(pae_data)
    if n_res == 0 and pae_data:
        n_res = len(pae_data)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            .chart-container {{ width: 100%; height: 400px; }}
        </style>
    </head>
    <body>
        <div class="chart-container">
            <canvas id="paeChart"></canvas>
        </div>
        <script>
            const paeData = {pae_json};
            const nRes = {n_res};

            const labels = Array.from({{length: nRes}}, (_, i) => i + 1);

            new Chart(document.getElementById('paeChart'), {{
                type: 'matrix',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'PAE',
                        data: paeData.flatMap((row, i) =>
                            row.map((val, j) => ({{ x: j + 1, y: i + 1, v: val }}))
                        ),
                        backgroundColor(ctx) {{
                            const v = ctx.dataset.data[ctx.dataIndex]?.v;
                            if (v === undefined) return 'rgba(0,0,0,0)';
                            const t = Math.min(v / 30, 1);
                            const r = Math.round(t * 255);
                            const b = Math.round((1 - t) * 255);
                            return `rgba(${{r}}, 0, ${{b}}, 0.8)`;
                        }},
                        width: ({{chart}}) => (chart.chartArea || {{}}).width / nRes,
                        height: ({{chart}}) => (chart.chartArea || {{}}).height / nRes
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Predicted Aligned Error (PAE)',
                            font: {{ size: 14 }}
                        }}
                    }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Scored Residue' }} }},
                        y: {{ title: {{ display: true, text: 'Aligned Residue' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html
