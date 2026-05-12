# Preview-Decoupled Experiment Results Design

## Goal

Make DeepBinder experiment records the source of truth for complete backend-generated results, while letting the frontend show and interact with a small preview set without changing the real experiment data.

## Problem

The current task flow mixes two concepts:

- Formal experiment output: persisted data used for experiment records, reporting, export, and thesis evidence.
- Interactive frontend exploration: clicking a displayed design to run MPNN/RF3 or inspect structures.

Because those flows share the same endpoints and persistence path, frontend actions can overwrite or append to formal experiment data. Separately, the "top three confidence designs" section often has empty metrics because only the first RF3-validated sequence gets full RF3 metrics and other candidates are stored with partial or null values.

## Desired Behavior

When a user starts a design task, the backend should run the configured real workload. If the user requests eight batches, the backend records and tracks that full job. The frontend should show only a small preview, capped at two designs, for usability and visualization speed.

When the user clicks a preview design and runs MPNN/RF3 from the frontend, that operation is preview-only. It may update the current UI state, but it must not overwrite formal experiment results or add formal candidates unless a future explicit "save as official result" action is added.

The experiment detail page should show meaningful metrics for the confidence-ranked designs. RF3 metrics are used when available. For candidates not RF3-validated, the UI should display available fallback metrics, such as RFD3 pLDDT or MPNN ranking score, and clearly mark unavailable metrics as not calculated instead of appearing broken.

## Architecture

### Formal Experiment Data

Formal data remains attached to the experiment record:

- `rfd3_results`: full backend RFD3 output, including generated batch/design metadata that the backend preserves.
- `mpnn_results`: formal sequence design output when generated as part of the formal pipeline.
- `rf3_results`: formal structure validation output.
- `experiment_designs`: official candidate rows used by the detail page, export, and confidence ranking.

Only backend-owned task execution and explicit formal save operations write these fields.

### Preview Data

Preview data is derived from formal results and limited for frontend display:

- `preview_designs`: at most two RFD3 designs selected from full RFD3 output.
- `preview_summary`: planned batch count, planned design count, available preview count, and whether the backend is still running.

Preview data can be returned by the experiment detail endpoint or a helper transformation in the existing response. It should not require a new database table for the first implementation because it is derived from persisted formal results.

### Preview-Only Runs

Interactive frontend runs use one of these safe paths:

- Existing run endpoints accept a `preview_only` flag and skip `_save_experiment_step()` / `_save_designs()` when true.
- Or the frontend omits `experiment_id` for interactive runs so the backend has nothing formal to mutate.

The preferred first implementation is a `preview_only` flag because it documents intent at the API boundary and is testable.

## Candidate Metrics

Candidate metric construction should use a predictable merge order:

1. RF3 metrics for candidates that have been RF3 validated.
2. RFD3 design pLDDT for backbone confidence when RF3 pLDDT is unavailable.
3. MPNN sequence score for ranking when RF3 ranking score is unavailable.
4. Null only when no corresponding metric exists.

Each candidate should carry enough metadata for the UI to label metric provenance:

- `plddt_source`: `rf3`, `rfd3`, or `none`.
- `ranking_source`: `rf3`, `mpnn`, or `none`.
- `validation_status`: `validated`, `not_validated`, or `failed`.

If schema churn is too high for the first pass, the UI can infer provenance from existing fields, but backend-side explicit source fields are preferred.

## Frontend Behavior

### New Task Flow

The "新建任务" action should submit the full backend workload parameters. It should not reduce `n_batches`, `diffusion_batch_size`, or other real generation settings for frontend convenience.

After submission, the UI should show task status and a preview panel. The panel may display only two designs, but the status text should make the distinction visible:

- "后台按完整参数生成：8 batch"
- "当前仅展示 2 个预览 design"

### Preview Interaction

For preview designs:

- "送入 MPNN" runs preview-only sequence generation.
- "送入 RF3 验证" runs preview-only validation.
- Results update the local page state and visualization components.
- Formal experiment data remains unchanged.

### Experiment Detail Page

The "置信度最高的三个设计结果" section should:

- Sort by the best available confidence/ranking metric.
- Prefer validated RF3 candidates.
- Show fallback metrics with labels when RF3 metrics are unavailable.
- Avoid showing empty metric cells without explanation.

## API and Data Flow

Recommended flow:

1. Frontend submits full parameters to `POST /api/run-rfd3` or pipeline endpoint.
2. Backend creates/updates a formal experiment and runs the full configured workload.
3. Backend persists full formal results.
4. Experiment detail response includes full formal result summary and a capped preview list.
5. Frontend renders only the capped preview list in the interactive result panel.
6. Frontend calls MPNN/RF3 with `preview_only=true` for interactive preview operations.
7. Backend returns preview results without writing formal experiment fields.

## Error Handling

- If preview-only MPNN/RF3 fails, show the error in the preview panel only.
- Do not mark the formal experiment as failed because a preview-only click failed.
- If the formal backend task fails, keep the formal experiment status as `failed` and display the failed step.
- If a candidate lacks RF3 metrics, display "未验证" or "未计算", not a blank metric.

## Testing

Backend tests:

- Preview-only MPNN/RF3 does not call formal save helpers.
- Candidate metric building fills fallback pLDDT/ranking fields when RF3 metrics are absent.
- Full task parameters are preserved in formal RFD3 config.
- Preview list is capped at two without truncating persisted formal results.

Frontend tests:

- Experiment detail top-three cards show fallback metric values and unavailable labels.
- RFD3 preview panel displays only two designs while showing the full planned/generated count.
- Preview-only MPNN/RF3 calls include `preview_only=true`.
- Preview-only results update visualization state without navigating away or implying formal save.

## Scope

In scope:

- Fix candidate metrics for confidence-ranked design cards.
- Add preview-only API semantics for interactive runs.
- Cap frontend preview rendering to two designs.
- Preserve full backend generation parameters and formal result persistence.

Out of scope:

- A persistent preview session table.
- Background job queue migration.
- A user-facing "save preview as official candidate" workflow.
- Large UI redesign of the job center.
