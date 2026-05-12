# Preview-Decoupled Experiment Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve complete backend experiment generation while making frontend preview interactions display only two designs and never mutate formal experiment data.

**Architecture:** Add explicit preview-only semantics to manual MPNN/RF3 endpoints, improve formal candidate metric construction with fallback metrics, and cap frontend RFD3 preview rendering at two designs. Keep formal experiment rows and persisted full RFD3 results unchanged; frontend preview results live in component state only.

**Tech Stack:** FastAPI, SQLAlchemy, pytest, React, TypeScript, Vitest, Testing Library.

---

## File Structure

- Modify `binder-design-system/backend/services/design_pipeline.py`: build official candidate rows with fallback metrics and metric provenance fields.
- Modify `binder-design-system/backend/routers/design.py`: add `preview_only` request fields and skip formal writes for preview-only MPNN/RF3 runs.
- Modify `binder-design-system/backend/repositories/experiments.py`: persist optional metric provenance fields if database columns are added.
- Modify `binder-design-system/backend/database.py`: add optional provenance columns to `ExperimentDesign`.
- Modify `binder-design-system/backend/routers/experiments.py`: expose provenance fields in detail/export payloads.
- Modify `binder-design-system/frontend/src/api/index.ts`: add `preview_only`, provenance fields, and optional preview metadata types.
- Modify `binder-design-system/frontend/src/pages/NewDesignPage.tsx`: cap displayed RFD3 designs at two and send preview-only flags for manual MPNN/RF3.
- Modify `binder-design-system/frontend/src/pages/ExperimentDetailPage.tsx`: rank top candidates by best available metric and display metric provenance/fallback labels.
- Modify tests in `binder-design-system/backend/tests/` and `binder-design-system/frontend/src/pages/__tests__/acceptance.test.tsx`.

## Task 1: Backend Candidate Metrics With Provenance

**Files:**
- Modify: `binder-design-system/backend/database.py`
- Modify: `binder-design-system/backend/repositories/experiments.py`
- Modify: `binder-design-system/backend/routers/experiments.py`
- Modify: `binder-design-system/backend/services/design_pipeline.py`
- Test: `binder-design-system/backend/tests/test_design_pipeline_service.py`
- Test: `binder-design-system/backend/tests/test_experiment_service.py`

- [ ] **Step 1: Write failing pipeline candidate metric test**

Add a second sequence and second RFD3 design to `SuccessfulRFD3` and `SuccessfulMPNN` in `binder-design-system/backend/tests/test_design_pipeline_service.py`, then update `test_pipeline_service_runs_all_steps` to expect fallback metrics:

```python
class SuccessfulRFD3:
    def run(self, config):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "first_backbone_pdb": "RFD3_PDB",
                "designs": [
                    {"name": "rfd3_0", "pdb_content": "RFD3_PDB", "plddt": 88.0},
                    {"name": "rfd3_1", "pdb_content": "RFD3_PDB_1", "plddt": 77.5},
                ],
            },
        )


class SuccessfulMPNN:
    def run(self, **kwargs):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "first_sequence_pdb": "MPNN_PDB",
                "sequences": [
                    {
                        "name": "seq_0",
                        "sequence": "ACD",
                        "pdb_content": "MPNN_PDB",
                        "score": -1.0,
                    },
                    {
                        "name": "seq_1",
                        "sequence": "EFG",
                        "pdb_content": "MPNN_PDB_1",
                        "score": -2.5,
                    },
                ],
            },
        )
```

Update the expected `experiments.designs` assertion:

```python
assert experiments.designs == [
    (
        "exp_1",
        [
            {
                "name": "seq_0",
                "sequence": "ACD",
                "pdb_content": "MPNN_PDB",
                "plddt": 90.0,
                "rmsd": 1.1,
                "ranking_score": -1.0,
                "passed_validation": True,
                "plddt_source": "rf3",
                "ranking_source": "mpnn",
                "validation_status": "validated",
            },
            {
                "name": "seq_1",
                "sequence": "EFG",
                "pdb_content": "MPNN_PDB_1",
                "plddt": 77.5,
                "rmsd": None,
                "ranking_score": -2.5,
                "passed_validation": False,
                "plddt_source": "rfd3",
                "ranking_source": "mpnn",
                "validation_status": "not_validated",
            },
        ],
    )
]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/pytest binder-design-system/backend/tests/test_design_pipeline_service.py::test_pipeline_service_runs_all_steps -q
```

Expected: FAIL because candidate dictionaries do not include `plddt_source`, `ranking_source`, and `validation_status`, and the second candidate pLDDT is currently `None`.

- [ ] **Step 3: Add provenance columns and repository persistence**

In `binder-design-system/backend/database.py`, add nullable string columns to `ExperimentDesign`:

```python
    plddt_source = Column(String, nullable=True)
    ranking_source = Column(String, nullable=True)
    validation_status = Column(String, nullable=True)
```

In `binder-design-system/backend/repositories/experiments.py`, update `save_designs()`:

```python
                        plddt_source=data.get("plddt_source"),
                        ranking_source=data.get("ranking_source"),
                        validation_status=data.get("validation_status"),
```

Also update `get_archive_payload()` design dictionaries:

```python
                        "plddt_source": design.plddt_source,
                        "ranking_source": design.ranking_source,
                        "validation_status": design.validation_status,
```

In `binder-design-system/backend/routers/experiments.py`, update every design dictionary returned from experiment details/export payload builders to include:

```python
            "plddt_source": getattr(d, "plddt_source", None),
            "ranking_source": getattr(d, "ranking_source", None),
            "validation_status": getattr(d, "validation_status", None),
```

- [ ] **Step 4: Implement fallback metric builder**

In `binder-design-system/backend/services/design_pipeline.py`, change the save call:

```python
self.experiment_service.save_designs(
    experiment_id,
    self._build_candidate_designs(
        mpnn_result.data or {},
        rf3_result.data or {},
        rfd3_result.data or {},
    ),
)
```

Replace `_build_candidate_designs()` with:

```python
    def _build_candidate_designs(
        self,
        mpnn_data: dict[str, Any],
        rf3_data: dict[str, Any],
        rfd3_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        sequences = mpnn_data.get("sequences") or []
        if not sequences:
            return []

        rfd3_designs = (rfd3_data or {}).get("designs") or []
        rf3_ranking = (rf3_data.get("summary") or {}).get("ranking_score")
        candidates = []
        for index, sequence in enumerate(sequences):
            is_validated_sequence = index == 0
            rfd3_design = rfd3_designs[index] if index < len(rfd3_designs) else {}
            fallback_plddt = rfd3_design.get("plddt")
            sequence_score = sequence.get("score")

            if is_validated_sequence and rf3_data.get("avg_plddt") is not None:
                plddt = rf3_data.get("avg_plddt")
                plddt_source = "rf3"
            elif fallback_plddt is not None:
                plddt = fallback_plddt
                plddt_source = "rfd3"
            else:
                plddt = None
                plddt_source = "none"

            if is_validated_sequence and rf3_ranking is not None:
                ranking_score = rf3_ranking
                ranking_source = "rf3"
            elif sequence_score is not None:
                ranking_score = sequence_score
                ranking_source = "mpnn"
            else:
                ranking_score = None
                ranking_source = "none"

            candidates.append(
                {
                    "name": sequence.get("name") or f"candidate_{index + 1:03d}",
                    "sequence": sequence.get("sequence", ""),
                    "pdb_content": sequence.get("pdb_content", ""),
                    "plddt": plddt,
                    "rmsd": rf3_data.get("rmsd") if is_validated_sequence else None,
                    "ranking_score": ranking_score,
                    "passed_validation": bool(rf3_data.get("passed", False))
                    if is_validated_sequence
                    else False,
                    "plddt_source": plddt_source,
                    "ranking_source": ranking_source,
                    "validation_status": "validated"
                    if is_validated_sequence
                    else "not_validated",
                }
            )
        return candidates
```

- [ ] **Step 5: Add repository persistence test**

In `binder-design-system/backend/tests/test_experiment_service.py`, update `test_experiment_service_saves_design_records` input with provenance:

```python
                "plddt_source": "rfd3",
                "ranking_source": "mpnn",
                "validation_status": "not_validated",
```

Then assert:

```python
        assert design.plddt_source == "rfd3"
        assert design.ranking_source == "mpnn"
        assert design.validation_status == "not_validated"
```

- [ ] **Step 6: Run backend metric tests**

Run:

```bash
.venv/bin/pytest binder-design-system/backend/tests/test_design_pipeline_service.py binder-design-system/backend/tests/test_experiment_service.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit backend candidate metrics**

Run:

```bash
git add binder-design-system/backend/database.py binder-design-system/backend/repositories/experiments.py binder-design-system/backend/routers/experiments.py binder-design-system/backend/services/design_pipeline.py binder-design-system/backend/tests/test_design_pipeline_service.py binder-design-system/backend/tests/test_experiment_service.py
git commit -m "Fix formal candidate metric fallbacks"
```

## Task 2: Preview-Only MPNN/RF3 API Writes

**Files:**
- Modify: `binder-design-system/backend/routers/design.py`
- Test: `binder-design-system/backend/tests/test_design_router_contract.py`

- [ ] **Step 1: Write failing preview-only router test**

In `binder-design-system/backend/tests/test_design_router_contract.py`, add fake MPNN and RF3 services:

```python
class FakeMPNNService:
    def run(self, **kwargs):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "sequences": [
                    {
                        "index": 0,
                        "name": "seq_0",
                        "sequence": "ACD",
                        "pdb_content": "MPNN_PDB",
                        "score": -1.0,
                    }
                ],
                "num_sequences": 1,
                "first_sequence_pdb": "MPNN_PDB",
                "output_dir": "",
            },
        )


class FakeRF3Service:
    def run(self, **kwargs):
        return AdapterResult(
            success=True,
            data={
                "success": True,
                "predicted_pdb": "RF3_PDB",
                "predicted_pdb_path": "",
                "num_models": 1,
                "summary": {"ranking_score": 0.8},
                "pae": None,
                "plddt": [90.0],
                "avg_plddt": 90.0,
                "rmsd": 1.0,
                "rmsd_interpretation": "Excellent",
                "per_res_rmsd": [1.0],
                "passed": True,
                "output_dir": "",
            },
        )
```

Add them to `FakeServices`:

```python
    mpnn = FakeMPNNService()
    rf3 = FakeRF3Service()
```

Add a test that patches save helpers and verifies they are not called:

```python
def test_preview_only_mpnn_and_rf3_do_not_persist(monkeypatch):
    design_router = patch_services(monkeypatch)
    persisted = []
    monkeypatch.setattr(
        design_router,
        "_save_experiment_step",
        lambda *args, **kwargs: persisted.append(("step", args, kwargs)),
    )
    monkeypatch.setattr(
        design_router,
        "_save_designs",
        lambda *args, **kwargs: persisted.append(("designs", args, kwargs)),
    )

    mpnn = asyncio.run(
        design_router.run_mpnn(
            design_router.MPNNRequest(
                backbone_pdb_content="RFD3_PDB",
                experiment_id="exp_formal",
                preview_only=True,
            )
        )
    )
    rf3 = asyncio.run(
        design_router.run_rf3(
            design_router.RF3Request(
                mpnn_pdb_content="MPNN_PDB",
                rfd3_pdb_content="RFD3_PDB",
                experiment_id="exp_formal",
                preview_only=True,
            )
        )
    )

    assert mpnn["success"] is True
    assert rf3["success"] is True
    assert persisted == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/pytest binder-design-system/backend/tests/test_design_router_contract.py::test_preview_only_mpnn_and_rf3_do_not_persist -q
```

Expected: FAIL because `MPNNRequest` and `RF3Request` do not accept `preview_only`.

- [ ] **Step 3: Add preview_only fields**

In `binder-design-system/backend/routers/design.py`, add fields:

```python
class MPNNRequest(BaseModel):
    backbone_pdb_content: Optional[str] = None
    backbone_pdb_path: Optional[str] = None
    batch_size: int = 10
    fixed_chains: Optional[List[str]] = None
    model_type: str = "ligand_mpnn"
    experiment_id: Optional[str] = None
    preview_only: bool = False


class RF3Request(BaseModel):
    mpnn_pdb_content: str
    rfd3_pdb_content: Optional[str] = None
    example_id: str = "binder_design"
    experiment_id: Optional[str] = None
    preview_only: bool = False
```

- [ ] **Step 4: Skip formal writes when preview_only is true**

In `run_mpnn()`, replace the persistence block with:

```python
        if not req.preview_only:
            _save_experiment_step(req.experiment_id, "mpnn", sanitize_model_result(result), config)

            if result.get("success") and result.get("sequences"):
                designs = []
                for s in result["sequences"]:
                    designs.append({
                        "name": s.get("name", f"seq_{s.get('index', 0)}"),
                        "sequence": s.get("sequence", ""),
                        "pdb_content": s.get("pdb_content", ""),
                        "ranking_score": s.get("score"),
                        "ranking_source": "mpnn" if s.get("score") is not None else "none",
                        "validation_status": "not_validated",
                    })
                _save_designs(req.experiment_id, designs)
```

In `run_rf3()`, wrap `_save_experiment_step()` and the duration update with:

```python
        if not req.preview_only:
            _save_experiment_step(req.experiment_id, "rf3", sanitize_model_result(result), config)

        if result.get("success") and req.experiment_id and not req.preview_only:
            ...
```

- [ ] **Step 5: Run preview-only backend tests**

Run:

```bash
.venv/bin/pytest binder-design-system/backend/tests/test_design_router_contract.py::test_preview_only_mpnn_and_rf3_do_not_persist -q
```

Expected: pass.

- [ ] **Step 6: Commit preview-only backend API**

Run:

```bash
git add binder-design-system/backend/routers/design.py binder-design-system/backend/tests/test_design_router_contract.py
git commit -m "Add preview-only design run semantics"
```

## Task 3: Frontend Preview Limit and Preview-Only Calls

**Files:**
- Modify: `binder-design-system/frontend/src/api/index.ts`
- Modify: `binder-design-system/frontend/src/pages/NewDesignPage.tsx`
- Test: `binder-design-system/frontend/src/pages/__tests__/acceptance.test.tsx`

- [ ] **Step 1: Write failing frontend preview limit test**

In `binder-design-system/frontend/src/pages/__tests__/acceptance.test.tsx`, add a test after `sends the clicked protein-to-protein RFD3 design into MPNN`:

```tsx
  it('shows only two RFD3 preview designs and sends preview-only MPNN/RF3 calls', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    useAppStore.getState().setPdbContent('TARGET_PDB')
    useAppStore.getState().setRfd3Config({
      targetStructure: 'A/1-100',
      hotspots: 'A/42',
    })
    mockRunRFD3.mockResolvedValue({
      success: true,
      experiment_id: 'exp-preview',
      designs: [
        { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
        { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
        { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
      ],
      batches: [
        {
          batch_idx: 0,
          num_structures: 3,
          designs: [
            { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
            { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
            { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
          ],
        },
      ],
      num_batches: 1,
      num_designs: 3,
      first_backbone_pdb: 'RFD3_0',
      output_dir: '',
    })
    mockRunMPNN.mockResolvedValue({
      success: true,
      sequences: [
        { index: 0, name: 'seq_0', sequence: 'ACDEFGHIK', pdb_path: '', pdb_content: 'MPNN_PDB', score: -0.3 },
      ],
      num_sequences: 1,
      first_sequence_pdb: 'MPNN_PDB',
      output_dir: '',
    })
    mockRunRF3.mockResolvedValue({
      success: true,
      predicted_pdb: 'RF3_PDB',
      predicted_pdb_path: '',
      num_models: 1,
      summary: {
        chain_ptm: [0.8],
        overall_plddt: 91.2,
        overall_pde: 0.2,
        overall_pae: 1.3,
        ptm: 0.81,
        iptm: 0.78,
        has_clash: false,
        ranking_score: 0.85,
      },
      pae: null,
      plddt: [91.2],
      avg_plddt: 91.2,
      rmsd: 1.35,
      rmsd_interpretation: 'Excellent',
      per_res_rmsd: [1.0],
      passed: true,
      output_dir: '',
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: '新建任务' }))
    await user.click(screen.getByRole('button', { name: '开始生成' }))
    await user.click(await screen.findByRole('button', { name: /① RFD3 骨架生成/ }))

    expect(screen.getByText('rfd3_0')).toBeInTheDocument()
    expect(screen.getByText('rfd3_1')).toBeInTheDocument()
    expect(screen.queryByText('rfd3_2')).not.toBeInTheDocument()
    expect(screen.getByText(/当前仅展示 2 个预览 design/)).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: /送入MPNN/ })[0])
    await screen.findByText('Sequence 1')
    expect(mockRunMPNN).toHaveBeenCalledWith({
      backbone_pdb_content: 'RFD3_0',
      batch_size: 10,
      fixed_chains: ['A'],
      experiment_id: 'exp-preview',
      preview_only: true,
    })

    await user.click(screen.getByRole('button', { name: /送入RF3验证/ }))
    await screen.findByText('验证通过 ✅')
    expect(mockRunRF3).toHaveBeenCalledWith({
      mpnn_pdb_content: 'MPNN_PDB',
      rfd3_pdb_content: 'RFD3_0',
      example_id: 'binder_design',
      experiment_id: 'exp-preview',
      preview_only: true,
    })
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd binder-design-system/frontend
npm test -- --run src/pages/__tests__/acceptance.test.tsx -t "shows only two RFD3 preview designs"
```

Expected: FAIL because all three RFD3 designs render and preview-only flags are not sent.

- [ ] **Step 3: Add API types**

In `binder-design-system/frontend/src/api/index.ts`, add optional fields:

```ts
export interface RFD3Response {
  success: boolean
  experiment_id?: string
  designs: RFD3Design[]
  batches: Array<{ batch_idx: number; num_structures: number; designs: RFD3Design[] }>
  num_batches: number
  num_designs: number
  first_backbone_pdb: string
  output_dir: string
  mock?: boolean
}
```

Update `runRF3` params:

```ts
  experiment_id?: string
  preview_only?: boolean
```

Update `runMPNN` params:

```ts
  experiment_id?: string
  preview_only?: boolean
```

- [ ] **Step 4: Cap RFD3 preview rendering to two**

In `binder-design-system/frontend/src/pages/NewDesignPage.tsx`, add near constants:

```ts
const PREVIEW_DESIGN_LIMIT = 2
```

Add helper before the component:

```ts
const previewBatches = (results: RFD3Response | null) => {
  if (!results?.batches) return []
  let remaining = PREVIEW_DESIGN_LIMIT
  return results.batches
    .map((batch) => {
      const designs = batch.designs.slice(0, Math.max(remaining, 0))
      remaining -= designs.length
      return { ...batch, designs }
    })
    .filter((batch) => batch.designs.length > 0)
}
```

Inside `NewDesignPage`, add:

```ts
  const visibleRFD3Batches = previewBatches(rfd3Results)
  const previewDesignCount = visibleRFD3Batches.reduce((total, batch) => total + batch.designs.length, 0)
```

Replace:

```tsx
{rfd3Results.batches?.map((batch) => (
```

with:

```tsx
{visibleRFD3Batches.map((batch) => (
```

Add preview text after the generated count line:

```tsx
                    <span className="ml-2 text-gray-400">
                      当前仅展示 {previewDesignCount} 个预览 design，不影响后台完整生成
                    </span>
```

- [ ] **Step 5: Send preview_only flags from frontend**

In `handleRunMPNN`, include experiment ID and preview flag:

```ts
      const res = await runMPNN({
        backbone_pdb_content: backbonePdb,
        batch_size: 10,
        fixed_chains: targetChains.length > 0 ? targetChains : undefined,
        experiment_id: rfd3Results?.experiment_id,
        preview_only: true,
      })
```

In `handleRunRF3`, include experiment ID and preview flag:

```ts
      const res = await runRF3({
        mpnn_pdb_content: mpnnPdb,
        rfd3_pdb_content: rfd3Pdb,
        example_id: 'binder_design',
        experiment_id: rfd3Results?.experiment_id,
        preview_only: true,
      })
```

Update `handleRunRF3` dependency array to include `rfd3Results?.experiment_id` or `rfd3Results`.

- [ ] **Step 6: Run frontend preview test**

Run:

```bash
cd binder-design-system/frontend
npm test -- --run src/pages/__tests__/acceptance.test.tsx -t "shows only two RFD3 preview designs"
```

Expected: pass.

- [ ] **Step 7: Commit frontend preview behavior**

Run:

```bash
git add binder-design-system/frontend/src/api/index.ts binder-design-system/frontend/src/pages/NewDesignPage.tsx binder-design-system/frontend/src/pages/__tests__/acceptance.test.tsx
git commit -m "Limit frontend RFD3 previews"
```

## Task 4: Experiment Detail Confidence Cards

**Files:**
- Modify: `binder-design-system/frontend/src/api/index.ts`
- Modify: `binder-design-system/frontend/src/pages/ExperimentDetailPage.tsx`
- Test: `binder-design-system/frontend/src/pages/__tests__/acceptance.test.tsx`

- [ ] **Step 1: Write failing top-three metric display test**

In `binder-design-system/frontend/src/pages/__tests__/acceptance.test.tsx`, add a test near the existing experiment detail tests:

```tsx
  it('shows fallback metric provenance in top confidence design cards', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    mockGetExperiment.mockResolvedValue({
      id: 'exp-candidates',
      name: '候选指标实验',
      status: 'completed',
      created_at: '2026-05-10T12:00:00',
      updated_at: '2026-05-10T12:10:00',
      input_pdb: 'ATOM',
      target: 'A/1-2',
      hotspots: [{ chain: 'A', residue: 1 }],
      rfd3_config: { task_type: 'protein', protein_chain: 'A/1-2' },
      mpnn_config: null,
      rf3_config: null,
      rfd3_results: null,
      mpnn_results: null,
      rf3_results: { summary: {} },
      duration_seconds: 10,
      gpu_info: null,
      user_id: 'u-researcher',
      num_designs: 3,
      designs: [
        {
          id: 'd1',
          design_name: 'validated',
          sequence: 'ACD',
          pdb_content: 'ATOM1',
          plddt: 90,
          rmsd: 1.2,
          ranking_score: 0.8,
          passed_validation: true,
          plddt_source: 'rf3',
          ranking_source: 'rf3',
          validation_status: 'validated',
        },
        {
          id: 'd2',
          design_name: 'fallback',
          sequence: 'EFG',
          pdb_content: 'ATOM2',
          plddt: 78.5,
          rmsd: null,
          ranking_score: -2.5,
          passed_validation: false,
          plddt_source: 'rfd3',
          ranking_source: 'mpnn',
          validation_status: 'not_validated',
        },
      ],
    })

    useAppStore.getState().setCurrentPage('experiment_exp-candidates')
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('候选指标实验')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /设计结果/ }))

    expect(screen.getByText('validated')).toBeInTheDocument()
    expect(screen.getByText('fallback')).toBeInTheDocument()
    expect(screen.getByText(/pLDDT 90.0/)).toBeInTheDocument()
    expect(screen.getByText(/RF3/)).toBeInTheDocument()
    expect(screen.getByText(/pLDDT 78.5/)).toBeInTheDocument()
    expect(screen.getByText(/RFD3/)).toBeInTheDocument()
    expect(screen.getByText(/未验证/)).toBeInTheDocument()
    expect(screen.getByText(/MPNN/)).toBeInTheDocument()
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd binder-design-system/frontend
npm test -- --run src/pages/__tests__/acceptance.test.tsx -t "shows fallback metric provenance"
```

Expected: FAIL because the UI does not render provenance labels.

- [ ] **Step 3: Update frontend types**

In `binder-design-system/frontend/src/api/index.ts`, add optional fields to `ExperimentDetail.designs` items:

```ts
    plddt_source?: 'rf3' | 'rfd3' | 'none' | null
    ranking_source?: 'rf3' | 'mpnn' | 'none' | null
    validation_status?: 'validated' | 'not_validated' | 'failed' | null
```

- [ ] **Step 4: Add metric label helpers**

In `binder-design-system/frontend/src/pages/ExperimentDetailPage.tsx`, add helpers near `metric`:

```ts
const sourceLabel = (source?: string | null) => {
  if (source === 'rf3') return 'RF3'
  if (source === 'rfd3') return 'RFD3'
  if (source === 'mpnn') return 'MPNN'
  return '未计算'
}

const validationLabel = (status?: string | null, passed?: boolean) => {
  if (status === 'validated' || passed) return '已验证'
  if (status === 'failed') return '验证失败'
  return '未验证'
}

const candidateSortScore = (candidate: ExperimentDetail['designs'][number]) => {
  if (candidate.validation_status === 'validated' && typeof candidate.plddt === 'number') {
    return 10_000 + candidate.plddt
  }
  if (typeof candidate.plddt === 'number') return candidate.plddt
  if (typeof candidate.ranking_score === 'number') return candidate.ranking_score
  return Number.NEGATIVE_INFINITY
}
```

Replace `topCandidates()` with:

```ts
const topCandidates = (experiment: ExperimentDetail) =>
  [...(experiment.designs || [])]
    .sort((a, b) => candidateSortScore(b) - candidateSortScore(a))
    .slice(0, 3)
```

- [ ] **Step 5: Render provenance labels in top cards**

In the top-three card body in `ExperimentDetailPage.tsx`, replace the pLDDT line:

```tsx
<span className="text-xs text-gray-500">pLDDT {metric(d.plddt, 1)}</span>
```

with:

```tsx
<span className="text-xs text-gray-500">
  pLDDT {metric(d.plddt, 1)} · {sourceLabel(d.plddt_source)}
</span>
```

Replace the metric footer:

```tsx
RMSD {hasRmsd(d.rmsd) ? `${d.rmsd.toFixed(2)} Å` : '未计算'} · Ranking {metric(d.ranking_score, 3)}
```

with:

```tsx
<div>
  RMSD {hasRmsd(d.rmsd) ? `${d.rmsd.toFixed(2)} Å` : '未计算'} · Ranking {metric(d.ranking_score, 3)} · {sourceLabel(d.ranking_source)}
</div>
<div className="mt-1">{validationLabel(d.validation_status, d.passed_validation)}</div>
```

Also update the table to show provenance in compact form:

```tsx
<td className="px-4 py-3">
  <div>{d.plddt?.toFixed(1) || '-'}</div>
  <div className="text-[11px] text-gray-400">{sourceLabel(d.plddt_source)}</div>
</td>
```

- [ ] **Step 6: Run frontend detail test**

Run:

```bash
cd binder-design-system/frontend
npm test -- --run src/pages/__tests__/acceptance.test.tsx -t "shows fallback metric provenance"
```

Expected: pass.

- [ ] **Step 7: Commit frontend confidence cards**

Run:

```bash
git add binder-design-system/frontend/src/api/index.ts binder-design-system/frontend/src/pages/ExperimentDetailPage.tsx binder-design-system/frontend/src/pages/__tests__/acceptance.test.tsx
git commit -m "Show candidate metric provenance"
```

## Task 5: Full Verification

**Files:**
- No production edits expected.

- [ ] **Step 1: Run backend focused suites**

Run:

```bash
.venv/bin/pytest binder-design-system/backend/tests/test_design_pipeline_service.py binder-design-system/backend/tests/test_experiment_service.py binder-design-system/backend/tests/test_design_router_contract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run frontend acceptance suite**

Run:

```bash
cd binder-design-system/frontend
npm test -- --run src/pages/__tests__/acceptance.test.tsx
```

Expected: all tests pass.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: clean worktree after all task commits.
