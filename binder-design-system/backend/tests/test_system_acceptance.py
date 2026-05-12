import asyncio

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, User, get_db
from main import app
from routers.auth import get_password_hash
from schemas.domain import AdapterResult, PipelineResult


PDB_CONTENT = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.112  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.300   9.112  1.00 20.00           C
ATOM      3  C   ALA A   1      13.021  14.725   9.412  1.00 20.00           C
ATOM      4  O   ALA A   1      12.334  15.685   9.102  1.00 20.00           O
ATOM      5  N   GLY A   2      14.210  14.855   9.998  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.780  16.169  10.329  1.00 20.00           C
ATOM      7  C   GLY A   2      16.293  16.101  10.480  1.00 20.00           C
ATOM      8  O   GLY A   2      16.907  17.102  10.847  1.00 20.00           O
TER
END
"""


async def run_with_acceptance_client(monkeypatch, scenario, tmp_path=None):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        db.add(
            User(
                id="admin-test",
                username="admin",
                email="admin@test.local",
                hashed_password=get_password_hash("admin123"),
                role="admin",
            )
        )
        db.commit()
    finally:
        db.close()

    async def override_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    import routers.design as design_router
    import routers.experiments as experiments_router
    from config.settings import AppSettings, get_settings

    app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(design_router, "get_design_services", lambda: FakeDesignServices())
    if tmp_path is not None:
        settings = get_settings()
        test_settings = AppSettings(
            paths=settings.paths.__class__(
                backend_root=settings.paths.backend_root,
                app_root=settings.paths.app_root,
                repo_root=settings.paths.repo_root,
                foundry_root=settings.paths.foundry_root,
                hotspot_dl_root=settings.paths.hotspot_dl_root,
                ppihotspotid_root=settings.paths.ppihotspotid_root,
                output_root=tmp_path,
            ),
            runtime=settings.runtime,
        )
        monkeypatch.setattr(experiments_router, "get_settings", lambda: test_settings)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await scenario(client)
    app.dependency_overrides.clear()


class FakePipelineService:
    def run_pipeline(self, pdb_content, hotspots, binder_length, job_id, user_id=None):
        return PipelineResult(
            job_id=job_id,
            experiment_id="acceptance-exp-1",
            status="completed",
            rfd3=AdapterResult(
                success=True,
                data={
                    "success": True,
                    "num_designs": 1,
                    "first_backbone_pdb": "RFD3_PDB",
                    "designs": [{"name": "backbone_0", "pdb_content": "RFD3_PDB"}],
                },
            ),
            mpnn=AdapterResult(
                success=True,
                data={
                    "success": True,
                    "num_sequences": 1,
                    "first_sequence_pdb": "MPNN_PDB",
                    "sequences": [{"name": "seq_0", "sequence": "ACDE", "pdb_content": "MPNN_PDB"}],
                },
            ),
            rf3=AdapterResult(
                success=True,
                data={
                    "success": True,
                    "predicted_pdb": "RF3_PDB",
                    "avg_plddt": 91.2,
                    "rmsd": 1.3,
                    "passed": True,
                },
            ),
        )


class FakeDesignServices:
    pipeline = FakePipelineService()


async def register_and_login(client: httpx.AsyncClient, username: str, password: str, email: str) -> dict:
    register_response = await client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "email": email},
    )
    assert register_response.status_code == 200
    assert register_response.json()["username"] == username

    login_response = await client.post(
        "/api/auth/login",
        data={"username": username, "password": password},
    )
    assert login_response.status_code == 200
    login_data = login_response.json()
    assert login_data["user"]["role"] == "researcher"
    return login_data


async def login_admin(client: httpx.AsyncClient) -> dict:
    response = await client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    return response.json()


def test_upload_pdb_returns_chain_summary_and_original_content(monkeypatch):
    async def scenario(client):
        response = await client.post(
            "/api/upload",
            files={"file": ("mini.pdb", PDB_CONTENT, "chemical/x-pdb")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["pdb_content"] == PDB_CONTENT
        assert data["chains"][0]["chain_id"] == "A"
        assert data["chains"][0]["length"] >= 2
        assert data["chains"][0]["sequence"] == "AG"

    asyncio.run(run_with_acceptance_client(monkeypatch, scenario))


def test_login_and_role_permissions_for_researcher_and_admin(monkeypatch):
    async def scenario(client):
        researcher = await register_and_login(
            client,
            username="researcher1",
            password="secret123",
            email="researcher1@test.local",
        )
        assert researcher["access_token"]

        headers = {"Authorization": f"Bearer {researcher['access_token']}"}
        me_response = await client.get("/api/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["role"] == "researcher"

        forbidden_response = await client.get("/api/users", headers=headers)
        assert forbidden_response.status_code == 403

        unauthorized_response = await client.get("/api/auth/me")
        assert unauthorized_response.status_code == 401

        admin_data = await login_admin(client)
        admin_headers = {"Authorization": f"Bearer {admin_data['access_token']}"}
        users_response = await client.get("/api/users", headers=admin_headers)
        assert users_response.status_code == 200
        assert admin_data["user"]["role"] == "admin"
        assert {user["username"] for user in users_response.json()["users"]} >= {"admin", "researcher1"}

    asyncio.run(run_with_acceptance_client(monkeypatch, scenario))


def test_experiment_record_lifecycle_acceptance(monkeypatch, tmp_path):
    async def scenario(client):
        login_data = await register_and_login(
            client,
            username="exp_user",
            password="secret123",
            email="exp_user@test.local",
        )
        headers = {"Authorization": f"Bearer {login_data['access_token']}"}

        create_response = await client.post(
            "/api/experiments",
            headers=headers,
            json={
                "name": "系统验收实验",
                "input_pdb": PDB_CONTENT,
                "target": "A/1-2",
                "hotspots": [{"chain": "A", "residue": 1}],
                "rfd3_config": {"binder_length": 80},
            },
        )
        assert create_response.status_code == 200
        experiment_id = create_response.json()["id"]

        list_response = await client.get("/api/experiments", headers=headers)
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed["total"] == 1
        assert listed["items"][0]["name"] == "系统验收实验"

        update_response = await client.put(
            f"/api/experiments/{experiment_id}",
            headers=headers,
            json={
                "status": "completed",
                "rf3_results": {"rmsd": 1.4, "avg_plddt": 90.0, "passed": True},
                "duration_seconds": 12.5,
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["ok"] is True

        design_response = await client.post(
            f"/api/experiments/{experiment_id}/designs",
            headers=headers,
            json={
                "design_name": "design_0",
                "sequence": "ACDE",
                "pdb_content": "ATOM",
                "plddt": 90.0,
                "rmsd": 1.4,
                "ranking_score": 0.82,
                "passed_validation": True,
            },
        )
        assert design_response.status_code == 200
        assert design_response.json()["ok"] is True

        detail_response = await client.get(f"/api/experiments/{experiment_id}", headers=headers)
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["status"] == "completed"
        assert detail["designs"][0]["sequence"] == "ACDE"
        assert detail["rf3_results"]["passed"] is True

        export_response = await client.get(f"/api/experiments/{experiment_id}/export", headers=headers)
        assert export_response.status_code == 200
        exported = export_response.json()
        assert exported["experiment"]["name"] == "系统验收实验"
        assert exported["designs"][0]["passed_validation"] is True
        assert exported["archive"]["candidates_csv"].endswith("candidates.csv")

        csv_response = await client.get(
            f"/api/experiments/{experiment_id}/export/csv",
            headers=headers,
        )
        assert csv_response.status_code == 200
        assert "design_name" in csv_response.text
        assert "sequence" in csv_response.text
        assert "design_0" in csv_response.text

        delete_response = await client.delete(f"/api/experiments/{experiment_id}", headers=headers)
        assert delete_response.status_code == 200
        assert delete_response.json()["ok"] is True

        missing_response = await client.get(f"/api/experiments/{experiment_id}", headers=headers)
        assert missing_response.status_code == 404

    asyncio.run(run_with_acceptance_client(monkeypatch, scenario, tmp_path))


def test_researcher_cannot_delete_another_users_experiment(monkeypatch):
    async def scenario(client):
        owner_login = await register_and_login(client, "owner", "secret123", "owner@test.local")
        other_login = await register_and_login(client, "other", "secret123", "other@test.local")
        owner_headers = {"Authorization": f"Bearer {owner_login['access_token']}"}
        other_headers = {"Authorization": f"Bearer {other_login['access_token']}"}

        create_response = await client.post(
            "/api/experiments",
            headers=owner_headers,
            json={"name": "归属测试"},
        )
        assert create_response.status_code == 200

        forbidden_response = await client.delete(
            f"/api/experiments/{create_response.json()['id']}",
            headers=other_headers,
        )
        assert forbidden_response.status_code == 403

    asyncio.run(run_with_acceptance_client(monkeypatch, scenario))


def test_mock_pipeline_end_to_end_acceptance(monkeypatch):
    async def scenario(client):
        response = await client.post(
            "/api/run-pipeline",
            json={
                "pdb_content": PDB_CONTENT,
                "hotspots": [{"chain": "A", "residue": 1}],
                "binder_length": 80,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["experiment_id"] == "acceptance-exp-1"
        assert data["rfd3_results"]["first_backbone_pdb"] == "RFD3_PDB"
        assert data["mpnn_results"]["first_sequence_pdb"] == "MPNN_PDB"
        assert data["rf3_results"]["passed"] is True

    asyncio.run(run_with_acceptance_client(monkeypatch, scenario))
