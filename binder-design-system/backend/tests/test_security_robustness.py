import asyncio
import os
import subprocess
import sys

import httpx
import pytest
from sqlalchemy.exc import OperationalError
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, User, get_db
from main import app
from routers.auth import ALGORITHM, get_password_hash


PDB_CONTENT = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.112  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.300   9.112  1.00 20.00           C
ATOM      3  C   ALA A   1      13.021  14.725   9.412  1.00 20.00           C
ATOM      4  O   ALA A   1      12.334  15.685   9.102  1.00 20.00           O
ATOM      5  N   GLY A   2      14.210  14.855   9.998  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.780  16.169  10.329  1.00 20.00           C
TER
END
"""


def test_init_db_does_not_create_default_admin_in_production(monkeypatch, tmp_path):
    import database

    db_path = tmp_path / "prod.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    monkeypatch.setenv("DEEPBINDER_ENV", "production")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", TestingSessionLocal)

    database.init_db()

    db = TestingSessionLocal()
    try:
        assert db.query(User).filter(User.username == "admin").first() is None
    finally:
        db.close()


def test_init_db_can_seed_admin_from_environment(monkeypatch, tmp_path):
    import database

    db_path = tmp_path / "seeded.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    monkeypatch.setenv("DEEPBINDER_ENV", "production")
    monkeypatch.setenv("DEEPBINDER_BOOTSTRAP_ADMIN_USERNAME", "root")
    monkeypatch.setenv("DEEPBINDER_BOOTSTRAP_ADMIN_PASSWORD", "long-secret-password")
    monkeypatch.setenv("DEEPBINDER_BOOTSTRAP_ADMIN_EMAIL", "root@test.local")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", TestingSessionLocal)

    database.init_db()

    db = TestingSessionLocal()
    try:
        admin = db.query(User).filter(User.username == "root").one()
        assert admin.email == "root@test.local"
        assert admin.role == "admin"
    finally:
        db.close()


async def run_with_client(monkeypatch, scenario):
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

    app.dependency_overrides[get_db] = override_db
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        await scenario(client)
    app.dependency_overrides.clear()


async def register_and_login(client: httpx.AsyncClient, username: str) -> dict:
    response = await client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": "secret123",
            "email": f"{username}@test.local",
        },
    )
    assert response.status_code == 200
    login_response = await client.post(
        "/api/auth/login",
        data={"username": username, "password": "secret123"},
    )
    assert login_response.status_code == 200
    return login_response.json()


async def create_experiment(client: httpx.AsyncClient, token: str, name: str) -> str:
    response = await client.post(
        "/api/experiments",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": name, "input_pdb": PDB_CONTENT, "target": "A/1-2"},
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_researcher_cannot_read_or_mutate_another_users_experiment(monkeypatch):
    async def scenario(client):
        owner = await register_and_login(client, "owner")
        other = await register_and_login(client, "other")
        owner_token = owner["access_token"]
        other_headers = {"Authorization": f"Bearer {other['access_token']}"}
        experiment_id = await create_experiment(client, owner_token, "owner-exp")

        checks = [
            await client.get(f"/api/experiments/{experiment_id}", headers=other_headers),
            await client.put(
                f"/api/experiments/{experiment_id}",
                headers=other_headers,
                json={"status": "completed"},
            ),
            await client.post(
                f"/api/experiments/{experiment_id}/designs",
                headers=other_headers,
                json={"design_name": "steal", "sequence": "ACDE"},
            ),
            await client.get(f"/api/experiments/{experiment_id}/export", headers=other_headers),
            await client.post(
                "/api/experiments/compare",
                headers=other_headers,
                json=[experiment_id],
            ),
        ]

        assert [response.status_code for response in checks] == [403, 403, 403, 403, 403]

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_experiment_interfaces_require_login(monkeypatch):
    async def scenario(client):
        owner = await register_and_login(client, "owner")
        experiment_id = await create_experiment(client, owner["access_token"], "owner-exp")

        checks = [
            await client.get(f"/api/experiments/{experiment_id}"),
            await client.post("/api/experiments", json={"name": "anonymous"}),
            await client.put(f"/api/experiments/{experiment_id}", json={"status": "completed"}),
            await client.post(
                f"/api/experiments/{experiment_id}/designs",
                json={"design_name": "anonymous"},
            ),
            await client.get(f"/api/experiments/{experiment_id}/export"),
            await client.post("/api/experiments/compare", json=[experiment_id]),
        ]

        assert [response.status_code for response in checks] == [401, 401, 401, 401, 401, 401]

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_register_rejects_invalid_identity_inputs(monkeypatch):
    async def scenario(client):
        cases = [
            {"username": "ab", "password": "secret123", "email": "short@test.local"},
            {"username": "bad name", "password": "secret123", "email": "bad@test.local"},
            {"username": "weakpass", "password": "123", "email": "weak@test.local"},
            {"username": "bademail", "password": "secret123", "email": "not-email"},
        ]

        responses = [await client.post("/api/auth/register", json=payload) for payload in cases]

        assert [response.status_code for response in responses] == [422, 422, 422, 422]

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_error_responses_use_unified_shape_for_auth_and_validation(monkeypatch):
    async def scenario(client):
        unauthorized = await client.get("/api/auth/me")
        invalid_payload = await client.post(
            "/api/auth/register",
            json={"username": "ab", "password": "123", "email": "not-email"},
        )

        assert unauthorized.status_code == 401
        assert unauthorized.json()["code"] == "unauthorized"
        assert unauthorized.json()["message"] == "请先登录"
        assert "details" in unauthorized.json()

        assert invalid_payload.status_code == 422
        data = invalid_payload.json()
        assert data["code"] == "validation_error"
        assert data["message"] == "请求参数校验失败"
        assert isinstance(data["details"], list)

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_forged_token_is_rejected_on_protected_endpoint(monkeypatch):
    async def scenario(client):
        forged_token = jwt.encode({"sub": "admin", "role": "admin"}, "wrong-secret", algorithm=ALGORITHM)
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {forged_token}"},
        )

        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_run_rfd3_rejects_invalid_bearer_token(monkeypatch):
    import routers.design as design_router

    class FakeRFD3Service:
        def run(self, config):
            pytest.fail("RFD3 should not run when bearer token is invalid")

    class FakeServices:
        rfd3 = FakeRFD3Service()

    monkeypatch.setattr(design_router, "get_design_services", lambda: FakeServices())

    async def scenario(client):
        forged_token = jwt.encode({"sub": "admin", "role": "admin"}, "wrong-secret", algorithm=ALGORITHM)
        response = await client.post(
            "/api/run-rfd3",
            headers={"Authorization": f"Bearer {forged_token}"},
            json={"binder_length": 72, "diffusion_batch_size": 1, "n_batches": 1},
        )

        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_monitor_endpoints_require_admin_role(monkeypatch):
    async def scenario(client):
        researcher = await register_and_login(client, "researcher")
        researcher_headers = {"Authorization": f"Bearer {researcher['access_token']}"}
        admin_login = await client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "admin123"},
        )
        assert admin_login.status_code == 200
        admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}

        anonymous = await client.get("/api/monitor/status")
        forbidden = await client.get("/api/monitor/status", headers=researcher_headers)
        allowed = await client.get("/api/monitor/status", headers=admin_headers)

        assert anonymous.status_code == 401
        assert forbidden.status_code == 403
        assert allowed.status_code == 200
        assert allowed.json()["status"] == "running"

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_login_rate_limit_blocks_repeated_failures(monkeypatch):
    import routers.auth as auth_router

    monkeypatch.setattr(auth_router, "MAX_LOGIN_FAILURES", 2)
    monkeypatch.setattr(auth_router, "LOGIN_FAILURE_WINDOW_SECONDS", 300)
    auth_router._login_failures.clear()

    async def scenario(client):
        responses = [
            await client.post(
                "/api/auth/login",
                data={"username": "admin", "password": "wrong-password"},
            )
            for _ in range(3)
        ]

        assert [response.status_code for response in responses] == [401, 401, 429]
        assert responses[-1].json()["code"] == "too_many_requests"

    asyncio.run(run_with_client(monkeypatch, scenario))
    auth_router._login_failures.clear()


def test_upload_rejects_invalid_file_types_empty_files_and_oversized_content(monkeypatch):
    import routers.upload as upload_router

    monkeypatch.setattr(upload_router, "MAX_UPLOAD_BYTES", 16)

    async def scenario(client):
        invalid_ext = await client.post(
            "/api/upload",
            files={"file": ("bad.txt", PDB_CONTENT, "text/plain")},
        )
        empty_file = await client.post(
            "/api/upload",
            files={"file": ("empty.pdb", "", "chemical/x-pdb")},
        )
        oversized_content = await client.post(
            "/api/upload",
            files={"file": ("huge.pdb", PDB_CONTENT, "chemical/x-pdb")},
        )

        assert invalid_ext.status_code == 400
        assert empty_file.status_code == 400
        assert oversized_content.status_code == 400

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_upload_parse_failure_returns_sanitized_message(monkeypatch):
    async def scenario(client):
        response = await client.post(
            "/api/upload",
            files={"file": ("invalid.pdb", "not a pdb structure", "chemical/x-pdb")},
        )

        assert response.status_code == 400
        assert response.json()["message"] == "文件解析失败，请确认文件为有效的 PDB/CIF/mmCIF 结构文件"

    asyncio.run(run_with_client(monkeypatch, scenario))


def test_cors_origin_configuration_uses_environment_allowlist(monkeypatch):
    env = {
        **os.environ,
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "DEEPBINDER_CORS_ORIGINS": "https://deepbinder.example.edu,http://localhost:5173",
    }
    script = """
import main
middleware = next(
    item for item in main.app.user_middleware
    if item.cls.__name__ == "CORSMiddleware"
)
print(",".join(middleware.kwargs["allow_origins"]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "https://deepbinder.example.edu,http://localhost:5173"


def test_monitor_status_falls_back_to_nvidia_smi_gpu_info(monkeypatch):
    import routers.monitor as monitor_router

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    class FakeCompletedProcess:
        stdout = "NVIDIA RTX 4090, 8192 MiB, 24576 MiB, 35\n"

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setattr(
        monitor_router.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompletedProcess(),
    )

    status = asyncio.run(monitor_router.system_status(current_user=User(role="admin")))

    assert status["gpu"] == {
        "name": "NVIDIA RTX 4090",
        "memory": "8192 / 24576 MiB",
        "memory_used_mib": 8192,
        "memory_total_mib": 24576,
        "utilization_percent": 35,
        "source": "nvidia-smi",
    }


def test_monitor_status_survives_system_metric_collection_errors(monkeypatch):
    import routers.monitor as monitor_router

    def broken_metric(*args, **kwargs):
        raise RuntimeError("metric unavailable")

    monkeypatch.setattr(monitor_router.psutil, "cpu_percent", broken_metric)
    monkeypatch.setattr(monitor_router.psutil, "virtual_memory", broken_metric)
    monkeypatch.setattr(monitor_router.psutil, "disk_usage", broken_metric)
    monkeypatch.setattr(
        monitor_router,
        "get_gpu_status",
        lambda: {
            "name": "N/A",
            "memory": "N/A",
            "memory_used_mib": None,
            "memory_total_mib": None,
            "utilization_percent": None,
            "source": "unavailable",
        },
    )

    status = asyncio.run(monitor_router.system_status(current_user=User(role="admin")))

    assert status["status"] == "running"
    assert status["cpu_percent"] is None
    assert status["memory"] == {"total_gb": None, "used_gb": None, "percent": None}
    assert status["disk"] == {"total_gb": None, "used_gb": None, "percent": None}


def test_experiment_list_returns_service_unavailable_when_database_is_locked(monkeypatch):
    from routers.auth import create_access_token

    class BrokenQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            raise OperationalError("SELECT count(*) FROM experiments", {}, Exception("database is locked"))

    class BrokenSession:
        def query(self, model):
            return BrokenQuery()

    async def override_locked_db():
        yield BrokenSession()

    async def scenario():
        app.dependency_overrides[get_db] = override_locked_db
        token = create_access_token({"sub": "admin", "role": "admin"})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/api/experiments",
                headers={"Authorization": f"Bearer {token}"},
            )
        app.dependency_overrides.clear()

        assert response.status_code == 503
        assert response.json()["message"] == "数据库繁忙，请稍后重试"

    try:
        asyncio.run(scenario())
    finally:
        app.dependency_overrides.clear()
