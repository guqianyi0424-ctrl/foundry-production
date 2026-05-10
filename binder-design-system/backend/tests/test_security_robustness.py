import asyncio
import os
import subprocess
import sys

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, User, get_db
from main import app
from routers.auth import get_password_hash


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
