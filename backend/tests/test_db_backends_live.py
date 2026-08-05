"""Live integration tests for PostgreSQL and MySQL backends.

These tests are skipped unless TEST_POSTGRES_URL or TEST_MYSQL_URL is set.
Use the docker test-db profile to spin up local databases:

    cd backend/docker
    docker compose --profile test-db up -d

"""

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"X-Admin-API-Key": os.environ.get("ADMIN_API_KEY", "test-admin-key")}


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set"
)
def test_postgres_backend_via_api(admin_headers):
    db_id = "test_postgres"
    # Clean up if previous run left it behind.
    client.delete(f"/api/v1/admin/databases/{db_id}", headers=admin_headers)

    res = client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": db_id,
            "display_name": "Test Postgres",
            "backend_type": "postgres",
            "connection_env": "TEST_POSTGRES_URL",
        },
        headers=admin_headers,
    )
    assert res.status_code == 201, res.json()

    schema = client.get(f"/api/v1/databases/{db_id}/schema")
    assert schema.status_code == 200
    assert "CREATE TABLE employees" in schema.text

    res = client.post(
        "/api/v1/execute-sql",
        json={"db_id": db_id, "sql": "SELECT name FROM employees WHERE age > 26"},
    )
    assert res.status_code == 200, res.json()
    data = res.json()
    assert data["valid"] is True
    assert ["Alice"] in data["execution_result"]
    assert data["execution_columns"] == ["name"]

    client.delete(f"/api/v1/admin/databases/{db_id}", headers=admin_headers)


@pytest.mark.skipif(
    not os.environ.get("TEST_MYSQL_URL"), reason="TEST_MYSQL_URL not set"
)
def test_mysql_backend_via_api(admin_headers):
    db_id = "test_mysql"
    client.delete(f"/api/v1/admin/databases/{db_id}", headers=admin_headers)

    res = client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": db_id,
            "display_name": "Test MySQL",
            "backend_type": "mysql",
            "connection_env": "TEST_MYSQL_URL",
        },
        headers=admin_headers,
    )
    assert res.status_code == 201, res.json()

    schema = client.get(f"/api/v1/databases/{db_id}/schema")
    assert schema.status_code == 200
    assert "CREATE TABLE employees" in schema.text

    res = client.post(
        "/api/v1/execute-sql",
        json={"db_id": db_id, "sql": "SELECT name FROM employees WHERE age > 26"},
    )
    assert res.status_code == 200, res.json()
    data = res.json()
    assert data["valid"] is True
    assert ["Alice"] in data["execution_result"]
    assert data["execution_columns"] == ["name"]

    client.delete(f"/api/v1/admin/databases/{db_id}", headers=admin_headers)
