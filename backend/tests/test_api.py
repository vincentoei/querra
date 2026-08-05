"""Integration tests for the public API endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_admin_endpoints_require_auth(test_db):
    response = client.post(
        "/api/v1/admin/databases",
        json={"db_id": "x", "display_name": "X", "db_path": str(test_db)},
    )
    assert response.status_code == 401


def test_register_database(test_db, admin_headers):
    response = client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": "company",
            "display_name": "Company DB",
            "db_path": str(test_db),
            "description": "test db",
        },
        headers=admin_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["db_id"] == "company"


def test_list_and_schema(test_db, admin_headers):
    client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": "company",
            "display_name": "Company DB",
            "db_path": str(test_db),
        },
        headers=admin_headers,
    )
    response = client.get("/api/v1/databases")
    assert response.status_code == 200
    assert any(db["db_id"] == "company" for db in response.json())

    schema = client.get("/api/v1/databases/company/schema")
    assert schema.status_code == 200
    assert "CREATE TABLE employees" in schema.text


def test_execute_sql(test_db, admin_headers):
    client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": "company",
            "display_name": "Company DB",
            "db_path": str(test_db),
        },
        headers=admin_headers,
    )
    response = client.post(
        "/api/v1/execute-sql",
        json={"db_id": "company", "sql": "SELECT name FROM employees WHERE age > 26"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["execution_error"] is None
    assert ["Alice"] in data["execution_result"]
    assert data["execution_columns"] == ["name"]


def test_execute_sql_blocks_destructive(test_db, admin_headers):
    client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": "company",
            "display_name": "Company DB",
            "db_path": str(test_db),
        },
        headers=admin_headers,
    )
    response = client.post(
        "/api/v1/execute-sql",
        json={"db_id": "company", "sql": "DROP TABLE employees"},
    )
    assert response.status_code == 400
    assert "not read-only" in response.json()["detail"].lower()


def test_generate_sql_without_model(test_db, admin_headers):
    client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": "company",
            "display_name": "Company DB",
            "db_path": str(test_db),
        },
        headers=admin_headers,
    )
    response = client.post(
        "/api/v1/generate-sql",
        json={
            "db_id": "company",
            "question": "Who is older than 26?",
            "execute": False,
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_database_not_found():
    response = client.post(
        "/api/v1/execute-sql",
        json={"db_id": "missing", "sql": "SELECT 1"},
    )
    assert response.status_code == 404


def test_execute_sql_blocks_nested_write_in_cte(test_db, admin_headers):
    client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": "company",
            "display_name": "Company DB",
            "db_path": str(test_db),
        },
        headers=admin_headers,
    )
    response = client.post(
        "/api/v1/execute-sql",
        json={
            "db_id": "company",
            "sql": "WITH cte AS (DELETE FROM employees) SELECT 1",
        },
    )
    assert response.status_code == 400
    assert "not read-only" in response.json()["detail"].lower()


def test_execute_sql_blocks_multistatement_attack(test_db, admin_headers):
    client.post(
        "/api/v1/admin/databases",
        json={
            "db_id": "company",
            "display_name": "Company DB",
            "db_path": str(test_db),
        },
        headers=admin_headers,
    )
    response = client.post(
        "/api/v1/execute-sql",
        json={"db_id": "company", "sql": "SELECT 1; DROP TABLE employees"},
    )
    assert response.status_code == 400
    assert "not read-only" in response.json()["detail"].lower()
