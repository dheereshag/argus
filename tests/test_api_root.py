from app.core.config import settings


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == settings.PROJECT_NAME
    assert data["version"] == settings.VERSION
    assert data["status"] == "running"
    assert data["docs"] == "/docs"



def test_process_time_header(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "x-process-time-ms" in response.headers
    process_time = float(response.headers["x-process-time-ms"])
    assert process_time >= 0.0


import pytest


@pytest.mark.anyio
async def test_lifespan_warmup_warning():
    from unittest.mock import patch

    from app.server import app, lifespan

    with patch("app.server.VehicleDetector.get_model", side_effect=RuntimeError("GPU warmup failure")):
        async with lifespan(app):
            pass



def test_exception_handlers():
    from fastapi.testclient import TestClient

    from app.core.contracts import ContractViolation
    from app.server import app

    @app.get("/_test_contract")
    def _raise_contract():
        raise ContractViolation("Test violation")

    @app.get("/_test_generic")
    def _raise_generic():
        raise ZeroDivisionError("division by zero")

    @app.get("/_test_val")
    def _raise_val(num: int):
        return {"num": num}

    client = TestClient(app, raise_server_exceptions=False)

    resp_contract = client.get("/_test_contract")
    assert resp_contract.status_code == 500
    assert resp_contract.json()["error_type"] == "ContractViolation"

    resp_val = client.get("/_test_val?num=invalid")
    assert resp_val.status_code == 422
    assert resp_val.json()["error_type"] == "RequestValidationError"

    resp_generic = client.get("/_test_generic")
    assert resp_generic.status_code == 500
    assert resp_generic.json()["error_type"] == "InternalServerError"

