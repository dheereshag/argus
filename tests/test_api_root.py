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
