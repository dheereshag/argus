from app.core.config import settings
from app.schemas.plate import ProvidersResponse


def test_list_providers_endpoint(client):
    response = client.get("/providers")
    assert response.status_code == 200

    data = response.json()
    validated = ProvidersResponse.model_validate(data)

    assert settings.DEFAULT_PROVIDER in validated.available_providers
    assert "docling" in [p.value for p in validated.available_providers]
    assert validated.default_provider == settings.DEFAULT_PROVIDER
