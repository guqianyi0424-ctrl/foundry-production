from adapters.model_adapters import HotspotModelAdapter


class HotspotPredictionService:
    def __init__(self, hotspot_adapter: HotspotModelAdapter):
        self.hotspot_adapter = hotspot_adapter

    def predict(self, pdb_content: str, top_k: int = 5):
        return self.hotspot_adapter.predict(pdb_content, top_k=top_k)
