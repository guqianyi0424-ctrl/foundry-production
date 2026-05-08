import sys
import types


def test_hotspot_predictor_loads_its_config_when_backend_config_is_preloaded(tmp_path, monkeypatch):
    import config as backend_config
    from utils.hotspot_predictor import HotspotPredictor

    hotspot_root = tmp_path / "hotspot-prediction"
    models_root = hotspot_root / "models"
    models_root.mkdir(parents=True)
    (models_root / "best_model_fold1.pth").write_bytes(b"weights")
    (hotspot_root / "config.py").write_text(
        "\n".join(
            [
                "INPUT_DIM = 321",
                "HIDDEN_DIM = 64",
                "NUM_HEADS = 4",
                "NUM_LAYERS = 3",
                "DROPOUT = 0.5",
            ]
        )
    )
    (hotspot_root / "model.py").write_text(
        "\n".join(
            [
                "from config import INPUT_DIM",
                "",
                "class PPIHotspotGAT:",
                "    def __init__(self, input_dim, **kwargs):",
                "        self.input_dim = input_dim",
                "    def load_state_dict(self, state_dict):",
                "        self.state_dict = state_dict",
                "    def eval(self):",
                "        self.evaluated = True",
                "    def to(self, device):",
                "        self.device = device",
                "        return self",
            ]
        )
    )

    fake_torch = types.SimpleNamespace(
        device=lambda name: name,
        load=lambda *args, **kwargs: {},
    )
    fake_dgl = types.SimpleNamespace(__version__="test")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "dgl", fake_dgl)

    predictor = HotspotPredictor()
    predictor.hotspot_dl_path = hotspot_root

    models = predictor._load_dl_models()

    assert models["fold1"].input_dim == 321
    assert sys.modules["config"] is backend_config


def test_esm_loader_uses_local_model_files(monkeypatch):
    from utils.hotspot_predictor import HotspotPredictor

    calls = []
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda name: name

    fake_transformers = types.ModuleType("transformers")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("tokenizer", model_name, kwargs))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("model", model_name, kwargs))
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True

    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModel = FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    esm = HotspotPredictor()._load_esm_model()

    assert esm is not None
    assert calls
    assert all(call[2]["local_files_only"] is True for call in calls)
