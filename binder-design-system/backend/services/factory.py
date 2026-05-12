from dataclasses import dataclass
from functools import lru_cache

from adapters.model_adapters import HotspotModelAdapter, MPNNAdapter, RF3Adapter, RFD3Adapter
from database import SessionLocal
from repositories.experiments import ExperimentRepository
from services.design_pipeline import DesignPipelineService
from services.experiment_archive import ExperimentArchiveService
from services.experiments import ExperimentService
from services.hotspot_prediction import HotspotPredictionService
from config.settings import get_settings


@dataclass(frozen=True)
class DesignServices:
    hotspot_prediction: HotspotPredictionService
    pipeline: DesignPipelineService
    experiments: ExperimentService
    rfd3: RFD3Adapter
    mpnn: MPNNAdapter
    rf3: RF3Adapter


@lru_cache(maxsize=1)
def get_design_services() -> DesignServices:
    settings = get_settings()
    archive_service = ExperimentArchiveService(settings.paths.output_root)
    experiment_service = ExperimentService(
        ExperimentRepository(SessionLocal),
        archive_service=archive_service,
    )
    rfd3 = RFD3Adapter()
    mpnn = MPNNAdapter()
    rf3 = RF3Adapter()
    return DesignServices(
        hotspot_prediction=HotspotPredictionService(HotspotModelAdapter()),
        pipeline=DesignPipelineService(rfd3, mpnn, rf3, experiment_service),
        experiments=experiment_service,
        rfd3=rfd3,
        mpnn=mpnn,
        rf3=rf3,
    )
