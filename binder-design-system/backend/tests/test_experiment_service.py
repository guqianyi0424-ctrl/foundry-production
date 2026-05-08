from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def make_session_factory():
    from database import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def test_experiment_service_creates_and_updates_pipeline_experiment():
    from database import Experiment
    from repositories.experiments import ExperimentRepository
    from services.experiments import ExperimentService

    session_factory = make_session_factory()
    service = ExperimentService(ExperimentRepository(session_factory))

    experiment_id = service.create_pipeline_experiment(
        name="Pipeline_20260507_120000",
        input_pdb="ATOM",
        hotspots=[{"chain": "A", "residue": 10}],
        rfd3_config={"binder_length": 80},
    )
    service.save_step(experiment_id, "rfd3", {"success": True}, {"binder_length": 80})

    db = session_factory()
    try:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).one()
        assert exp.status == "rfd3_completed"
        assert exp.rfd3_results == {"success": True}
        assert exp.rfd3_config == {"binder_length": 80}
    finally:
        db.close()


def test_experiment_service_saves_design_records():
    from database import ExperimentDesign
    from repositories.experiments import ExperimentRepository
    from services.experiments import ExperimentService

    session_factory = make_session_factory()
    service = ExperimentService(ExperimentRepository(session_factory))
    experiment_id = service.create_pipeline_experiment("Pipeline", "ATOM", [], {})

    service.save_designs(
        experiment_id,
        [
            {
                "name": "design_0",
                "sequence": "ACD",
                "pdb_content": "ATOM",
                "plddt": 91.0,
                "rmsd": 1.2,
                "ranking_score": 0.4,
                "passed_validation": True,
            }
        ],
    )

    db = session_factory()
    try:
        design = db.query(ExperimentDesign).filter(
            ExperimentDesign.experiment_id == experiment_id
        ).one()
        assert design.design_name == "design_0"
        assert design.sequence == "ACD"
        assert design.passed_validation is True
    finally:
        db.close()
