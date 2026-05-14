from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, 'deepbinder.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="researcher")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    experiments = relationship("Experiment", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(String, default="created")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    input_pdb = Column(Text, nullable=True)
    target = Column(String, nullable=True)
    hotspots = Column(JSON, nullable=True)

    rfd3_config = Column(JSON, nullable=True)
    mpnn_config = Column(JSON, nullable=True)
    rf3_config = Column(JSON, nullable=True)

    rfd3_results = Column(JSON, nullable=True)
    mpnn_results = Column(JSON, nullable=True)
    rf3_results = Column(JSON, nullable=True)

    duration_seconds = Column(Float, nullable=True)
    gpu_info = Column(String, nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)

    user = relationship("User", back_populates="experiments")
    designs = relationship("ExperimentDesign", back_populates="experiment", cascade="all, delete-orphan")


class ExperimentDesign(Base):
    __tablename__ = "experiment_designs"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    design_name = Column(String, nullable=True)
    sequence = Column(Text, nullable=True)
    pdb_content = Column(Text, nullable=True)
    plddt = Column(Float, nullable=True)
    rmsd = Column(Float, nullable=True)
    ranking_score = Column(Float, nullable=True)
    passed_validation = Column(Boolean, default=False)
    plddt_source = Column(String, nullable=True)
    ranking_source = Column(String, nullable=True)
    validation_status = Column(String, nullable=True)

    experiment = relationship("Experiment", back_populates="designs")


class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"

    id = Column(String, primary_key=True)
    status = Column(String, default="running", nullable=False)
    experiment_id = Column(String, nullable=True)
    failed_step = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    rfd3_results = Column(JSON, nullable=True)
    mpnn_results = Column(JSON, nullable=True)
    rf3_results = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    target = Column(String, nullable=True)
    detail = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")


def ensure_schema_compatibility():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "experiment_designs" not in table_names:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("experiment_designs")}
    required_columns = {
        "plddt_source": "VARCHAR",
        "ranking_source": "VARCHAR",
        "validation_status": "VARCHAR",
    }
    missing_columns = {
        name: column_type
        for name, column_type in required_columns.items()
        if name not in existing_columns
    }
    if not missing_columns:
        return

    with engine.begin() as connection:
        for name, column_type in missing_columns.items():
            connection.execute(text(f"ALTER TABLE experiment_designs ADD COLUMN {name} {column_type}"))


def _is_production_environment() -> bool:
    return os.getenv("DEEPBINDER_ENV", "development").strip().lower() in {"prod", "production"}


def _bootstrap_admin_config() -> tuple[str, str, str] | None:
    username = os.getenv("DEEPBINDER_BOOTSTRAP_ADMIN_USERNAME", "").strip()
    password = os.getenv("DEEPBINDER_BOOTSTRAP_ADMIN_PASSWORD", "")
    email = os.getenv("DEEPBINDER_BOOTSTRAP_ADMIN_EMAIL", "admin@deepbinder.local").strip()
    if not username and not password:
        if _is_production_environment():
            return None
        return ("admin", "admin123", email)
    if not username or not password:
        raise RuntimeError("DEEPBINDER_BOOTSTRAP_ADMIN_USERNAME 和 DEEPBINDER_BOOTSTRAP_ADMIN_PASSWORD 必须同时设置")
    if len(password) < 8:
        raise RuntimeError("DEEPBINDER_BOOTSTRAP_ADMIN_PASSWORD 长度不能少于8位")
    return (username, password, email)


def init_db():
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility()
    bootstrap = _bootstrap_admin_config()
    if bootstrap is None:
        return
    username, password, email = bootstrap
    db = SessionLocal()
    try:
        from routers.auth import get_password_hash
        existing = db.query(User).filter(User.username == username).first()
        if not existing:
            admin = User(
                id=f"admin_{username}",
                username=username,
                email=email,
                hashed_password=get_password_hash(password),
                role="admin",
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()
