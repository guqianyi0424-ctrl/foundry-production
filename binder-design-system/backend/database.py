from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON
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


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    target = Column(String, nullable=True)
    detail = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        from routers.auth import get_password_hash
        existing = db.query(User).filter(User.username == "admin").first()
        if not existing:
            admin = User(
                id="admin_001",
                username="admin",
                email="admin@deepbinder.local",
                hashed_password=get_password_hash("admin123"),
                role="admin",
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()
