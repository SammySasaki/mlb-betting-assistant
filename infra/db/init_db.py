from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.db_models import Base

# Example connection string for local Postgres
DATABASE_URL = "postgresql://postgres:s@db:5432/sports_betting"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)   
    print("Database tables created.")

if __name__ == "__main__":
    init_db()