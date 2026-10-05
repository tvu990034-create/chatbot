"""
Database configuration and initialization for Phase 2
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

# Database URL from environment variable or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

# For async operations (if using async database)
ASYNC_DATABASE_URL = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://") if DATABASE_URL.startswith("sqlite://") else DATABASE_URL

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create async engine
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False
)

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

def get_db() -> Session:
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncSession:
    """Get an async database session."""
    async with AsyncSessionLocal() as session:
        yield session

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def drop_tables():
    """Drop all database tables (use with caution!)."""
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped!")

def reset_database():
    """Reset the database by dropping and recreating tables."""
    drop_tables()
    create_tables()

# Initialize database on import
def init_db():
    """Initialize the database with tables and default data."""
    create_tables()
    
    # Create default admin user if it doesn't exist
    from models import User, create_user
    from auth import get_password_hash
    
    db = SessionLocal()
    try:
        # Check if admin user exists
        admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
        admin_password = os.getenv("ADMIN_PASSWORD")
        if not admin_password:
            raise ValueError("ADMIN_PASSWORD must be set for secure deployment")
        
        existing_admin = db.query(User).filter(User.email == admin_email).first()
        if not existing_admin:
            admin_user = User(
                email=admin_email,
                hashed_password=get_password_hash(admin_password),
                is_admin=True
            )
            try:
                db.add(admin_user)
                db.commit()
                print(f"Default admin user created: {admin_email}")
            except Exception as e:
                print(f"Error creating admin user: {e}")
                db.rollback()
        else:
            print("Admin user already exists")
            
    except Exception as e:
        print(f"Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()


# Test database connection
def test_db_connection():
    """Test the database connection."""
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        print("Database connection successful!")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    # Initialize database when run directly
    print("Initializing database...")
    init_db()
    test_db_connection()
