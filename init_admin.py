"""
Initialize admin user for VidyaTid.
Run this once on deployment to create the admin account.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.database import SessionLocal, create_tables
from models.user import User

def create_admin_user():
    """Create admin user if not exists."""
    db = SessionLocal()
    try:
        # Check if admin exists
        admin = db.query(User).filter_by(email='admin@test.com').first()
        if admin:
            print("Admin user already exists!")
            return
        
        # Also check by username
        admin = db.query(User).filter_by(username='admin').first()
        if admin:
            print("Admin user (by username) already exists!")
            return
        
        # Create admin user with full access
        admin_user = User(
            username='admin',
            password='Admin@123',
            email='admin@test.com',
            preferences={
                'role': 'admin',
                'tier': 'premium',
                'is_admin': True,
                'unlimited_access': True,
                'all_features_enabled': True
            }
        )
        
        db.add(admin_user)
        db.commit()
        print("✅ Admin user created successfully!")
        print(f"   Email: admin@test.com")
        print(f"   Password: Admin@123")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating admin: {e}")
    finally:
        db.close()

if __name__ == '__main__':
    create_tables()
    create_admin_user()
