"""
Initialize payment and subscription tables in the database.
"""
from models.database import init_db, create_tables

print("Initializing database...")
init_db()

print("Creating payment and subscription tables...")
create_tables()

print("✓ Tables created successfully!")
print("\nTables created:")
print("  - payments")
print("  - subscriptions")
print("\nYou can now start the app: python app.py")
