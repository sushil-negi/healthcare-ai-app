#!/usr/bin/env python3
"""
Fix Model Registry Schema - Add missing metadata column
"""

import os

import psycopg2
from psycopg2 import sql

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mlops",
    "user": "mlops_user",
    "password": "mlops_password",
}


def fix_model_registry_schema():
    """Add metadata column to models table if it doesn't exist"""

    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        print("🔧 Checking Model Registry schema...")

        # Check if metadata column exists
        cur.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'models' 
            AND column_name = 'metadata'
        """
        )

        if cur.fetchone() is None:
            print("📝 Adding metadata column to models table...")

            # Add metadata column
            cur.execute(
                """
                ALTER TABLE models 
                ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb
            """
            )

            # Add other potentially missing columns
            cur.execute(
                """
                ALTER TABLE models 
                ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT ARRAY[]::TEXT[]
            """
            )

            cur.execute(
                """
                ALTER TABLE models 
                ADD COLUMN IF NOT EXISTS metrics JSONB DEFAULT '{}'::jsonb
            """
            )

            conn.commit()
            print("✅ Schema updated successfully!")
        else:
            print("✅ Schema already up to date")

        # Show current schema
        cur.execute(
            """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'models' 
            ORDER BY ordinal_position
        """
        )

        print("\n📊 Current models table schema:")
        for col in cur.fetchall():
            print(f"   - {col[0]}: {col[1]}")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure MLOps platform is running:")
        print("   cd /Users/snegi/Documents/github/enterprise-repos/mlops-platform")
        print("   docker compose -f docker-compose.platform.yml up -d")


if __name__ == "__main__":
    fix_model_registry_schema()
