from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class MongoDB:
    """
    MongoDB connection manager sử dụng config hiện tại
    Hỗ trợ cả async (Motor) và sync (PyMongo) operations
    """
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.sync_client: Optional[MongoClient] = None
        self.database = None
        self.sync_database = None
        
    async def connect(self):
        """Kết nối async MongoDB"""
        try:
            # Sử dụng config đã có sẵn
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.database = self.client[settings.MONGODB_DATABASE]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {settings.MONGODB_DATABASE}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def connect_sync(self):
        """Kết nối sync MongoDB cho embedding operations"""
        try:
            self.sync_client = MongoClient(settings.MONGODB_URL)
            self.sync_database = self.sync_client[settings.MONGODB_DATABASE]
            
            # Test connection
            self.sync_client.admin.command('ping')
            logger.info(f"Connected to MongoDB (sync): {settings.MONGODB_DATABASE}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB (sync): {e}")
            raise
    
    async def disconnect(self):
        """Đóng kết nối async"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB (async)")
    
    def disconnect_sync(self):
        """Đóng kết nối sync"""
        if self.sync_client:
            self.sync_client.close()
            logger.info("Disconnected from MongoDB (sync)")
    
    def get_collection(self, collection_name: str):
        """Lấy collection async"""
        if self.database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.database[collection_name]
    
    def get_collection_sync(self, collection_name: str):
        """Lấy collection sync"""
        if self.sync_database is None:
            raise RuntimeError("Sync database not connected. Call connect_sync() first.")
        return self.sync_database[collection_name]

# Global MongoDB instance
mongodb = MongoDB()

async def get_database():
    """Dependency để lấy database instance"""
    if mongodb.database is None:
        await mongodb.connect()
    return mongodb.database

def get_database_sync():
    """Lấy sync database instance"""
    if mongodb.sync_database is None:
        mongodb.connect_sync()
    return mongodb.sync_database

# Collections cho Chemistry content
CHEMISTRY_TEXTBOOK_COLLECTION = "chemistry_textbooks"
CHEMISTRY_CHAPTERS_COLLECTION = "chemistry_chapters" 
CHEMISTRY_LESSONS_COLLECTION = "chemistry_lessons"
CHEMISTRY_EMBEDDINGS_COLLECTION = "chemistry_embeddings"
LESSON_PLANS_COLLECTION = "lesson_plans"

# GridFS Collections
GRIDFS_BUCKET_NAME = "planbook_files"
FILE_METADATA_COLLECTION = "file_metadata"
