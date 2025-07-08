# db.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import logging
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()

class DBCounter:
    def __init__(self):
        self.client = None
        self.db = None
        self.counter_collection = None
        self.connect()

    def connect(self):
        try:
            self.client = MongoClient(os.getenv('MONGO_URI'))
            self.db = self.client.get_database('mediscan')
            self.counter_collection = self.db['upload_counter']
            
            # Initialize counter if it doesn't exist
            if self.counter_collection.count_documents({}) == 0:
                self.counter_collection.insert_one({
                    '_id': 'global_counter',
                    'count': 0
                })
                
            logging.info("Connected to MongoDB successfully")
            
        except ConnectionFailure as e:
            logging.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def increment_counter(self, increment_by=1):
        try:
            result = self.counter_collection.update_one(
                {'_id': 'global_counter'},
                {'$inc': {'count': increment_by}}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error incrementing counter: {str(e)}")
            return False

    def get_current_count(self):
        try:
            doc = self.counter_collection.find_one({'_id': 'global_counter'})
            return doc['count'] if doc else 0
        except Exception as e:
            logging.error(f"Error getting counter: {str(e)}")
            return 0

    def reset_counter(self):
        try:
            result = self.counter_collection.update_one(
                {'_id': 'global_counter'},
                {'$set': {'count': 0}}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error resetting counter: {str(e)}")
            return False

# Global instance
db_counter = DBCounter()