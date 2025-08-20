from tinydb import TinyDB, where
from datetime import datetime

# Store DB file in a known location
db = TinyDB("document_status.json")

def insert_document_status(doc_id, file_name, folder, url, status, sender):
    db.insert({
        "doc_id": doc_id,
        "file_name": file_name,
        "folder": folder,
        "url": url,
        "status": status,
        "sender": sender,
        "timestamp": datetime.utcnow().isoformat()
    })
