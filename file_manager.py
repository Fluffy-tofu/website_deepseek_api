import os
import shutil
from werkzeug.utils import secure_filename
from typing import Optional, List


class FileManager:
    def __init__(self, upload_folder: str = 'uploads'):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def get_user_directory(self, user_id: str) -> str:
        """Get the upload directory for a specific user"""
        user_dir = os.path.join(self.upload_folder, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    def save_file(self, file, filename: str, user_id: str) -> bool:
        """Save an uploaded file to the user's directory"""
        try:
            user_dir = self.get_user_directory(user_id)
            filepath = os.path.join(user_dir, secure_filename(filename))
            file.save(filepath)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False

    def delete_file(self, filename: str, user_id: str) -> bool:
        """Delete a file from the user's directory"""
        try:
            filepath = os.path.join(self.upload_folder, user_id, secure_filename(filename))
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

    def get_file_path(self, filename: str, user_id: str) -> Optional[str]:
        """Get the full path of a file in the user's directory"""
        filepath = os.path.join(self.upload_folder, user_id, secure_filename(filename))
        return filepath if os.path.exists(filepath) else None

    def clear_user_files(self, chat_id: str, user_id: str) -> bool:
        """Delete all files associated with a specific chat for a user"""
        try:
            user_dir = self.get_user_directory(user_id)
            pattern = f"{chat_id}_*"
            for filename in os.listdir(user_dir):
                if filename.startswith(f"{chat_id}_"):
                    os.remove(os.path.join(user_dir, filename))
            return True
        except Exception as e:
            print(f"Error clearing files: {e}")
            return False

    def get_file_size(self, filename: str, user_id: str) -> Optional[int]:
        """Get the size of a file in the user's directory"""
        try:
            filepath = os.path.join(self.upload_folder, user_id, secure_filename(filename))
            return os.path.getsize(filepath) if os.path.exists(filepath) else None
        except Exception:
            return None

    def list_user_files(self, user_id: str) -> List[str]:
        """List all files in a user's directory"""
        try:
            user_dir = self.get_user_directory(user_id)
            return os.listdir(user_dir)
        except Exception:
            return []