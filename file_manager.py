import os
import shutil
from werkzeug.utils import secure_filename
from typing import Optional, List


class FileManager:
    def __init__(self, upload_folder: str = 'uploads'):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def save_file(self, file, filename: str) -> bool:
        """Save an uploaded file"""
        try:
            filepath = os.path.join(self.upload_folder, secure_filename(filename))
            file.save(filepath)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False

    def delete_file(self, filename: str) -> bool:
        """Delete a file from the uploads folder"""
        try:
            filepath = os.path.join(self.upload_folder, secure_filename(filename))
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

    def get_file_path(self, filename: str) -> Optional[str]:
        """Get the full path of a file"""
        filepath = os.path.join(self.upload_folder, secure_filename(filename))
        return filepath if os.path.exists(filepath) else None

    def clear_user_files(self, chat_id: str) -> bool:
        """Delete all files associated with a specific chat"""
        try:
            pattern = f"{chat_id}_*"
            for filename in os.listdir(self.upload_folder):
                if filename.startswith(f"{chat_id}_"):
                    os.remove(os.path.join(self.upload_folder, filename))
            return True
        except Exception as e:
            print(f"Error clearing files: {e}")
            return False

    def cleanup_old_files(self, active_files: List[str]) -> None:
        """Remove any files in the upload folder that aren't in the active_files list"""
        try:
            active_files = set(secure_filename(f) for f in active_files)
            for filename in os.listdir(self.upload_folder):
                if filename not in active_files:
                    os.remove(os.path.join(self.upload_folder, filename))
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_file_size(self, filename: str) -> Optional[int]:
        """Get the size of a file in bytes"""
        try:
            filepath = os.path.join(self.upload_folder, secure_filename(filename))
            return os.path.getsize(filepath) if os.path.exists(filepath) else None
        except Exception:
            return None