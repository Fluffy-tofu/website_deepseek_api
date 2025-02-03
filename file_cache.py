import os
import hashlib

class FileCache:
    _instance = None

    def __new__(cls, cache_dir='file_cache'):
        if cls._instance is None:
            cls._instance = super(FileCache, cls).__new__(cls)
            cls._instance.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        return cls._instance

    def get_cache_path(self, filename):
        """Get the cache path for a file"""
        file_path = os.path.join('uploads', filename)  # Assuming 'uploads' is your upload directory
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            cache_key = f"{filename}_{mtime}"
            hash_key = hashlib.md5(cache_key.encode()).hexdigest()
            return os.path.join(self.cache_dir, f"{hash_key}.txt")
        return None

    def get(self, filename):
        """Get content from cache"""
        cache_path = self.get_cache_path(filename)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Cache read error: {str(e)}")
        return None

    def set(self, filename, content):
        """Set content in cache"""
        cache_path = self.get_cache_path(filename)
        if cache_path:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                print(f"Cache write error: {str(e)}")

    def clear(self):
        """Clear all cached files"""
        try:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
        except Exception as e:
            print(f"Cache clear error: {str(e)}")

# Create a global instance
file_cache = FileCache()