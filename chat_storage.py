from datetime import datetime
import os
import json
from typing import Dict
from document_processor import DocumentProcessor
from file_cache import file_cache


class ChatStorage:
    def __init__(self, storage_dir: str = 'chat_storage'):
        self.storage_dir = storage_dir
        self.file_cache = file_cache  # Make sure file_cache is passed in or globally available
        self.upload_folder = 'uploads'  # Or pass this in through constructor
        os.makedirs(storage_dir, exist_ok=True)

    def _get_user_file(self, user_id: str) -> str:
        """Get the path to a user's chat storage file"""
        return os.path.join(self.storage_dir, f'user_{user_id}.json')

    def load_user_chats(self, user_id: str) -> Dict:
        """Load all chats for a user"""
        file_path = self._get_user_file(user_id)
        if not os.path.exists(file_path):
            return {
                'chats': {
                    'chat-1': {
                        'messages': [],
                        'files': [],
                        'name': 'Chat 1',
                        'created_at': datetime.now().isoformat()
                    }
                }
            }

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {'chats': {}}

    def save_user_chats(self, user_id: str, chats: Dict) -> None:
        """Save all chats for a user"""
        file_path = self._get_user_file(user_id)
        with open(file_path, 'w') as f:
            json.dump(chats, f, indent=2)

    def create_chat(self, user_id: str) -> Dict:
        """Create a new chat for a user"""
        user_data = self.load_user_chats(user_id)
        chat_count = len(user_data['chats'])
        new_chat_id = f'chat-{chat_count + 1}'

        new_chat = {
            'messages': [],
            'files': [],
            'name': f'Chat {chat_count + 1}',
            'created_at': datetime.now().isoformat()
        }

        user_data['chats'][new_chat_id] = new_chat
        self.save_user_chats(user_id, user_data)

        return {
            'chatId': new_chat_id,
            'name': new_chat['name']
        }

    def add_message(self, user_id: str, chat_id: str, content: str, is_user: bool) -> bool:
        """Add a message to a specific chat"""
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return False

        message = {
            'content': content,
            'isUser': is_user,
            'timestamp': datetime.now().isoformat()
        }

        user_data['chats'][chat_id]['messages'].append(message)
        self.save_user_chats(user_id, user_data)
        return True

    def add_file(self, user_id: str, chat_id: str, filename: str, display_name: str) -> bool:
        """Add a file reference to a specific chat"""
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return False

        file_info = {
            'name': filename,
            'displayName': display_name,
            'uploaded_at': datetime.now().isoformat()
        }

        user_data['chats'][chat_id]['files'].append(file_info)
        self.save_user_chats(user_id, user_data)
        return True

    def remove_file(self, user_id: str, chat_id: str, filename: str) -> bool:
        """Remove a file reference from a specific chat"""
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return False

        user_data['chats'][chat_id]['files'] = [
            f for f in user_data['chats'][chat_id]['files']
            if f['name'] != filename
        ]

        self.save_user_chats(user_id, user_data)
        return True

    def clear_chat(self, user_id: str, chat_id: str) -> bool:
        """Clear all messages and files from a specific chat"""
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return False

        user_data['chats'][chat_id]['messages'] = []
        user_data['chats'][chat_id]['files'] = []
        self.save_user_chats(user_id, user_data)
        return True

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        """Delete a specific chat"""
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return False

        del user_data['chats'][chat_id]
        self.save_user_chats(user_id, user_data)
        return True

    def get_context(self, user_id: str, chat_id: str, max_length: int = 10000) -> str:
        """Get the context for a specific chat including messages and file contents"""
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return ""

        context_parts = []

        # Add recent messages context
        messages = user_data['chats'][chat_id].get('messages', [])
        if messages:
            recent_messages = messages[-5:]  # Get last 5 messages
            message_context = "\n".join([
                f"{'User' if msg['isUser'] else 'Assistant'}: {msg['content']}"
                for msg in recent_messages
            ])
            context_parts.append(f"Recent conversation:\n{message_context}")

        # Add file contents
        for file_info in user_data['chats'][chat_id].get('files', []):
            if not file_info:
                continue

            filename = file_info.get('name')
            if not filename:
                continue

            # Try to get content from cache first
            content = self.file_cache.get(filename)

            # If not in cache, process the file
            if not content:
                filepath = os.path.join(self.upload_folder, filename)
                if os.path.exists(filepath):
                    try:
                        processor = DocumentProcessor(filepath)
                        content = processor.extract_text()
                        if content:
                            self.file_cache.set(filename, content)
                    except Exception as e:
                        print(f"Error processing file {filename}: {str(e)}")
                        continue

            if content:
                context_parts.append(f"Content from {filename}:\n{content}")

        # Combine all contexts
        context = "\n\n---\n\n".join(context_parts)

        # Handle length limitation
        if len(context) > max_length:
            # Try to find a good breaking point
            break_point = context.rfind("\n\n---\n\n", 0, max_length)
            if break_point > 0:
                context = context[:break_point]
            else:
                context = context[:max_length]

        return context