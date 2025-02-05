from datetime import datetime
import os
import json
from typing import Dict
from document_processor import DocumentProcessor
from file_cache import file_cache


class ChatStorage:
    def __init__(self, storage_dir: str = 'chat_storage'):
        self.file_cache = file_cache  # Make sure file_cache is passed in or globally available
        self.upload_folder = 'uploads'  # Or pass this in through constructor
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _get_user_file(self, user_id: str) -> str:
        return os.path.join(self.storage_dir, f'user_{user_id}.json')

    def _get_next_chat_id(self, existing_chats: Dict) -> str:
        """Generate the next available chat ID based on existing chats"""
        if not existing_chats:
            return 'chat-1'

        # Extract existing chat numbers
        chat_numbers = []
        for chat_id in existing_chats.keys():
            try:
                number = int(chat_id.split('-')[1])
                chat_numbers.append(number)
            except (IndexError, ValueError):
                continue

        # Find the lowest available number
        if not chat_numbers:
            return 'chat-1'

        chat_numbers.sort()
        next_number = 1
        for number in chat_numbers:
            if number == next_number:
                next_number += 1
            elif number > next_number:
                break

        return f'chat-{next_number}'

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
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Pretty print JSON for debugging
        with open(file_path, 'w') as f:
            json.dump(chats, f, indent=2)

    def create_chat(self, user_id: str) -> Dict:
        """Create a new chat for a user"""
        user_data = self.load_user_chats(user_id)
        new_chat_id = self._get_next_chat_id(user_data['chats'])
        chat_number = new_chat_id.split('-')[1]

        new_chat = {
            'messages': [],
            'files': [],
            'name': f'Chat {chat_number}',
            'created_at': datetime.now().isoformat()
        }

        user_data['chats'][new_chat_id] = new_chat
        self.save_user_chats(user_id, user_data)

        return {
            'chatId': new_chat_id,
            'name': new_chat['name']
        }

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
    """Delete a specific chat and its associated files"""
    try:
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return False

        # Don't allow deleting the last chat
        if len(user_data['chats']) <= 1:
            return False

        # Get list of files to delete
        files_to_delete = []
        if 'files' in user_data['chats'][chat_id]:
            files_to_delete = [
                file_info['name']
                for file_info in user_data['chats'][chat_id]['files']
            ]

        # Delete the chat from storage
        del user_data['chats'][chat_id]
        self.save_user_chats(user_id, user_data)

        # Delete associated files
        user_upload_dir = os.path.join(self.upload_folder, user_id)
        if os.path.exists(user_upload_dir):
            for filename in files_to_delete:
                file_path = os.path.join(user_upload_dir, filename)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting file {filename}: {str(e)}")

        return True
    except Exception as e:
        print(f"Error deleting chat: {str(e)}")
        return False

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

    def add_message(self, user_id: str, chat_id: str, content: str, is_user: bool) -> bool:
        """Add a message to a specific chat"""
        try:
            print(f"Adding message for user {user_id}, chat {chat_id}, is_user: {is_user}")
            user_data = self.load_user_chats(user_id)

            if chat_id not in user_data['chats']:
                print(f"Chat {chat_id} not found")
                return False

            message = {
                'content': content,
                'isUser': is_user,
                'timestamp': datetime.now().isoformat()
            }

            # Add message to chat
            if 'messages' not in user_data['chats'][chat_id]:
                user_data['chats'][chat_id]['messages'] = []

            user_data['chats'][chat_id]['messages'].append(message)

            # Save the updated data
            self.save_user_chats(user_id, user_data)

            # Verify the message was saved
            verification = self.load_user_chats(user_id)
            message_saved = any(
                msg['content'] == content and msg['isUser'] == is_user
                for msg in verification['chats'][chat_id]['messages']
            )
            print(f"Message verification: {'successful' if message_saved else 'failed'}")

            return True
        except Exception as e:
            print(f"Error adding message: {str(e)}")
            return False

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