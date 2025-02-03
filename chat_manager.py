from typing import Dict, List, Optional
import os
from werkzeug.utils import secure_filename
from flask import session
from document_processor import DocumentProcessor


class ChatManager:
    """
    Manages chat contexts, messages, and associated files.
    Ensures proper isolation between different chats and handles context organization.
    """

    def __init__(self, upload_folder: str, file_cache):
        self.upload_folder = upload_folder
        self.file_cache = file_cache

    def initialize_session(self) -> None:
        """Initialize chat session if not already present"""
        if 'chats' not in session:
            session['chats'] = {
                'chat-1': {
                    'messages': [],
                    'files': [],
                    'name': 'Chat 1'
                }
            }
            session.modified = True

    def create_chat(self) -> Dict:
        """Create a new chat and return its details"""
        chat_id = f"chat-{len(session['chats']) + 1}"
        session['chats'][chat_id] = {
            'messages': [],
            'files': [],
            'name': f"Chat {len(session['chats']) + 1}"
        }
        session.modified = True
        return {
            'id': chat_id,
            'name': session['chats'][chat_id]['name']
        }

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get chat data by ID"""
        return session['chats'].get(chat_id)

    def add_message(self, chat_id: str, content: str, is_user: bool) -> bool:
        """Add a message to a specific chat"""
        if chat_id not in session['chats']:
            return False

        session['chats'][chat_id]['messages'].append({
            'content': content,
            'isUser': is_user
        })
        session.modified = True
        return True

    def get_messages(self, chat_id: str) -> List:
        """Get all messages for a specific chat"""
        return session['chats'].get(chat_id, {}).get('messages', [])

    def add_file(self, chat_id: str, file, unique_filename: str) -> Dict:
        """Add a file to a specific chat and process it"""
        if chat_id not in session['chats']:
            return {'error': 'Chat not found'}

        try:
            filepath = os.path.join(self.upload_folder, unique_filename)

            # Save file
            file.save(filepath)

            # Process and cache file content
            processor = DocumentProcessor(filepath)
            content = processor.extract_text()
            if content:
                self.file_cache.set(unique_filename, content)

            # Add to chat's files
            if unique_filename not in session['chats'][chat_id]['files']:
                session['chats'][chat_id]['files'].append(unique_filename)
                session.modified = True

            return {
                'success': True,
                'filename': unique_filename
            }
        except Exception as e:
            return {'error': str(e)}

    def remove_file(self, chat_id: str, filename: str) -> Dict:
        """Remove a file from a specific chat"""
        if chat_id not in session['chats']:
            return {'error': 'Chat not found'}

        try:
            filepath = os.path.join(self.upload_folder, secure_filename(filename))
            if os.path.exists(filepath):
                os.remove(filepath)

            if filename in session['chats'][chat_id]['files']:
                session['chats'][chat_id]['files'].remove(filename)
                session.modified = True

            return {'success': True}
        except Exception as e:
            return {'error': str(e)}

    def clear_chat(self, chat_id: str) -> Dict:
        """Clear all messages and files from a specific chat"""
        if chat_id not in session['chats']:
            return {'error': 'Chat not found'}

        try:
            # Remove all files
            for filename in session['chats'][chat_id]['files']:
                filepath = os.path.join(self.upload_folder, secure_filename(filename))
                if os.path.exists(filepath):
                    os.remove(filepath)

            # Reset chat data
            session['chats'][chat_id]['messages'] = []
            session['chats'][chat_id]['files'] = []
            session.modified = True

            return {'success': True}
        except Exception as e:
            return {'error': str(e)}

    def delete_chat(self, chat_id: str) -> Dict:
        """Delete a chat and its associated files"""
        if chat_id not in session['chats']:
            return {'error': 'Chat not found'}

        try:
            # Remove all files first
            for filename in session['chats'][chat_id]['files']:
                filepath = os.path.join(self.upload_folder, secure_filename(filename))
                if os.path.exists(filepath):
                    os.remove(filepath)

            # Remove chat from session
            del session['chats'][chat_id]
            session.modified = True

            return {'success': True}
        except Exception as e:
            return {'error': str(e)}

    def get_context(self, chat_id: str, max_length: int = 10000) -> str:
        """Get organized context for a specific chat"""
        if chat_id not in session['chats']:
            return ""

        context_parts = []

        # Add recent messages context first
        messages = session['chats'][chat_id]['messages']
        if messages:
            recent_messages = messages[-5:]  # Get last 5 messages for context
            message_context = "\n".join([
                f"{'User' if msg['isUser'] else 'Assistant'}: {msg['content']}"
                for msg in recent_messages
            ])
            context_parts.append(f"Recent conversation:\n{message_context}")

        # Add file contexts
        for filename in session['chats'][chat_id]['files']:
            content = self.file_cache.get(filename)
            if not content:
                filepath = os.path.join(self.upload_folder, filename)
                if os.path.exists(filepath):
                    try:
                        processor = DocumentProcessor(filepath)
                        content = processor.extract_text()
                        if content:
                            self.file_cache.set(filename, content)
                    except Exception:
                        continue

            if content:
                context_parts.append(f"Content from {filename}:\n{content}")

        # Combine contexts with clear separation
        context = "\n\n---\n\n".join(context_parts)

        # Smart context truncation
        if len(context) > max_length:
            # Try to find a good breaking point
            break_point = context.rfind("\n\n---\n\n", 0, max_length)
            if break_point > 0:
                context = context[:break_point]
            else:
                # Fallback to hard truncation
                context = context[:max_length]

        return context

    def rename_chat(self, chat_id: str, new_name: str) -> Dict:
        """Rename a chat"""
        if chat_id not in session['chats']:
            return {'error': 'Chat not found'}

        session['chats'][chat_id]['name'] = new_name
        session.modified = True
        return {
            'success': True,
            'chatId': chat_id,
            'name': new_name
        }