from flask import Flask, request, render_template, redirect, url_for, jsonify, session, Response
import os
import requests
from werkzeug.utils import secure_filename
import json
import hashlib
import shutil
from document_processor import DocumentProcessor
from dotenv import load_dotenv
from chat_manager import ChatManager
from chat_storage import ChatStorage
from file_manager import FileManager
from file_cache import file_cache


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_PDF_SIZE', 16 * 1024 * 1024))  # Default 16MB
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}

chat_storage = ChatStorage()
file_manager = FileManager(app.config['UPLOAD_FOLDER'])

# Simplified user management (for demo purposes)
USERS = {
    'user1': 'password1',
    'user2': 'password2',
    'user3': 'password3',
    'user4': 'password4',
    'user5': 'password5'
}

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('chat_interface'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in USERS and USERS[username] == password:
            session['user_id'] = username
            return redirect(url_for('chat_interface'))

        return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.before_request
def before_request():
    if 'chats' not in session:
        session['chats'] = {
            'chat-1': {
                'messages': [],
                'files': []
            }
        }
        session.modified = True

@app.route('/debug/routes')
def list_routes():
    """List all registered routes for debugging"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify(routes)

# Load and parse API keys from .env
def load_api_configs():
    raw_keys = os.getenv('API_KEYS', '{}')
    try:
        api_keys = json.loads(raw_keys)
    except json.JSONDecodeError:
        app.logger.error("Failed to parse API_KEYS from .env")
        api_keys = {}

    return {
        'deepseek-coder': {
            'url': 'https://openrouter.ai/api/v1/chat/completions',
            'model': 'deepseek/deepseek-r1:free',
            'key': api_keys.get('openrouter'),
            'type': 'openrouter'
        },
        'deepseek-chat': {
            'url': 'https://openrouter.ai/api/v1/chat/completions',
            'model': 'deepseek/deepseek-r1:free',
            'key': api_keys.get('openrouter'),
            'type': 'openrouter'
        },
        'mixtral': {
            'url': 'https://openrouter.ai/api/v1/chat/completions',
            'model': 'mistralai/mixtral-8x7b-instruct',
            'key': api_keys.get('openrouter'),
            'type': 'openrouter'
        },
        'gpt-4': {
            'url': 'https://api.openai.com/v1/chat/completions',
            'model': 'gpt-4-turbo-preview',
            'key': api_keys.get('openai'),
            'type': 'openai'
        },
        'gpt-3.5': {
            'url': 'https://api.openai.com/v1/chat/completions',
            'model': 'gpt-3.5-turbo',
            'key': api_keys.get('openai'),
            'type': 'openai'
        },
        'claude-3': {
            'url': 'https://api.anthropic.com/v1/messages',
            'model': 'claude-3-5-sonnet-20241022',
            'key': api_keys.get('anthropic'),
            'type': 'anthropic'
        }
    }


chat_manager = ChatManager(app.config['UPLOAD_FOLDER'], file_cache)

# Initialize API configurations
API_CONFIGS = load_api_configs()


def clean_upload_folder():
    """Clean the upload folder on application start"""
    upload_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir)


# Clean upload folder on startup
clean_upload_folder()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'txt', 'doc', 'docx'}


def create_upload_directory():
    """Create upload directory if it doesn't exist"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.before_request
def before_request():
    """Ensure upload directory exists and initialize session if needed"""
    create_upload_directory()
    if 'chats' not in session:
        session['chats'] = {
            'chat-1': {
                'messages': [],
                'files': []
            }
        }
        session.modified = True



def get_api_response(api_name, messages, context="", stream=False):
    if api_name not in API_CONFIGS:
        return {"error": "Invalid API selected"}

    config = API_CONFIGS[api_name]
    if not config['key']:
        return {"error": f"API key not found for {api_name}"}

    headers = {}
    if config['type'] == 'openrouter':
        headers = {
            "Authorization": f"Bearer {config['key']}",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Local LLM Chat App",
            "Content-Type": "application/json"
        }
    elif config['type'] == 'openai':
        headers = {
            "Authorization": f"Bearer {config['key']}",
            "Content-Type": "application/json"
        }
    elif config['type'] == 'anthropic':
        headers = {
            "x-api-key": config['key'],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    try:
        if config['type'] == 'anthropic':
            # For Anthropic/Claude, format the context and message differently
            user_message = messages[-1]['content'] if messages else ""
            full_prompt = f"Context from the documents:\n\n{context}\n\nUser question: {user_message}"
            payload = {
                "model": config['model'],
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "max_tokens": 40096,
                "stream": stream
            }
        else:
            # For OpenAI/OpenRouter, include context in system message
            system_message = "You are a helpful assistant. Please analyze the provided context and answer questions about it."
            if context:
                system_message += f"\n\nContext from the documents:\n\n{context}"

            # Construct messages array with system message and user message
            constructed_messages = [{"role": "system", "content": system_message}]
            # Add the actual user message
            if messages and isinstance(messages, list) and len(messages) > 0:
                constructed_messages.extend(messages)

            payload = {
                "model": config['model'],
                "messages": constructed_messages,
                "max_tokens": 40096,
                "stream": stream
            }

            if config['type'] == 'openrouter':
                payload.update({
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "provider": {
                        "sort": "throughput",
                        "allow_fallbacks": True,
                        "require_parameters": True
                    }
                })

        if stream:
            return requests.post(config['url'], headers=headers, json=payload, stream=True)

        response = requests.post(config['url'], headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if config['type'] == 'anthropic':
            return {"response": response_json['content'][0]['text']}
        else:
            return {"response": response_json['choices'][0]['message']['content']}

    except requests.exceptions.RequestException as e:
        app.logger.error(f"API request failed: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            app.logger.error(f"Response text: {e.response.text}")
        return {"error": f"API request failed: {str(e)}"}


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    chat_id = request.form.get('chatId', 'chat-1')

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Create a directory structure that includes the user_id to prevent file name conflicts
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'])
    os.makedirs(user_upload_dir, exist_ok=True)

    # Create unique filename with user and chat info
    base_filename = secure_filename(file.filename)
    unique_filename = f"{chat_id}_{base_filename}"
    file_path = os.path.join(user_upload_dir, unique_filename)

    # Save the file
    try:
        file.save(file_path)

        # Add file to chat storage with relative path
        chat_storage.add_file(
            session['user_id'],
            chat_id,
            os.path.join(session['user_id'], unique_filename),  # Store relative path
            base_filename
        )

        return jsonify({
            "success": True,
            "filename": os.path.join(session['user_id'], unique_filename),
            "displayName": base_filename
        })
    except Exception as e:
        app.logger.error(f"Error saving file: {str(e)}")
        return jsonify({"error": "Failed to save file"}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        chat_id = request.form.get('chatId', 'chat-1')

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        base_filename = secure_filename(file.filename)
        unique_filename = f"{chat_id}_{base_filename}"

        # Save file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))

        # Add file to chat storage
        chat_storage.add_file(session['user_id'], chat_id, unique_filename, base_filename)

        return jsonify({
            "success": True,
            "filename": unique_filename,
            "displayName": base_filename
        })

    return render_template('index.html', apis=list(API_CONFIGS.keys()))


@app.route('/remove-file/<path:filename>', methods=['POST'])
def remove_file(filename):
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json(silent=True) or {}
    chat_id = data.get('chatId', 'chat-1')

    # Verify the file belongs to the user
    if not filename.startswith(session['user_id']):
        return jsonify({"error": "Access denied"}), 403

    # Get the full file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Remove file from storage
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove file from chat storage
        chat_storage.remove_file(session['user_id'], chat_id, filename)

        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error removing file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['GET', 'POST'])
def chat_interface():  # This function name must match what's used in url_for()
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('chat.html', apis=list(API_CONFIGS.keys()))

    # Handle POST request for chat messages
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    user_input = data.get('message', '')
    api_name = data.get('api', 'deepseek-chat')
    chat_id = data.get('chatId', 'chat-1')

    # Store user message
    chat_manager.add_message(
        chat_id,
        user_input,
        True
    )

    # Get API response
    response = get_api_response(api_name, [{"role": "user", "content": user_input}])

    if response.get("response"):
        # Store AI response
        chat_manager.add_message(
            chat_id,
            response["response"],
            False
        )

    return jsonify(response)


@app.teardown_appcontext
def cleanup(error):
    file_cache.clear()


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    try:
        app.logger.info("Starting chat stream request")
        if 'user_id' not in session:
            return jsonify({"error": "Not authenticated"}), 401

        data = request.get_json()
        if not data:
            app.logger.error("No JSON data received")
            return jsonify({"error": "No data received"}), 400

        user_input = data.get('message', '')
        api_name = data.get('api', 'deepseek-chat')
        chat_id = data.get('chatId', 'chat-1')
        user_id = session['user_id']

        app.logger.info(f"Processing request for user {user_id}, chat {chat_id}")

        # Store user message
        success = chat_storage.add_message(user_id, chat_id, user_input, True)
        app.logger.info(f"User message stored: {success}")

        # Get file context for this chat
        try:
            context = chat_storage.get_context(user_id, chat_id)
            app.logger.info(f"Context retrieved, length: {len(context)}")
        except Exception as e:
            app.logger.error(f"Error getting context: {str(e)}")
            return jsonify({"error": "Failed to get context"}), 500

        config = API_CONFIGS[api_name]
        if not config['key']:
            app.logger.error(f"No API key found for {api_name}")
            return jsonify({"error": "API key not found"}), 500

        # Set up appropriate headers based on API type
        headers = {
            "Content-Type": "application/json"
        }

        if config['type'] == 'openrouter':
            headers.update({
                "Authorization": f"Bearer {config['key']}",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Local LLM Chat App"
            })
        elif config['type'] == 'openai':
            headers.update({
                "Authorization": f"Bearer {config['key']}"
            })
        elif config['type'] == 'anthropic':
            headers.update({
                "x-api-key": config['key'],
                "anthropic-version": "2023-06-01"
            })

        # Prepare the payload based on API type
        if config['type'] == 'anthropic':
            full_prompt = f"Context from the documents:\n\n{context}\n\nUser question: {user_input}"
            payload = {
                "model": config['model'],
                "messages": [{
                    "role": "user",
                    "content": full_prompt
                }],
                "max_tokens": 4096,
                "stream": True
            }
        else:
            # For OpenAI and OpenRouter
            system_message = "You are a helpful assistant. Please analyze the provided context and answer questions about it."
            if context:
                system_message += f"\n\nContext from the documents:\n\n{context}"

            messages = [{
                "role": "system",
                "content": system_message
            }, {
                "role": "user",
                "content": user_input
            }]

            payload = {
                "model": config['model'],
                "messages": messages,
                "max_tokens": 4096,
                "stream": True
            }

            if config['type'] == 'openrouter':
                payload.update({
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "provider": {
                        "sort": "throughput",
                        "allow_fallbacks": True,
                        "require_parameters": True
                    }
                })

        app.logger.info(f"Making request to {config['url']}")
        response = requests.post(
            config['url'],
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
        )

        if response.status_code != 200:
            app.logger.error(f"API error: {response.status_code} - {response.text}")
            return jsonify({"error": f"API error: {response.status_code}"}), response.status_code

        app.logger.info("Stream started successfully")

        def generate():
            full_response = ""
            try:
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            app.logger.debug(f"Received line: {line_text}")

                            if line_text.startswith('data: '):
                                data = line_text[6:]
                                if data == '[DONE]':
                                    app.logger.info(f"Stream completed. Final response length: {len(full_response)}")
                                    if full_response:
                                        # Log before storing
                                        app.logger.info(
                                            f"Storing AI response for chat {chat_id}: {full_response[:100]}...")

                                        # Store in chat storage
                                        success = chat_storage.add_message(
                                            user_id,
                                            chat_id,
                                            full_response,
                                            False  # This is an AI response
                                        )
                                        app.logger.info(f"AI response stored: {success}")

                                        # Verify storage
                                        user_data = chat_storage.load_user_chats(user_id)
                                        chat = user_data['chats'].get(chat_id, {})
                                        messages = chat.get('messages', [])
                                        app.logger.info(f"Current chat messages: {len(messages)}")

                                    yield f"data: [DONE]\n\n"
                                    break

                                try:
                                    parsed_data = json.loads(data)

                                    # Handle different API response formats
                                    content = None
                                    if config['type'] == 'anthropic':
                                        content = parsed_data.get('delta', {}).get('text', '')
                                    else:
                                        content = parsed_data.get('choices', [{}])[0].get('delta', {}).get('content',
                                                                                                           '')

                                    if content:
                                        app.logger.debug(f"Received content chunk: {content}")
                                        full_response += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"

                                except json.JSONDecodeError as e:
                                    app.logger.error(f"JSON decode error in data: {str(e)}")
                                    continue

                        except json.JSONDecodeError as e:
                            app.logger.error(f"JSON decode error: {str(e)} - Line: {line}")
                            continue
                        except Exception as e:
                            app.logger.error(f"Error processing stream line: {str(e)}")
                            continue

            except Exception as e:
                app.logger.error(f"Stream processing error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                response.close()

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Unexpected error in chat stream: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat/message', methods=['POST'])
def chat_message():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    user_input = data.get('message', '')
    api_name = data.get('api', 'deepseek-chat')
    chat_id = data.get('chatId', 'chat-1')

    # Store user message
    chat_storage.add_message(session['user_id'], chat_id, user_input, True)

    # Get API response
    response = get_api_response(api_name, [{"role": "user", "content": user_input}])

    if response.get("response"):
        chat_storage.add_message(
            session['user_id'],
            chat_id,
            response["response"],
            False
        )

    return jsonify(response)

@app.route('/create-chat', methods=['POST'])
def create_chat():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    result = chat_storage.create_chat(session['user_id'])
    return jsonify({
        'success': True,
        'chatId': result['chatId']
    })


@app.route('/switch-chat/<chat_id>', methods=['GET'])
def switch_chat(chat_id):
    chat = chat_manager.get_chat(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    return jsonify({
        'success': True,
        'chat': chat
    })


@app.route('/init-chats', methods=['GET'])
def init_chats():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_data = chat_storage.load_user_chats(session['user_id'])
    return jsonify({
        'success': True,
        'chats': user_data['chats']
    })


@app.route('/close-chat/<chat_id>', methods=['POST'])
def close_chat(chat_id):
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session['user_id']

    # First clear associated files
    try:
        file_manager.clear_user_files(chat_id, user_id)
    except Exception as e:
        app.logger.error(f"Error clearing files for chat {chat_id}: {str(e)}")
        return jsonify({'error': 'Failed to clear chat files'}), 500

    # Then delete chat data
    success = chat_storage.delete_chat(user_id, chat_id)
    if not success:
        return jsonify({'error': 'Failed to delete chat'}), 400

    return jsonify({'success': True})


def clear_chat(self, user_id: str, chat_id: str) -> bool:
    """Clear all messages and files from a specific chat"""
    try:
        user_data = self.load_user_chats(user_id)

        if chat_id not in user_data['chats']:
            return False

        # Get list of files to delete
        files_to_delete = []
        if 'files' in user_data['chats'][chat_id]:
            files_to_delete = [
                file_info['name']
                for file_info in user_data['chats'][chat_id]['files']
            ]

        # Clear chat data
        user_data['chats'][chat_id]['messages'] = []
        user_data['chats'][chat_id]['files'] = []
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
        print(f"Error clearing chat: {str(e)}")
        return False


# Modified get_chat_context function
def get_chat_context(chat_id):
    """Get the context from all files in a specific chat"""
    if chat_id not in session.get('chats', {}):
        return ""

    chat = session['chats'][chat_id]
    context_parts = []

    for filename in chat.get('files', []):
        # Try to get content from cache first
        content = file_cache.get(filename)

        if content is None:
            # If not in cache, process the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                try:
                    processor = DocumentProcessor(filepath)
                    content = processor.extract_text()
                    # Store in cache for future use
                    if content:
                        file_cache.set(filename, content)
                except Exception as e:
                    app.logger.error(f"Error processing file {filename}: {str(e)}")
                    continue

        if content:
            context_parts.append(f"Content from {filename}:\n{content}")

    # Combine all contexts with clear separation
    context = "\n\n---\n\n".join(context_parts)

    # Limit total context length while trying to keep complete documents
    max_length = 10000
    if len(context) > max_length:
        # Try to find a good breaking point
        break_point = context.rfind("\n\n---\n\n", 0, max_length)
        if break_point > 0:
            context = context[:break_point]
        else:
            context = context[:max_length]

    return context
@app.route('/chat/<chat_id>/files', methods=['GET'])
def get_chat_files(chat_id):
        """Get all files associated with a specific chat"""
        if chat_id not in session['chats']:
            return jsonify({'error': 'Chat not found'}), 404

        files = []
        for filename in session['chats'][chat_id].get('files', []):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                files.append({
                    'name': filename,
                    'size': os.path.getsize(filepath)
                })

        return jsonify({
            'success': True,
            'files': files
        })

@app.route('/chat/<chat_id>/messages', methods=['GET'])
def get_chat_messages(chat_id):
        """Get all messages in a specific chat"""
        if chat_id not in session['chats']:
            return jsonify({'error': 'Chat not found'}), 404

        return jsonify({
            'success': True,
            'messages': session['chats'][chat_id].get('messages', [])
        })

@app.route('/chat/<chat_id>/rename', methods=['POST'])
def rename_chat(chat_id):
        """Rename a chat"""
        if chat_id not in session['chats']:
            return jsonify({'error': 'Chat not found'}), 404

        data = request.get_json()
        new_name = data.get('name')

        if not new_name:
            return jsonify({'error': 'New name not provided'}), 400

        session['chats'][chat_id]['name'] = new_name
        session.modified = True

        return jsonify({
            'success': True,
            'chatId': chat_id,
            'name': new_name
        })

@app.errorhandler(413)
def request_entity_too_large(error):
        """Handle file size exceeded error"""
        return jsonify({
            'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)}MB'
        }), 413

@app.errorhandler(500)
def internal_server_error(error):
        """Handle internal server errors"""
        app.logger.error(f"Internal server error: {str(error)}")
        return jsonify({
            'error': 'An internal server error occurred. Please try again later.'
        }), 500

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)