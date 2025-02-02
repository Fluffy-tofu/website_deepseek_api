from flask import Flask, request, render_template, redirect, url_for, jsonify, session, Response
import os
import requests
from werkzeug.utils import secure_filename
import json
import hashlib
import shutil
from document_processor import DocumentProcessor
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_PDF_SIZE', 16 * 1024 * 1024))  # Default 16MB
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}


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

class FileCache:
    def __init__(self, cache_dir='file_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, filename):
        # Create a unique cache key based on filename and last modification time
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            cache_key = f"{filename}_{mtime}"
            hash_key = hashlib.md5(cache_key.encode()).hexdigest()
            return os.path.join(self.cache_dir, f"{hash_key}.txt")
        return None

    def get(self, filename):
        cache_path = self.get_cache_path(filename)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                app.logger.error(f"Cache read error: {str(e)}")
        return None

    def set(self, filename, content):
        cache_path = self.get_cache_path(filename)
        if cache_path:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                app.logger.error(f"Cache write error: {str(e)}")

    def clear(self):
        try:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
        except Exception as e:
            app.logger.error(f"Cache clear error: {str(e)}")

file_cache = FileCache()

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_upload_directory():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.before_request
def before_request():
    create_upload_directory()
    if 'chats' not in session:
        session['chats'] = {
            'chat-1': {
                'messages': [],
                'files': []
            }
        }


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


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        chat_id = request.form.get('chatId', 'chat-1')

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            # Initialize chat storage
            if 'chats' not in session:
                session['chats'] = {}
            if chat_id not in session['chats']:
                session['chats'][chat_id] = {'messages': [], 'files': []}
            if 'files' not in session['chats'][chat_id]:
                session['chats'][chat_id]['files'] = []

            # Add file to chat's files list
            if filename not in session['chats'][chat_id]['files']:
                session['chats'][chat_id]['files'].append(filename)
                session.modified = True

            # Process file and store in cache
            try:
                processor = DocumentProcessor(filepath)
                content = processor.extract_text()
                if content:
                    file_cache.set(filename, content)
            except Exception as e:
                app.logger.error(f"Error processing file content: {str(e)}")

            return jsonify({
                "success": True,
                "filename": filename,
                "chatId": chat_id
            })

        except Exception as e:
            app.logger.error(f"Error saving file: {str(e)}")
            return jsonify({"error": "Failed to save file"}), 500

    return render_template('index.html', apis=list(API_CONFIGS.keys()))

@app.route('/remove-file/<filename>', methods=['POST'])
def remove_file(filename):
    try:
        # Get chat ID from request body, default to 'chat-1' if not provided
        data = request.get_json(silent=True) or {}
        chat_id = data.get('chatId', 'chat-1')

        # First try to remove the file from the filesystem
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)

        # Then remove from session storage
        if chat_id in session.get('chats', {}):
            chat = session['chats'][chat_id]
            if 'files' in chat and filename in chat['files']:
                chat['files'].remove(filename)
                session.modified = True

        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Error removing file: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('message', '')
        api_name = data.get('api', 'deepseek-chat')
        chat_id = data.get('chatId', 'chat-1')

        context = get_chat_context(chat_id)
        messages = [{"role": "user", "content": user_input}]
        response = get_api_response(api_name, messages, context)

        if response.get("response"):
            if chat_id in session['chats']:
                session['chats'][chat_id]['messages'].append({
                    "content": user_input,
                    "isUser": True
                })
                session['chats'][chat_id]['messages'].append({
                    "content": response["response"],
                    "isUser": False
                })
                session.modified = True

        return jsonify(response)

    return render_template('chat.html', apis=list(API_CONFIGS.keys()))

@app.teardown_appcontext
def cleanup(error):
    file_cache.clear()


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    try:
        app.logger.info("Starting chat stream request")
        data = request.get_json()

        if not data:
            app.logger.error("No JSON data received")
            return jsonify({"error": "No data received"}), 400

        user_input = data.get('message', '')
        api_name = data.get('api', 'deepseek-chat')
        chat_id = data.get('chatId', 'chat-1')

        app.logger.info(f"Processing request - API: {api_name}, Chat ID: {chat_id}")

        # Get file context for this chat
        try:
            context = get_chat_context(chat_id)
            app.logger.info(f"Context retrieved, length: {len(context)}")
        except Exception as e:
            app.logger.error(f"Error getting context: {str(e)}")
            return jsonify({"error": "Failed to get context"}), 500

        # Construct messages
        messages = [{
            "role": "system",
            "content": f"Here is the context from the uploaded documents:\n\n{context}"
        }, {
            "role": "user",
            "content": user_input
        }]

        app.logger.info("Initiating API request")
        try:
            config = API_CONFIGS[api_name]
            if not config['key']:
                app.logger.error(f"No API key found for {api_name}")
                return jsonify({"error": "API key not found"}), 500

            headers = {
                "Authorization": f"Bearer {config['key']}",
                "Content-Type": "application/json"
            }

            if config['type'] == 'openrouter':
                headers.update({
                    "HTTP-Referer": "http://localhost:5000",
                    "X-Title": "Local LLM Chat App"
                })
            elif config['type'] == 'anthropic':
                headers = {
                    "x-api-key": config['key'],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }

            # Prepare payload based on API type
            if config['type'] == 'anthropic':
                payload = {
                    "model": config['model'],
                    "messages": [{
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {user_input}"
                    }],
                    "max_tokens": 4096,
                    "stream": True
                }
            else:
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
                timeout=30  # Add timeout
            )

            if response.status_code != 200:
                app.logger.error(f"API error: {response.status_code} - {response.text}")
                return jsonify({"error": f"API error: {response.status_code}"}), response.status_code

            app.logger.info("Stream started successfully")

            def generate():
                try:
                    for line in response.iter_lines():
                        if line:
                            try:
                                line_text = line.decode('utf-8')
                                if line_text.startswith('data: '):
                                    data = line_text[6:]
                                    if data == '[DONE]':
                                        yield f"data: [DONE]\n\n"
                                        break

                                    parsed_data = json.loads(data)

                                    # Handle different API response formats
                                    if config['type'] == 'anthropic':
                                        content = parsed_data.get('delta', {}).get('text', '')
                                    else:
                                        content = parsed_data.get('choices', [{}])[0].get('delta', {}).get('content',
                                                                                                           '')

                                    if content:
                                        yield f"data: {json.dumps({'content': content})}\n\n"

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

        except requests.exceptions.RequestException as e:
            app.logger.error(f"Request failed: {str(e)}")
            return jsonify({"error": f"Request failed: {str(e)}"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in chat stream: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/create-chat', methods=['POST'])
def create_chat():
    chat_id = f"chat-{len(session['chats']) + 1}"
    session['chats'][chat_id] = {
        'messages': [],
        'files': []
    }
    session.modified = True
    return jsonify({
        'success': True,
        'chatId': chat_id
    })


@app.route('/switch-chat/<chat_id>', methods=['GET'])
def switch_chat(chat_id):
    if chat_id not in session['chats']:
        return jsonify({'error': 'Chat not found'}), 404

    return jsonify({
        'success': True,
        'chat': session['chats'][chat_id]
    })


@app.route('/close-chat/<chat_id>', methods=['POST'])
def close_chat(chat_id):
    if chat_id not in session['chats']:
        return jsonify({'error': 'Chat not found'}), 404

    # Remove chat's files
    for filename in session['chats'][chat_id].get('files', []):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)

    del session['chats'][chat_id]
    session.modified = True
    return jsonify({'success': True})


@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        # Clear all uploaded files
        clean_upload_folder()

        # Reset session chats
        session['chats'] = {
            'chat-1': {
                'messages': [],
                'files': []
            }
        }
        session.modified = True

        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({"error": str(e)}), 500



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