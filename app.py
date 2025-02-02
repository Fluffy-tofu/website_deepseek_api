from flask import Flask, request, render_template, redirect, url_for, jsonify, session, Response
import os
import requests
from document_processor import DocumentProcessor
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json

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
            'model': 'claude-3-opus-20240229',
            'key': api_keys.get('anthropic'),
            'type': 'anthropic'
        }
    }


# Initialize API configurations
API_CONFIGS = load_api_configs()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Create uploads directory on startup
def create_upload_directory():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Initialize upload directory when the app starts
create_upload_directory()


@app.before_request
def before_request():
    """Ensure upload directory exists before each request"""
    create_upload_directory()


def get_api_response(api_name, messages, context=""):
    if api_name not in API_CONFIGS:
        return {"error": "Invalid API selected"}

    config = API_CONFIGS[api_name]
    if not config['key']:
        return {"error": f"API key not found for {api_name}"}

    # Prepare headers based on API type
    headers = {}
    if config['type'] == 'openrouter':
        headers = {
            "Authorization": f"Bearer {config['key']}",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Local LLM Chat App"
        }
    elif config['type'] == 'openai':
        headers = {"Authorization": f"Bearer {config['key']}"}
    elif config['type'] == 'anthropic':
        headers = {
            "x-api-key": config['key'],
            "anthropic-version": "2023-06-01"
        }

    try:
        # Prepare payload based on API type
        if config['type'] == 'openrouter':
            payload = {
                "model": config['model'],
                "messages": [
                    {"role": "system", "content": f"Context from the uploaded document: {context}"},
                    *messages
                ],
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "provider": {
                    "sort": "throughput",
                    "allow_fallbacks": True,
                    "require_parameters": True
                }
            }
        elif config['type'] == 'anthropic':
            payload = {
                "model": config['model'],
                "messages": [
                    {
                        "role": "user",
                        "content": f"Context from the document: {context}\n\nUser question: {messages[-1]['content']}"
                    }
                ],
                "max_tokens": 4096
            }
        else:  # OpenAI
            payload = {
                "model": config['model'],
                "messages": [
                    {"role": "system", "content": f"Context from the uploaded document: {context}"},
                    *messages
                ],
                "max_tokens": 4096
            }

        # Make the API request
        response = requests.post(config['url'], headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        # Parse response based on API type
        if config['type'] == 'anthropic':
            return {"response": response_json['content'][0]['text']}
        else:  # OpenRouter and OpenAI use similar response formats
            return {"response": response_json['choices'][0]['message']['content']}

    except requests.exceptions.RequestException as e:
        app.logger.error(f"API request failed: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            app.logger.error(f"Response text: {e.response.text}")
        return {"error": f"API request failed: {str(e)}"}


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Validate file type
            if not allowed_file(file.filename):
                return jsonify({
                    "error": f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400

            # Create a safe filename and ensure upload directory exists
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file
            try:
                file.save(filepath)
            except IOError as e:
                app.logger.error(f"Failed to save file: {str(e)}")
                return jsonify({"error": "Failed to save file"}), 500

            # Store the file path in session
            session['current_file'] = filepath
            return jsonify({
                "success": True,
                "filename": filename,
                "redirect": url_for('chat')
            })

        except Exception as e:
            app.logger.error(f"Error processing upload: {str(e)}")
            return jsonify({"error": "Server error processing upload"}), 500

    return render_template('index.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    context = ""
    if 'current_file' in session and os.path.exists(session['current_file']):
        processor = DocumentProcessor(session['current_file'])
        context = processor.extract_text()[:3000]  # Limit context length

    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('message', '')
        api_name = data.get('api', 'deepseek-chat')

        messages = [{"role": "user", "content": user_input}]
        response = get_api_response(api_name, messages, context)

        return jsonify(response)

    return render_template('chat.html', apis=list(API_CONFIGS.keys()))


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Endpoint for streaming chat responses"""
    try:
        data = request.get_json()
        if not data:
            app.logger.error("No JSON data received")
            return jsonify({"error": "No data received"}), 400

        user_input = data.get('message', '')
        api_name = data.get('api', 'deepseek-chat')
        context = ""

        if 'current_file' in session and os.path.exists(session['current_file']):
            processor = DocumentProcessor(session['current_file'])
            context = processor.extract_text()[:3000]

        config = API_CONFIGS[api_name]
        if not config['key']:
            return jsonify({"error": f"API key not found for {api_name}"})

        # Prepare headers for OpenRouter
        headers = {
            "Authorization": f"Bearer {config['key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",  # Required by OpenRouter
            "X-Title": "Local LLM Chat"  # Required by OpenRouter
        }

        # Prepare the messages array
        messages = [
            {"role": "system", "content": f"Context from the uploaded document: {context}"},
            {"role": "user", "content": user_input}
        ]

        if api_name == "deepseek-chat" or api_name == "deepseek-coder":
            # Prepare payload according to OpenRouter's API specification
            payload = {
                "model": config['model'],
                "messages": messages,
                "stream": True,
                # OpenRouter recommended parameters
                "temperature": 0.7,
                "max_tokens": 4096,  # Request maximum possible tokens
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                # Provider preferences to maximize output tokens
                "provider": {
                    "sort": "throughput",  # Prioritize providers with better performance
                }
            }

        else:
            payload = {
                "model": config['model'],
                "messages": messages,
                "stream": True,
                # OpenRouter recommended parameters
                "temperature": 0.7,
                "max_tokens": 4096,  # Request maximum possible tokens
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }

    except Exception as e:
        print(e)

    def generate():
        try:
            app.logger.info(f"Making streaming request to {config['url']}")
            response = requests.post(
                config['url'],
                headers=headers,
                json=payload,
                stream=True
            )

            if response.status_code != 200:
                app.logger.error(f"API error: {response.status_code} - {response.text}")
                yield f"data: {json.dumps({'error': f'API error: {response.status_code}'})}\n\n"
                return

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    line = line.decode('utf-8')
                    if not line.startswith('data: '):
                        continue

                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        yield f"data: {json.dumps({'content': ''})}\n\n"
                        break

                    json_data = json.loads(data)
                    if 'choices' not in json_data:
                        app.logger.error(f"Unexpected response format: {json_data}")
                        continue

                    delta = json_data['choices'][0].get('delta', {})
                    content = delta.get('content', '')

                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"

                except json.JSONDecodeError as e:
                    app.logger.error(f"JSON decode error: {str(e)} - Line: {line}")
                    continue
                except Exception as e:
                    app.logger.error(f"Error processing stream: {str(e)}")
                    continue

        except requests.exceptions.RequestException as e:
            app.logger.error(f"Request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                app.logger.error(f"Response text: {e.response.text}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            app.logger.error(f"Unexpected error: {str(e)}")
            yield f"data: {json.dumps({'error': 'An unexpected error occurred'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    if 'current_file' in session:
        try:
            os.remove(session['current_file'])
        except OSError:
            pass
        session.pop('current_file', None)
    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)