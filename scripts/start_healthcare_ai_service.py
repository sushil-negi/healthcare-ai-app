#!/usr/bin/env python3
"""
Healthcare AI Service with Advanced Response Generation
Port: 8091 (to avoid conflicts)
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add model path to system
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../models/healthcare-ai/src")
)

try:
    # Try to use the trained model engine first
    from healthcare_trained_engine import HealthcareTrainedEngine

    USE_TRAINED_MODEL = True
except ImportError:
    try:
        from healthcare_ai_engine import HealthcareAIEngine

        USE_TRAINED_MODEL = False
    except ImportError:
        print("Error: Could not import Healthcare engines")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("healthcare_ai_service.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class HealthcareAIHandler(BaseHTTPRequestHandler):
    """HTTP handler for healthcare AI chatbot"""

    def __init__(self, *args, **kwargs):
        # Initialize AI engine (shared across requests)
        if not hasattr(self.__class__, "ai_engine"):
            logger.info("Initializing Healthcare AI Engine...")
            if USE_TRAINED_MODEL:
                logger.info("Using trained ML model engine")
                self.__class__.ai_engine = HealthcareTrainedEngine()
                stats = self.__class__.ai_engine.get_stats()
            else:
                logger.info("Using knowledge base engine")
                self.__class__.ai_engine = HealthcareAIEngine(use_llm=True)
                stats = (
                    self.__class__.ai_engine.get_stats()
                    if USE_TRAINED_MODEL
                    else self.__class__.ai_engine.get_conversation_stats()
                )
            logger.info(f"AI Engine initialized: {stats}")
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            stats = (
                self.__class__.ai_engine.get_stats()
                if USE_TRAINED_MODEL
                else self.__class__.ai_engine.get_conversation_stats()
            )
            response = {
                "status": "healthy",
                "service": "Healthcare AI Assistant",
                "version": "3.0.0",
                "engine_stats": stats,
                "timestamp": datetime.now().isoformat(),
            }
            self.wfile.write(json.dumps(response, indent=2).encode())

        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "service": "Healthcare AI Assistant",
                "version": "3.0.0",
                "description": "Advanced healthcare assistant with LLM and 525K conversation knowledge base",
                "endpoints": {
                    "health": "/health",
                    "chat": "/chat (POST)",
                    "stats": "/stats",
                    "chat_ui": "/chat.html",
                },
            }
            self.wfile.write(json.dumps(response, indent=2).encode())

        elif self.path == "/stats":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            stats = (
                self.__class__.ai_engine.get_stats()
                if USE_TRAINED_MODEL
                else self.__class__.ai_engine.get_conversation_stats()
            )
            self.wfile.write(json.dumps(stats, indent=2).encode())

        elif self.path == "/chat.html":
            self.serve_chat_interface()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/chat":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))

                user_message = data.get("message", "").strip()
                if not user_message:
                    self.send_error(400, "Message is required")
                    return

                # Generate response using AI engine
                logger.info(f"Processing message: {user_message[:100]}...")
                result = self.__class__.ai_engine.generate_response(user_message)

                # Format response
                response_data = {
                    "response": result["response"],
                    "category": result["category"],
                    "confidence": result["confidence"],
                    "method": result["method"],
                    "generation_time": result["generation_time"],
                    "timestamp": datetime.now().isoformat(),
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())

                logger.info(
                    f"Response generated in {result['generation_time']:.2f}s using {result['method']}"
                )

            except Exception as e:
                logger.error(f"Error processing chat request: {e}")
                logger.error(traceback.format_exc())
                self.send_error(500, str(e))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def serve_chat_interface(self):
        """Serve the chat interface"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare AI Assistant</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { margin: 0 0 10px 0; }
        .header p { margin: 0; opacity: 0.9; }
        .stats {
            background: #f8f9fa;
            padding: 15px 30px;
            border-bottom: 1px solid #e1e5e9;
            font-size: 14px;
            text-align: center;
        }
        .chat-container {
            height: 450px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin: 15px 0;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
            position: relative;
        }
        .user {
            background: #667eea;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .bot {
            background: white;
            border: 1px solid #e1e5e9;
            border-bottom-left-radius: 4px;
        }
        .bot-meta {
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e5e9;
        }
        .examples {
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .example-btn {
            padding: 8px 12px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 15px;
            cursor: pointer;
            font-size: 13px;
            text-align: center;
            transition: all 0.2s;
        }
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
        .input-group {
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }
        button:hover {
            background: #5a67d8;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .loading {
            color: #666;
            font-style: italic;
        }
        .error {
            color: #dc3545;
            padding: 10px;
            background: #f8d7da;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Healthcare AI Assistant</h1>
            <p>Advanced AI with 525K+ Healthcare Conversations</p>
        </div>
        
        <div class="stats" id="stats">
            Loading AI engine stats...
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message bot">
                <strong>Healthcare AI:</strong> Hello! I'm an advanced healthcare AI assistant trained on over 525,000 healthcare conversations. I can provide unique, contextual responses about:
                <br><br>
                🏠 Activities of Daily Living (ADL)<br>
                👥 Senior Care & Aging Support<br>
                🧠 Mental Health & Wellness<br>
                🤝 Respite Care & Caregiver Support<br>
                ♿ Disabilities Support & Accessibility<br>
                <br>
                Each response is generated specifically for your question. How can I help you today?
            </div>
        </div>
        
        <div class="input-area">
            <div class="examples">
                <button class="example-btn" onclick="setInput('I need help transferring from bed to wheelchair safely')">🏠 Bed Transfer</button>
                <button class="example-btn" onclick="setInput('My mother with dementia keeps forgetting to take medications')">👵 Medication Management</button>
                <button class="example-btn" onclick="setInput('I feel overwhelmed caring for my disabled spouse')">🤝 Caregiver Stress</button>
                <button class="example-btn" onclick="setInput('What exercises can help with balance and fall prevention?')">🚶 Fall Prevention</button>
                <button class="example-btn" onclick="setInput('How do I apply for disability benefits and support services?')">♿ Disability Benefits</button>
            </div>
            
            <div class="input-group">
                <input type="text" id="userInput" placeholder="Ask your healthcare question..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()" id="sendBtn">Send</button>
            </div>
            
            <div id="error" style="display: none;" class="error"></div>
        </div>
    </div>
    
    <script>
        const API_URL = window.location.origin;
        
        // Load stats on page load
        async function loadStats() {
            try {
                const response = await fetch(`${API_URL}/stats`);
                const stats = await response.json();
                document.getElementById('stats').innerHTML = `
                    AI Engine: ${stats.llm_enabled ? 'LLM + Knowledge Base' : 'Knowledge Base'} | 
                    Training: ${stats.total_training_conversations.toLocaleString()} conversations | 
                    Loaded: ${stats.loaded_conversations.toLocaleString()} | 
                    Categories: ${stats.categories.length}
                `;
            } catch (e) {
                document.getElementById('stats').innerHTML = 'AI Engine Active';
            }
        }
        
        function setInput(text) {
            document.getElementById('userInput').value = text;
            document.getElementById('userInput').focus();
        }
        
        function addMessage(message, isUser, metadata = null) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'bot');
            
            let content = '<strong>' + (isUser ? 'You:' : 'Healthcare AI:') + '</strong> ';
            content += message.replace(/\\n/g, '<br>');
            
            if (metadata && !isUser) {
                content += `<div class="bot-meta">
                    Category: ${metadata.category} | 
                    Confidence: ${(metadata.confidence * 100).toFixed(0)}% | 
                    Method: ${metadata.method} | 
                    Time: ${metadata.generation_time.toFixed(2)}s
                </div>`;
            }
            
            div.innerHTML = content;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const sendBtn = document.getElementById('sendBtn');
            const errorDiv = document.getElementById('error');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, true);
            input.value = '';
            sendBtn.disabled = true;
            errorDiv.style.display = 'none';
            
            // Show loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message bot loading';
            loadingDiv.innerHTML = '<strong>Healthcare AI:</strong> <span class="loading">Generating personalized response...</span>';
            document.getElementById('chatContainer').appendChild(loadingDiv);
            
            try {
                const response = await fetch(`${API_URL}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                // Remove loading message
                loadingDiv.remove();
                
                // Add AI response with metadata
                addMessage(data.response, false, {
                    category: data.category,
                    confidence: data.confidence,
                    method: data.method,
                    generation_time: data.generation_time
                });
                
            } catch (error) {
                console.error('Error:', error);
                loadingDiv.remove();
                
                errorDiv.textContent = `Error: ${error.message}. Please try again.`;
                errorDiv.style.display = 'block';
                
                // Add error message to chat
                addMessage('I apologize, but I encountered an error. Please try again or contact support if the issue persists.', false);
            } finally {
                sendBtn.disabled = false;
                input.focus();
            }
        }
        
        // Initialize
        loadStats();
        document.getElementById('userInput').focus();
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


def start_healthcare_ai_service(port=8091):
    """Start the healthcare AI service"""
    server_address = ("", port)

    logger.info(f"Starting Healthcare AI Service on port {port}")
    logger.info("Initializing AI engine with LLM and knowledge base...")

    try:
        httpd = HTTPServer(server_address, HealthcareAIHandler)
        logger.info(f"Healthcare AI Service ready at http://localhost:{port}")
        logger.info(f"Chat interface: http://localhost:{port}/chat.html")
        logger.info("Press Ctrl+C to stop")

        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down Healthcare AI Service...")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise


if __name__ == "__main__":
    # Note: For full LLM support, install: pip install transformers torch
    # The service will work without it using the knowledge base
    start_healthcare_ai_service()
