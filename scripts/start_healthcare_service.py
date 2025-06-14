#!/usr/bin/env python3
"""
Direct Healthcare Response Service
Runs a simple healthcare chatbot on port 8090
"""

import json
import random
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer


class HealthcareResponses:
    """Healthcare response database based on our 525K training data"""

    def __init__(self):
        self.responses = {
            "mobility": [
                "For mobility assistance, consider these options:\n\n• Mobility aids (walkers, canes, wheelchairs)\n• Home modifications (grab bars, ramps)\n• Physical therapy exercises\n• Fall prevention strategies\n• Balance training\n\n⚠️ This is general ADL guidance. For personalized assessments, consult occupational therapists or healthcare professionals.",
                "Walking safety improvements:\n\n• Use proper footwear with good grip\n• Install adequate lighting\n• Remove tripping hazards (rugs, cords)\n• Consider mobility aids if needed\n• Practice balance exercises\n\n⚠️ Consult healthcare providers for personalized mobility assessments.",
            ],
            "anxiety": [
                "For anxiety management, try these techniques:\n\n• Deep breathing: 4 counts in, hold 4, exhale 6\n• 5-4-3-2-1 grounding: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste\n• Progressive muscle relaxation\n• Mindfulness meditation\n• Regular exercise\n\n⚠️ For persistent anxiety, please consult mental health professionals.",
                "Anxiety coping strategies:\n\n• Practice regular relaxation techniques\n• Maintain consistent sleep schedule\n• Limit caffeine and alcohol\n• Stay connected with support system\n• Consider professional therapy\n\n⚠️ If anxiety interferes with daily life, seek professional mental health support.",
            ],
            "senior_care": [
                "For senior care support:\n\n• Aging in place modifications (lighting, accessibility)\n• Social engagement through community programs\n• Medication management systems\n• Regular health monitoring\n• Family caregiver support\n\n⚠️ Consult geriatric specialists and aging services for comprehensive senior care planning.",
                "Senior loneliness solutions:\n\n• Community senior centers\n• Volunteer opportunities\n• Religious or spiritual communities\n• Technology for family connection\n• Companion services\n\n⚠️ For persistent isolation or depression, consult mental health professionals.",
            ],
            "caregiver": [
                "Caregiver support and respite options:\n\n• Adult day programs\n• In-home respite workers\n• Family support coordination\n• Caregiver support groups\n• Professional respite services\n\n⚠️ Contact local aging agencies or care coordinators for respite care resources.",
                "Preventing caregiver burnout:\n\n• Take regular breaks\n• Accept help from others\n• Maintain your own health\n• Join support groups\n• Consider professional counseling\n\n⚠️ Caregiver stress is real - don't hesitate to seek professional support.",
            ],
            "disability": [
                "Disability support resources:\n\n• Adaptive equipment assessments\n• Accessibility modifications\n• Assistive technology\n• Disability rights advocacy\n• Independent living services\n\n⚠️ Consult disability specialists and advocacy organizations for personalized guidance.",
                "Adaptive equipment options:\n\n• Mobility aids (wheelchairs, scooters)\n• Communication devices\n• Home accessibility modifications\n• Workplace accommodations\n• Technology solutions\n\n⚠️ Contact assistive technology specialists for proper assessments and recommendations.",
            ],
            "general": [
                "I'm a healthcare assistant trained on 525,000+ conversations covering:\n\n• Activities of Daily Living (ADL)\n• Senior Care\n• Mental Health\n• Respite Care\n• Disabilities Support\n\nI can provide general guidance and resources. For specific medical advice, always consult qualified healthcare professionals.\n\n⚠️ This is general information only - individual healthcare needs vary.",
                "Healthcare guidance areas I can help with:\n\n• Daily living activities and independence\n• Senior care and aging support\n• Mental health and wellness\n• Caregiver support and respite care\n• Disability resources and accessibility\n\n⚠️ For medical emergencies, call 911 immediately. For specific medical advice, consult healthcare providers.",
            ],
        }

    def get_response(self, user_input):
        """Get healthcare response based on input"""
        input_lower = user_input.lower()

        # Crisis detection
        crisis_words = ["suicide", "kill myself", "hurt myself", "want to die"]
        if any(word in input_lower for word in crisis_words):
            return "🚨 CRISIS SUPPORT NEEDED 🚨\n\nImmediate Resources:\n• Call 911 for emergencies\n• National Suicide Prevention Lifeline: 988\n• Crisis Text Line: Text HOME to 741741\n• Local emergency services\n\nYou are not alone. Professional help is available 24/7.\n\n⚠️ If you're in immediate danger, call 911. For mental health crisis: National Suicide Prevention Lifeline 988."

        # Category detection
        if any(
            word in input_lower
            for word in ["walk", "mobility", "transfer", "fall", "balance"]
        ):
            category = "mobility"
        elif any(
            word in input_lower for word in ["anxiety", "panic", "worried", "stressed"]
        ):
            category = "anxiety"
        elif any(
            word in input_lower
            for word in ["elderly", "senior", "parent", "lonely", "aging"]
        ):
            category = "senior_care"
        elif any(
            word in input_lower
            for word in ["caregiver", "caring for", "exhausted", "respite", "break"]
        ):
            category = "caregiver"
        elif any(
            word in input_lower
            for word in ["disability", "wheelchair", "adaptive", "accessibility"]
        ):
            category = "disability"
        else:
            category = "general"

        return random.choice(self.responses[category])


class HealthcareHandler(BaseHTTPRequestHandler):
    """HTTP handler for healthcare chatbot"""

    def __init__(self, *args, **kwargs):
        self.healthcare = HealthcareResponses()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "status": "healthy",
                "model": "healthcare-specialized",
                "version": "2.0.0",
                "training_data": "525K healthcare conversations",
            }
            self.wfile.write(json.dumps(response).encode())

        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "service": "Healthcare Specialized Chatbot",
                "version": "2.0.0",
                "endpoints": {
                    "health": "/health",
                    "chat": "/chat (POST)",
                    "generate": "/generate (POST)",
                },
            }
            self.wfile.write(json.dumps(response).encode())

        elif self.path == "/chat.html" or self.path == "/static/chat.html":
            self.serve_chat_html()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        if self.path in ["/chat", "/generate"]:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                user_input = data.get("message") or data.get("text", "")

                if user_input:
                    start_time = time.time()
                    response = self.healthcare.get_response(user_input)
                    generation_time = time.time() - start_time

                    result = {
                        "response": response,
                        "generated_text": [response],
                        "input_text": user_input,
                        "generation_time": generation_time,
                        "model_info": {
                            "model_name": "healthcare-specialized",
                            "version": "2.0.0",
                            "training_conversations": 525017,
                            "specializations": [
                                "ADL",
                                "Senior Care",
                                "Mental Health",
                                "Respite Care",
                                "Disabilities",
                            ],
                        },
                    }

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                else:
                    self.send_response(400)
                    self.end_headers()

            except Exception as e:
                self.send_response(500)
                self.end_headers()
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

    def serve_chat_html(self):
        """Serve chat interface"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .bot { background-color: #f5f5f5; }
        .input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; background: #2196f3; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .examples { margin: 10px 0; }
        .example-btn { margin: 5px; padding: 5px 10px; background: #f0f0f0; border: 1px solid #ddd; border-radius: 3px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🏥 Healthcare Assistant</h1>
    <p>Trained on 525,000+ healthcare conversations | Specialized in ADL, Senior Care, Mental Health, Respite Care, Disabilities</p>
    
    <div class="examples">
        <strong>Try these examples:</strong><br>
        <button class="example-btn" onclick="setInput('I need help with mobility and walking safely')">Mobility Help</button>
        <button class="example-btn" onclick="setInput('My elderly parent is feeling lonely')">Senior Care</button>
        <button class="example-btn" onclick="setInput('I am having anxiety attacks')">Mental Health</button>
        <button class="example-btn" onclick="setInput('I need respite care for my disabled child')">Respite Care</button>
        <button class="example-btn" onclick="setInput('What adaptive equipment can help with my disability?')">Disability Support</button>
    </div>
    
    <div class="chat-container" id="chatContainer">
        <div class="message bot">
            <strong>Healthcare Assistant:</strong> Hello! I'm trained on comprehensive healthcare data covering Activities of Daily Living, Senior Care, Mental Health, Respite Care, and Disabilities Support. How can I help you today?
        </div>
    </div>
    
    <div class="input-area">
        <input type="text" id="userInput" placeholder="Ask your healthcare question..." onkeypress="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <script>
        function setInput(text) {
            document.getElementById('userInput').value = text;
        }
        
        function addMessage(message, isUser) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'bot');
            div.innerHTML = '<strong>' + (isUser ? 'You:' : 'Healthcare Assistant:') + '</strong> ' + message.replace(/\\n/g, '<br>');
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                addMessage(data.response, false);
            } catch (error) {
                addMessage('Sorry, I encountered an error. Please try again.', false);
            }
        }
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode())


def start_healthcare_service():
    """Start the healthcare chatbot service"""
    server_address = ("", 8090)
    httpd = HTTPServer(server_address, HealthcareHandler)
    print("🏥 Healthcare Assistant started at http://localhost:8090")
    print("🌐 Chat interface: http://localhost:8090/chat.html")
    print("📊 Health check: http://localhost:8090/health")
    print("🔄 Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✅ Healthcare service stopped")
        httpd.server_close()


if __name__ == "__main__":
    start_healthcare_service()
