from flask import Blueprint, request, jsonify
import os
import requests

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    # Get the Gemini API key from environment variables
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    # The correct model identifier
    model = "models/gemini-1.5-flash"  # or "models/gemini-1.5-pro" depending on your needs

    # Correct API URL format for Google Gemini API
    api_url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={gemini_api_key}"
    
    # Prepare the request payload in the format expected by the Gemini API
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_message
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    }

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        # Send the request to the Gemini API
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Process the response
        response_data = response.json()
        
        # Extract the generated text from the response
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            if 'content' in response_data['candidates'][0]:
                text_parts = [part.get('text', '') for part in response_data['candidates'][0]['content']['parts']]
                bot_response = ''.join(text_parts)
                return jsonify({'response': bot_response})
        
        return jsonify({'error': 'Unexpected response format from Gemini API', 'raw_response': response_data}), 500
        
    except requests.exceptions.HTTPError as err:
        return jsonify({'error': f'HTTP error occurred: {err}', 'details': response.json() if response.text else None}), response.status_code
    except requests.exceptions.RequestException as err:
        return jsonify({'error': f'Request error occurred: {err}'}), 500
    except Exception as err:
        return jsonify({'error': f'Unexpected error: {str(err)}'}), 500