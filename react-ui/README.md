# AI Chat Companion

i want to make an chatbot interface,here are request,btw i hate icons and emojis:Core Functionality:

Chat Interface - Send/receive messages with the AI

Model Selection - Dropdown to choose between phi3:mini, gemma2:2b, llama3.2, tinyllama

API Configuration - Input field for API URL (default: http://localhost:8000)

Cache Toggle - Enable/disable response caching

Real-time Performance Stats - Show response time, optimizations applied, cache status

API Endpoints to Call:

POST /api/v1/chat - Main chat endpoint

GET /api/v1/health - Check if API is running

GET /api/v1/models - Get available models list

Required Features:

Message history display (user vs assistant messages)

Loading state while waiting for AI response

Error handling with user-friendly messages

Auto-scroll to latest message

Responsive design (mobile friendly)

Modern, clean UI

Data Format:

Chat Request:

json

{

"messages": [{"role": "user", "content": "user message"}],

"model": "phi3:mini",

"provider": "ollama", 

"use_cache": true

}

Chat Response:

json

{

"response": "AI response text",

"model": "phi3:mini",

"duration": 3.45,

"optimizations_applied": 68,

"cache_hit": false

}

Nice-to-Have Features:

Dark/light mode toggle

Export chat history

Clear chat button

Keyboard shortcuts (Enter to send, Shift+Enter for new line)

Typing indicator

Model performance comparison

Optimization details panel

Technical Requirements:

Pure HTML/CSS/JavaScript (no frameworks needed)

Must work with the provided API endpoints

Handle CORS properly

Mobile responsive

Cross-browser compatible

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/6fc7ea06-0473-425d-900b-9a74abf3cd75).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
