# VoiceAI - Multi-Component Voice Assistant System

A comprehensive voice AI assistant system built with FastAPI backends, Next.js frontend, and various testing components for real-time voice interactions.

## 🏗️ Project Structure

```
VoiceAI/
├── agent_server/          # Main agent server with tools integration
├── server/               # Core voice AI backend server  
├── voice-client/         # Next.js frontend application
├── test_agent/          # Agent testing and development tools
└── tests/               # Various test scripts and experiments
```

## 🚀 Components Overview

### Agent Server
- **Location**: `agent_server/`
- **Purpose**: Main backend server with integrated tools (Tavily search, hotel booking)
- **Key Features**: 
  - Tool-based agent system using LangChain
  - Tavily API integration for web search
  - Hotel search functionality
  - PostgreSQL database integration

### Core Server  
- **Location**: `server/`
- **Purpose**: Core voice AI backend with memory and conversation management
- **Key Features**:
  - FastAPI REST API
  - Conversation memory system
  - Database integration with SQLAlchemy
  - CORS enabled for frontend integration

### Voice Client
- **Location**: `voice-client/`
- **Purpose**: Next.js frontend for voice interactions
- **Key Features**:
  - React 19 with TypeScript
  - Tailwind CSS styling
  - Real-time voice interface
  - Modern Next.js 16 architecture

### Test Agent
- **Location**: `test_agent/`
- **Purpose**: Development and testing tools for agent functionality
- **Key Features**:
  - Agent testing framework
  - Intent detection system
  - LLM integration testing
  - Search tool validation

### Tests Directory
- **Location**: `tests/`
- **Purpose**: Experimental scripts and various AI model tests
- **Key Features**:
  - Gemini API integration tests
  - Speech-to-text and text-to-speech testing
  - Real-time conversation experiments
  - Memory system validation

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 18+
- PostgreSQL database
- API Keys:
  - Google Gemini API key
  - Tavily API key (for search functionality)

### 1. Agent Server Setup
```bash
cd agent_server
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Core Server Setup  
```bash
cd server
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Voice Client Setup
```bash
cd voice-client
npm install
```

### 4. Test Agent Setup
```bash
cd test_agent
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 5. Tests Setup
```bash
cd tests
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 🔧 Configuration

### Environment Variables
Create `.env` files in each component directory:

**Agent Server & Server (.env)**:
```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://username:password@host:port/database
```

**Voice Client (.env)**:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Database Setup
The system uses PostgreSQL with the following connection string format:
```
postgresql://postgres:password@db.host.supabase.co:5432/postgres
```

## 🚀 Running the System

### Start the Backend Services
```bash
# Terminal 1 - Agent Server
cd agent_server
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8001

# Terminal 2 - Core Server  
cd server
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

### Start the Frontend
```bash
# Terminal 3 - Voice Client
cd voice-client
npm run dev
```

### Run Tests
```bash
# Test Agent
cd test_agent
.venv\Scripts\activate
python test_agent.py

# Various Tests
cd tests
.venv\Scripts\activate
python test_v3.py  # Main voice assistant test
```

## 📋 Key Dependencies

### Python Backend
- FastAPI - Web framework
- SQLAlchemy - Database ORM
- LangChain - Agent framework
- Google Generative AI - Gemini integration
- Tavily - Web search API
- Sentence Transformers - Embeddings
- Various audio libraries (sounddevice, librosa, etc.)

### Frontend
- Next.js 16 - React framework
- React 19 - UI library
- Tailwind CSS 4 - Styling
- TypeScript - Type safety

## 🎯 Usage

1. **Start all services** following the running instructions above
2. **Access the web interface** at `http://localhost:3000`
3. **Test agent functionality** using the test scripts in `test_agent/`
4. **Experiment with features** using various test files in `tests/`

## 🔍 API Endpoints

### Core Server
- `GET /health` - Health check
- `POST /chat` - Chat endpoint (expects `conversation_id` and `user_text`)

### Agent Server  
- `GET /` - Root endpoint
- Various tool endpoints for hotel search and other functionalities

## 🧪 Testing

The system includes comprehensive testing capabilities:
- **Intent Detection**: `test_agent/test_intent.py`
- **Agent Functionality**: `test_agent/test_agent.py`
- **Voice Processing**: Various files in `tests/`
- **Real-time Conversations**: `tests/gemini_test_2.py`

## 📝 Notes

- The system supports both text and voice interactions
- Memory management is implemented for conversation continuity
- Real-time audio processing capabilities are included
- Multiple AI model integrations (Gemini, embeddings)
- Tool-based agent architecture for extensible functionality