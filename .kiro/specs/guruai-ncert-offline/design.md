# Design Document

## Overview

GuruAI is an offline-first AI learning companion that provides JEE and NEET students with an integrated platform for studying NCERT content, practicing with previous year questions, and receiving personalized guidance. The system architecture prioritizes local execution, efficient resource management, and seamless user experience while maintaining complete independence from internet connectivity and external APIs.

The design follows a three-tier architecture: a web-based frontend (HTML/CSS/JavaScript), a Flask backend handling business logic and AI orchestration, and a data layer comprising ChromaDB vector store, local AI models, and structured content databases.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│                  (HTML/CSS/JavaScript)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Chat         │  │  Progress    │  │   Search     │      │
│  │ Interface    │  │  Dashboard   │  │   Interface  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│                    (Flask Backend)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Query      │  │    Image     │  │   Question   │      │
│  │  Handler     │  │  Processor   │  │  Generator   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   User       │  │   Progress   │  │   Model      │      │
│  │  Manager     │  │   Tracker    │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              ChromaDB Vector Store                   │   │
│  │  - NCERT Content Embeddings                          │   │
│  │  - Diagram Embeddings                                │   │
│  │  - Previous Papers Index                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   SQLite     │  │   File       │  │   Local AI   │      │
│  │  (User Data) │  │   Storage    │  │   Models     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- HTML5 for structure
- CSS3 for styling (modern, responsive design)
- Vanilla JavaScript (ES6+) for interactivity
- No external frameworks to minimize bundle size

**Backend:**
- Python 3.10+
- Flask 3.0+ (lightweight web framework)
- Flask-CORS for cross-origin requests
- Flask-Login for session management

**AI & ML:**
- Llama.cpp for local LLM inference (Llama 3.2 3B or Phi-3 Mini)
- sentence-transformers for embeddings (all-MiniLM-L6-v2)
- Tesseract + EasyOCR for optical character recognition
- OpenCV + Pillow for image processing
- SpeechRecognition + pyttsx3 for voice input/output
- Google Cloud Vision API (offline mode) for handwriting recognition

**Data Storage:**
- ChromaDB for vector storage and semantic search
- SQLite for user data, progress tracking, and session management
- Local file system for NCERT diagrams and generated content

**Content Processing:**
- PyPDF2 + pdfplumber for PDF extraction
- NLTK for text preprocessing
- NumPy for numerical operations
- Matplotlib for graph generation

## Components and Interfaces

### 1. Frontend Components

#### 1.1 Chat Interface Component
**Responsibility:** Handle user queries and display AI responses

**Key Elements:**
- Message input field with auto-resize
- Send button and image upload button
- Chat history display with user/bot message distinction
- Loading indicator during AI processing
- Quiz display component for adaptive questions

**JavaScript Functions:**
```javascript
// Core functions
async function sendQuery(queryText)
async function uploadImage(imageFile)
function displayMessage(message, sender, metadata)
function displayQuiz(quizData)
function handleQuizAnswer(selectedOption, correctAnswer)
```

#### 1.2 Progress Dashboard Component
**Responsibility:** Display user's learning progress and statistics

**Key Elements:**
- Subject-wise accuracy charts
- Chapter completion status
- Recent activity timeline
- Weak areas identification
- Recommended topics

**JavaScript Functions:**
```javascript
async function loadProgressData()
function renderSubjectChart(subjectData)
function displayWeakAreas(topics)
function generateRecommendations(progressData)
```

#### 1.3 Search Interface Component
**Responsibility:** Enable content search across NCERT materials

**Key Elements:**
- Search input with autocomplete
- Filter options (subject, class, chapter)
- Results list with relevance scores
- Preview pane for selected result
- Diagram thumbnails in results

**JavaScript Functions:**
```javascript
async function searchContent(query, filters)
function displaySearchResults(results)
function previewResult(resultId)
```

#### 1.4 Question Paper Generator Component
**Responsibility:** Create custom practice tests

**Key Elements:**
- Chapter/topic selection checkboxes
- Difficulty distribution sliders
- Question count input
- Generate button
- Download/print options

**JavaScript Functions:**
```javascript
async function generateQuestionPaper(config)
function displayQuestionPaper(paper)
function submitAnswers(answers)
function showSolutions(paper, userAnswers)
```

### 2. Backend Components

#### 2.1 Query Handler
**Responsibility:** Process user queries and orchestrate AI response generation

**Python Class:**
```python
class QueryHandler:
    def __init__(self, rag_system, llm_manager, diagram_retriever):
        self.rag = rag_system
        self.llm = llm_manager
        self.diagrams = diagram_retriever
    
    async def process_query(self, query: str, user_id: str) -> dict:
        # 1. Retrieve relevant NCERT context
        context = await self.rag.retrieve(query, top_k=5)
        
        # 2. Find relevant diagrams
        diagrams = await self.diagrams.find_relevant(query, context)
        
        # 3. Generate response using LLM
        response = await self.llm.generate(query, context)
        
        # 4. Generate adaptive quiz
        quiz = await self.generate_quiz(query, response)
        
        return {
            "explanation": response,
            "diagrams": diagrams,
            "quiz": quiz,
            "references": self._extract_references(context)
        }
```

#### 2.2 Image Processor
**Responsibility:** Handle image uploads and extract information

**Python Class:**
```python
class ImageProcessor:
    def __init__(self, ocr_engine, diagram_matcher):
        self.ocr = ocr_engine
        self.matcher = diagram_matcher
    
    async def process_image(self, image_data: bytes) -> dict:
        # 1. Preprocess image
        processed_img = self._preprocess(image_data)
        
        # 2. Extract text using OCR
        extracted_text = await self.ocr.extract(processed_img)
        
        # 3. Match against NCERT diagrams
        matched_diagram = await self.matcher.find_match(processed_img)
        
        # 4. Determine if it's a problem or diagram
        content_type = self._classify_content(extracted_text, matched_diagram)
        
        return {
            "type": content_type,
            "text": extracted_text,
            "matched_diagram": matched_diagram
        }
```

#### 2.3 Question Generator
**Responsibility:** Generate practice questions based on patterns and content

**Python Class:**
```python
class QuestionGenerator:
    def __init__(self, previous_papers_db, llm_manager):
        self.papers_db = previous_papers_db
        self.llm = llm_manager
    
    async def generate_paper(self, config: dict) -> dict:
        # 1. Retrieve questions matching criteria
        candidate_questions = self.papers_db.query(
            topics=config['topics'],
            difficulty=config['difficulty_distribution']
        )
        
        # 2. Select questions ensuring no duplicates
        selected = self._select_questions(
            candidate_questions,
            count=config['question_count']
        )
        
        # 3. Generate additional questions if needed
        if len(selected) < config['question_count']:
            generated = await self._generate_new_questions(
                config['topics'],
                config['question_count'] - len(selected)
            )
            selected.extend(generated)
        
        # 4. Create answer key with solutions
        answer_key = self._create_answer_key(selected)
        
        return {
            "questions": selected,
            "answer_key": answer_key,
            "metadata": config
        }
```

#### 2.4 Model Manager
**Responsibility:** Manage AI model lifecycle and resource usage

**Python Class:**
```python
class ModelManager:
    def __init__(self, model_path: str, config: dict):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.last_used = None
        self.idle_timeout = config.get('idle_timeout', 600)  # 10 minutes
    
    async def load_model(self):
        if self.model is None:
            self.model = await self._load_llm(self.model_path)
        self.last_used = time.time()
    
    async def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
    
    async def generate(self, prompt: str, context: str) -> str:
        await self.load_model()
        self.last_used = time.time()
        return await self.model.generate(prompt, context)
    
    async def check_idle(self):
        if self.model and (time.time() - self.last_used) > self.idle_timeout:
            await self.unload_model()
```

#### 2.5 RAG System
**Responsibility:** Retrieve relevant context from NCERT content

**Python Class:**
```python
class RAGSystem:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedder = embedding_model
    
    async def retrieve(self, query: str, top_k: int = 5) -> list:
        # 1. Embed the query
        query_embedding = await self.embedder.encode(query)
        
        # 2. Search vector store
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 3. Rerank results by relevance
        reranked = self._rerank_results(results, query)
        
        return reranked
```

#### 2.6 Voice Handler
**Responsibility:** Handle voice input and output for natural interaction

**Python Class:**
```python
class VoiceHandler:
    def __init__(self, speech_recognizer, tts_engine):
        self.recognizer = speech_recognizer
        self.tts = tts_engine
        self.supported_languages = ['en', 'hi', 'hinglish']
    
    async def transcribe_audio(self, audio_data: bytes, language: str = 'hinglish') -> str:
        # 1. Process audio data
        processed_audio = self._preprocess_audio(audio_data)
        
        # 2. Transcribe using local speech recognition
        text = await self.recognizer.transcribe(processed_audio, language=language)
        
        # 3. Post-process transcription
        cleaned_text = self._clean_transcription(text)
        
        return cleaned_text
    
    async def synthesize_speech(self, text: str, language: str = 'en', speed: float = 1.0) -> bytes:
        # 1. Prepare text for TTS
        prepared_text = self._prepare_for_tts(text, language)
        
        # 2. Generate speech audio
        audio = await self.tts.synthesize(prepared_text, language=language, speed=speed)
        
        return audio
```

#### 2.7 Handwriting Recognizer
**Responsibility:** Process handwritten content from images

**Python Class:**
```python
class HandwritingRecognizer:
    def __init__(self, hw_model, math_recognizer):
        self.hw_model = hw_model
        self.math_recognizer = math_recognizer
    
    async def recognize_handwriting(self, image_data: bytes) -> dict:
        # 1. Preprocess image
        processed_img = self._preprocess_handwriting(image_data)
        
        # 2. Detect text regions
        text_regions = self._detect_text_regions(processed_img)
        
        # 3. Recognize text in each region
        recognized_text = []
        for region in text_regions:
            if self._is_mathematical(region):
                text = await self.math_recognizer.recognize(region)
            else:
                text = await self.hw_model.recognize(region)
            recognized_text.append(text)
        
        # 4. Identify uncertain characters
        uncertain = self._identify_uncertain_chars(recognized_text)
        
        return {
            "text": " ".join(recognized_text),
            "uncertain_chars": uncertain,
            "confidence": self._calculate_confidence(recognized_text)
        }
```

#### 2.8 Deep Explanation Generator
**Responsibility:** Generate comprehensive, multi-layered explanations

**Python Class:**
```python
class DeepExplanationGenerator:
    def __init__(self, rag_system, llm_manager):
        self.rag = rag_system
        self.llm = llm_manager
    
    async def generate_deep_explanation(self, concept: str, user_level: str = 'intermediate') -> dict:
        # 1. Identify prerequisites
        prerequisites = await self._identify_prerequisites(concept)
        
        # 2. Retrieve comprehensive context
        context = await self.rag.retrieve(concept, top_k=10)
        
        # 3. Find multiple examples
        examples = await self._find_examples(concept, context)
        
        # 4. Generate analogies
        analogies = await self._generate_analogies(concept, context)
        
        # 5. Create layered explanation
        explanation = {
            "prerequisites": prerequisites,
            "basic": await self._generate_basic_explanation(concept, context),
            "intermediate": await self._generate_intermediate_explanation(concept, context, examples),
            "advanced": await self._generate_advanced_explanation(concept, context),
            "examples": examples,
            "analogies": analogies,
            "real_world_applications": await self._find_applications(concept)
        }
        
        return explanation
```

#### 2.9 Peer Comparison Manager
**Responsibility:** Manage anonymous peer performance comparisons

**Python Class:**
```python
class PeerComparisonManager:
    def __init__(self, stats_db):
        self.stats_db = stats_db
    
    async def get_peer_comparison(self, user_id: str) -> dict:
        # 1. Get user's anonymous ID
        anon_id = self._get_anonymous_id(user_id)
        
        # 2. Fetch user statistics
        user_stats = await self._get_user_stats(user_id)
        
        # 3. Calculate percentiles
        percentiles = await self._calculate_percentiles(user_stats)
        
        # 4. Get aggregated peer statistics
        peer_stats = await self._get_aggregated_peer_stats()
        
        return {
            "anonymous_id": anon_id,
            "percentile_overall": percentiles['overall'],
            "percentile_by_subject": percentiles['by_subject'],
            "average_accuracy": user_stats['accuracy'],
            "peer_average_accuracy": peer_stats['average_accuracy'],
            "topics_mastered": user_stats['topics_mastered'],
            "peer_average_topics": peer_stats['average_topics_mastered'],
            "questions_attempted": user_stats['questions_attempted'],
            "peer_average_questions": peer_stats['average_questions_attempted']
        }
```

### 3. API Endpoints

#### 3.1 Query Endpoints

**POST /api/ask**
- Request: `{"query": "string", "user_id": "string"}`
- Response: `{"explanation": "string", "diagrams": [], "quiz": {}, "references": []}`
- Purpose: Handle text-based queries

**POST /api/solve-image**
- Request: `multipart/form-data` with image file
- Response: `{"solution": "string", "steps": [], "diagrams": []}`
- Purpose: Process uploaded images

#### 3.2 Question Paper Endpoints

**POST /api/generate-paper**
- Request: `{"topics": [], "difficulty": {}, "count": int}`
- Response: `{"questions": [], "answer_key": {}, "paper_id": "string"}`
- Purpose: Generate custom question papers

**GET /api/previous-papers**
- Query params: `?exam=JEE&year=2023`
- Response: `{"papers": [], "metadata": {}}`
- Purpose: Retrieve previous year papers

#### 3.3 Progress Endpoints

**GET /api/progress/{user_id}**
- Response: `{"subjects": {}, "chapters": {}, "accuracy": {}, "recommendations": []}`
- Purpose: Fetch user progress data

**POST /api/progress/record**
- Request: `{"user_id": "string", "question_id": "string", "correct": bool}`
- Response: `{"status": "success"}`
- Purpose: Record question attempt

#### 3.4 Search Endpoints

**GET /api/search**
- Query params: `?q=query&subject=Physics&class=12`
- Response: `{"results": [], "total": int}`
- Purpose: Search NCERT content

#### 3.5 User Management Endpoints

**POST /api/auth/register**
- Request: `{"username": "string", "password": "string"}`
- Response: `{"user_id": "string", "token": "string"}`
- Purpose: Create new user account

**POST /api/auth/login**
- Request: `{"username": "string", "password": "string"}`
- Response: `{"user_id": "string", "token": "string"}`
- Purpose: Authenticate user

**POST /api/auth/logout**
- Request: `{"user_id": "string"}`
- Response: `{"status": "success"}`
- Purpose: End user session

#### 3.6 Voice Interaction Endpoints

**POST /api/voice/transcribe**
- Request: `multipart/form-data` with audio file
- Response: `{"text": "string", "language": "string"}`
- Purpose: Transcribe voice input to text

**POST /api/voice/synthesize**
- Request: `{"text": "string", "language": "string", "speed": float}`
- Response: Audio file (binary)
- Purpose: Convert text to speech

#### 3.7 Handwriting Recognition Endpoints

**POST /api/handwriting/recognize**
- Request: `multipart/form-data` with image file
- Response: `{"text": "string", "uncertain_chars": [], "confidence": float}`
- Purpose: Recognize handwritten content

#### 3.8 Deep Explanation Endpoints

**POST /api/explain/deep**
- Request: `{"concept": "string", "user_level": "string"}`
- Response: `{"prerequisites": [], "basic": "string", "intermediate": "string", "advanced": "string", "examples": [], "analogies": []}`
- Purpose: Generate comprehensive deep explanation

#### 3.9 Peer Comparison Endpoints

**GET /api/peer-comparison/{user_id}**
- Response: `{"anonymous_id": "string", "percentile_overall": float, "percentile_by_subject": {}, "peer_stats": {}}`
- Purpose: Get anonymous peer performance comparison

**POST /api/peer-comparison/opt-in**
- Request: `{"user_id": "string", "opt_in": bool}`
- Response: `{"status": "success"}`
- Purpose: Opt in or out of peer comparison

## Data Models

### 1. User Model
```python
class User:
    user_id: str  # UUID
    username: str
    password_hash: str  # bcrypt hashed
    created_at: datetime
    last_login: datetime
    preferences: dict  # JSON field
```

### 2. Progress Model
```python
class Progress:
    progress_id: str
    user_id: str  # Foreign key
    subject: str  # Physics, Chemistry, Maths, Biology
    chapter: str
    topic: str
    questions_attempted: int
    questions_correct: int
    accuracy: float
    last_studied: datetime
```

### 3. Question Model
```python
class Question:
    question_id: str
    source: str  # "JEE Main 2023", "Generated", etc.
    subject: str
    chapter: str
    topic: str
    difficulty: str  # easy, medium, hard
    question_text: str
    options: list  # For MCQs
    correct_answer: str
    solution: str
    ncert_reference: str  # Chapter and page
```

### 4. NCERT Content Model
```python
class NCERTContent:
    content_id: str
    subject: str
    class_level: int  # 11 or 12
    chapter: int
    section: str
    page_number: int
    content_text: str
    embedding: list  # Vector representation
    diagrams: list  # Associated diagram IDs
```

### 5. Diagram Model
```python
class Diagram:
    diagram_id: str
    subject: str
    chapter: int
    figure_number: str
    caption: str
    file_path: str  # Local path to image
    embedding: list  # Visual embedding
    labels: list  # Labeled parts
```

### 6. Session Model
```python
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool
```

### 7. Voice Interaction Model
```python
class VoiceInteraction:
    interaction_id: str
    user_id: str
    audio_duration: float  # seconds
    transcribed_text: str
    language: str  # 'en', 'hi', 'hinglish'
    confidence: float
    timestamp: datetime
```

### 8. Handwriting Recognition Model
```python
class HandwritingRecognition:
    recognition_id: str
    user_id: str
    image_path: str
    recognized_text: str
    uncertain_characters: list
    confidence: float
    timestamp: datetime
```

### 9. Peer Statistics Model
```python
class PeerStatistics:
    stat_id: str
    anonymous_user_id: str  # Anonymized identifier
    subject: str
    accuracy: float
    topics_mastered: int
    questions_attempted: int
    last_updated: datetime
    opt_in_status: bool
```

## Error Handling

### Error Categories

**1. User Input Errors (400 series)**
- Invalid query format
- Missing required fields
- Image file too large or wrong format

**2. Authentication Errors (401, 403)**
- Invalid credentials
- Session expired
- Unauthorized access

**3. Resource Errors (404)**
- Content not found
- User not found
- Question paper not found

**4. Server Errors (500 series)**
- Model loading failure
- Database connection error
- OCR processing failure

### Error Response Format
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable message",
        "details": "Additional context",
        "timestamp": "ISO 8601 timestamp"
    }
}
```

### Error Handling Strategy

**Frontend:**
- Display user-friendly error messages
- Provide retry options for transient failures
- Log errors to browser console for debugging
- Graceful degradation when features unavailable

**Backend:**
- Comprehensive try-catch blocks
- Logging with different severity levels
- Automatic retry for transient failures
- Fallback mechanisms for AI model failures

## Testing Strategy

### Unit Testing

**Backend Unit Tests:**
- Test each component in isolation
- Mock external dependencies (database, AI models)
- Focus on business logic correctness
- Use pytest framework

**Example Test Cases:**
- `test_query_handler_retrieves_context()`
- `test_image_processor_extracts_text()`
- `test_question_generator_creates_valid_paper()`
- `test_model_manager_loads_and_unloads()`
- `test_rag_system_returns_relevant_results()`

**Frontend Unit Tests:**
- Test JavaScript functions independently
- Mock API responses
- Test DOM manipulation
- Use Jest or Mocha framework

**Example Test Cases:**
- `test_send_query_calls_api()`
- `test_display_message_renders_correctly()`
- `test_quiz_answer_validation()`

### Integration Testing

**API Integration Tests:**
- Test complete request-response cycles
- Verify data flow between components
- Test with real database (test instance)
- Use Flask test client

**Example Test Cases:**
- `test_ask_endpoint_returns_complete_response()`
- `test_image_upload_processes_correctly()`
- `test_question_paper_generation_end_to_end()`
- `test_user_authentication_flow()`

### Property-Based Testing

GuruAI will use property-based testing to verify correctness properties across many inputs. We'll use the **Hypothesis** library for Python.

**Configuration:**
- Minimum 100 iterations per property test
- Each property test tagged with design document reference


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Context Retrieval Completeness
*For any* valid user query about Physics, Chemistry, Mathematics, or Biology concepts, the RAG system should retrieve at least one relevant passage from the NCERT vector store.
**Validates: Requirements 1.1**

### Property 2: Response Grounding in NCERT Content
*For any* generated response, all factual claims should be traceable to the retrieved NCERT context passages.
**Validates: Requirements 1.2**

### Property 3: Multi-Chapter Reference Inclusion
*For any* query that retrieves context from multiple NCERT chapters, the response should include explicit chapter references for each source.
**Validates: Requirements 1.4**

### Property 4: OCR Text Extraction Success
*For any* uploaded image containing text from NCERT textbooks, the OCR system should extract the text with reasonable accuracy (>80% character accuracy).
**Validates: Requirements 2.1**

### Property 5: Diagram Matching Accuracy
*For any* uploaded NCERT diagram image, the system should correctly identify the matching diagram from the stored collection or indicate no match found.
**Validates: Requirements 2.2**

### Property 6: Diagram-Explanation Association
*For any* matched diagram, the system should retrieve and return the associated NCERT explanation from the correct chapter and section.
**Validates: Requirements 2.3**

### Property 7: Question Pattern Alignment
*For any* generated practice question, the question structure and difficulty should match patterns found in the previous 20 years of JEE/NEET papers.
**Validates: Requirements 3.1**

### Property 8: Question-NCERT Alignment
*For any* generated practice question, the concepts tested should be covered in the NCERT textbooks for the specified subject and class.
**Validates: Requirements 3.2**

### Property 9: Solution NCERT Referencing
*For any* practice question solution, the solution should include at least one explicit reference to NCERT chapter, formula, or concept.
**Validates: Requirements 3.3**

### Property 10: Practice Test Structure Conformance
*For any* generated full-length practice test, the number of questions, section distribution, and difficulty distribution should match the structure of actual JEE/NEET exams.
**Validates: Requirements 3.4**

### Property 11: Previous Year Question Metadata
*For any* displayed previous year question, the response should include the year and exam name (JEE Main/Advanced/NEET) in the metadata.
**Validates: Requirements 3.5**

### Property 12: Offline Operation Completeness
*For any* user interaction (query, image upload, question generation), the system should complete the operation without making external network calls.
**Validates: Requirements 4.2, 4.3, 4.4, 4.5**

### Property 13: Diagram Display for Relevant Concepts
*For any* concept explanation where associated NCERT diagrams exist, the response should include all relevant diagrams from local storage.
**Validates: Requirements 5.1, 5.2, 5.3**

### Property 14: Diagram Metadata Completeness
*For any* displayed diagram, the response should include the figure caption and page reference from the NCERT textbook.
**Validates: Requirements 5.4**

### Property 15: Labeled Diagram Explanation
*For any* diagram containing labeled parts, the explanation should describe each labeled component.
**Validates: Requirements 5.5**

### Property 16: Solution Step Structure
*For any* numerical problem solution, the response should contain at least two sequential steps with clear progression.
**Validates: Requirements 6.1**

### Property 17: Formula Reference in Solutions
*For any* problem solution, each step that uses a formula should include the NCERT formula reference.
**Validates: Requirements 6.2**

### Property 18: Mathematical Expression Formatting
*For any* response containing mathematical expressions, the equations should be formatted using proper notation (LaTeX or MathML).
**Validates: Requirements 6.5**

### Property 19: Question Paper Topic Coverage
*For any* generated custom question paper with specified topics, all questions should belong to the selected topics only.
**Validates: Requirements 7.1**

### Property 20: Question Paper Difficulty Distribution
*For any* generated custom question paper with specified difficulty distribution, the actual distribution should match the requested percentages within ±5%.
**Validates: Requirements 7.2**

### Property 21: Question Paper Count Accuracy
*For any* question paper generation request with specified count N, the generated paper should contain exactly N questions.
**Validates: Requirements 7.3**

### Property 22: Question Paper Uniqueness
*For any* generated question paper, no two questions should be identical (no duplicates).
**Validates: Requirements 7.4**

### Property 23: Answer Key Completeness
*For any* generated question paper, the answer key should contain solutions for all questions in the paper.
**Validates: Requirements 7.5**

### Property 24: Progress Record Completeness
*For any* completed practice question, the progress record should include topic, difficulty, and correctness fields.
**Validates: Requirements 8.1**

### Property 25: Progress Statistics Coverage
*For any* user's progress view, statistics should be displayed for all subjects (Physics, Chemistry, Mathematics, Biology) and all chapters within each subject.
**Validates: Requirements 8.2**

### Property 26: Accuracy Calculation Correctness
*For any* topic with attempted questions, the displayed accuracy percentage should equal (correct_answers / total_attempts) × 100.
**Validates: Requirements 8.3**

### Property 27: Recommendation Prioritization
*For any* user's recommendation list, topics with lower accuracy should be ranked higher than topics with higher accuracy.
**Validates: Requirements 8.4**

### Property 28: Progress Data Persistence
*For any* user session, progress data recorded during the session should persist after logout and be available in the next login.
**Validates: Requirements 8.5**

### Property 29: Search Result Metadata Completeness
*For any* search result, the result should include subject, class, chapter, and page number fields.
**Validates: Requirements 9.2**

### Property 30: Search Result Context Display
*For any* selected search result, the displayed content should include the matching passage plus surrounding context paragraphs.
**Validates: Requirements 9.3**

### Property 31: Diagram Search Inclusion
*For any* search query matching diagram captions, the search results should include the corresponding diagrams.
**Validates: Requirements 9.4**

### Property 32: Quiz Generation Count
*For any* completed explanation, the generated quiz should contain between 2 and 4 multiple-choice questions.
**Validates: Requirements 10.1**

### Property 33: Quiz Feedback Provision
*For any* answered quiz question, the system should provide immediate feedback indicating whether the answer is correct or incorrect.
**Validates: Requirements 10.3**

### Property 34: Incorrect Answer Explanation
*For any* incorrectly answered quiz question, the feedback should include an explanation of why the answer is wrong and the correct reasoning.
**Validates: Requirements 10.4**

### Property 35: Model Lazy Loading
*For any* application startup, the LLM should not be loaded into memory until the first query is submitted.
**Validates: Requirements 11.1**

### Property 36: Model Idle Unloading
*For any* idle period exceeding the configured timeout, the LLM should be unloaded from memory.
**Validates: Requirements 11.2**

### Property 37: Model Reload with Indicator
*For any* query submitted after model unloading, the system should reload the model and display a loading indicator to the user.
**Validates: Requirements 11.3**

### Property 38: Model Status Display
*For any* model load or unload operation, the system should display the current status to the user.
**Validates: Requirements 11.4**

### Property 39: Settings Configurability
*For any* user-modified system setting (memory limits, idle timeout), the new value should be applied and persist across sessions.
**Validates: Requirements 11.5**

### Property 40: Password Encryption
*For any* created user account, the stored password should be encrypted (not plaintext) using bcrypt or equivalent.
**Validates: Requirements 12.2**

### Property 41: User Data Loading
*For any* successful login, the system should load the correct user's progress data, bookmarks, and preferences.
**Validates: Requirements 12.3**

### Property 42: User Data Isolation
*For any* two different users, their progress data should be completely isolated with no cross-contamination.
**Validates: Requirements 12.4**

### Property 43: Session Clearing on Logout
*For any* user logout, the session should be cleared and subsequent requests should require re-authentication.
**Validates: Requirements 12.5**

### Property 44: Voice Transcription Accuracy
*For any* voice input in English or Hinglish, the transcription should achieve at least 85% word accuracy.
**Validates: Requirements 14.2, 14.3**

### Property 45: Voice Output Completeness
*For any* text response converted to speech, the audio output should contain all words from the original text.
**Validates: Requirements 15.1, 15.2**

### Property 46: Handwriting Recognition Success
*For any* uploaded image containing handwritten text, the system should extract at least 80% of the characters correctly.
**Validates: Requirements 16.1, 16.2**

### Property 47: Mathematical Expression Recognition
*For any* handwritten mathematical expression, the system should convert it to a valid digital format that can be processed.
**Validates: Requirements 16.3**

### Property 48: Deep Explanation Layering
*For any* deep explanation request, the response should contain at least three distinct layers (basic, intermediate, advanced) based on NCERT content.
**Validates: Requirements 17.1, 17.5**

### Property 49: Example Multiplicity in Deep Explanations
*For any* deep explanation, the response should include at least two different examples from NCERT textbooks.
**Validates: Requirements 17.2**

### Property 50: Prerequisite Identification
*For any* complex concept requiring prerequisites, the deep explanation should identify and list all prerequisite concepts before the main explanation.
**Validates: Requirements 17.4**

### Property 51: Peer Comparison Anonymity
*For any* peer comparison data, no personally identifiable information should be visible or derivable from the displayed statistics.
**Validates: Requirements 18.1, 18.4**

### Property 52: Percentile Calculation Accuracy
*For any* user's peer comparison, the percentile ranking should accurately reflect their position relative to all opted-in users.
**Validates: Requirements 18.2**

### Property 53: Peer Data Privacy
*For any* individual user, their performance data should remain private and only aggregated anonymous statistics should be shared.
**Validates: Requirements 18.5**

---

## Performance Considerations

### Response Time Targets

**Query Processing:**
- Context retrieval from vector store: <500ms
- LLM response generation: 2-5 seconds (depending on response length)
- Total query response time: <6 seconds

**Image Processing:**
- OCR text extraction: 1-3 seconds
- Diagram matching: <1 second
- Total image processing time: <5 seconds

**Question Generation:**
- Single question: <2 seconds
- Full practice paper (30 questions): <30 seconds

**Search:**
- Semantic search across NCERT content: <1 second
- Result rendering: <500ms

### Memory Management

**Model Memory Usage:**
- LLM (Llama 3.2 3B): ~4GB RAM when loaded
- Embedding model: ~500MB RAM
- OCR models: ~300MB RAM
- Total AI models: ~5GB RAM maximum

**Data Storage:**
- ChromaDB vector store: ~2GB disk space
- NCERT diagrams: ~500MB disk space
- Previous papers database: ~1GB disk space
- User data (SQLite): <100MB per user

**Dynamic Loading Strategy:**
- Load LLM only when needed
- Unload after 10 minutes of inactivity (configurable)
- Keep embedding model loaded (small footprint)
- Lazy load diagrams on demand

### Scalability

**Single-User Performance:**
- Optimized for desktop/laptop usage
- Smooth operation on 8GB RAM systems
- Acceptable performance on 4GB RAM systems (with longer load times)

**Multi-User Support:**
- Each user has isolated data
- Shared AI models (single instance)
- Concurrent users limited by available RAM
- Recommended: 1-3 concurrent users per installation

### Optimization Strategies

**Vector Search Optimization:**
- Use approximate nearest neighbor (ANN) algorithms
- Index optimization for faster retrieval
- Cache frequently accessed embeddings

**LLM Inference Optimization:**
- Use quantized models (4-bit or 8-bit)
- Batch processing where possible
- Context length optimization

**Image Processing Optimization:**
- Resize images before OCR
- Use GPU acceleration if available
- Cache processed images

---

## Security Considerations

### Data Security

**Local Data Protection:**
- User passwords hashed with bcrypt (cost factor 12)
- Session tokens generated with cryptographically secure random
- SQLite database file permissions restricted to application user
- No sensitive data transmitted over network

**Authentication:**
- Session-based authentication with secure tokens
- Session expiration after 24 hours of inactivity
- Password requirements: minimum 8 characters
- Account lockout after 5 failed login attempts

### Privacy

**Data Collection:**
- No telemetry or analytics collected
- No data sent to external servers
- All processing happens locally
- User data never leaves the device

**User Isolation:**
- Each user's data stored separately
- No cross-user data access
- Session isolation enforced

### Input Validation

**Query Validation:**
- Maximum query length: 1000 characters
- Sanitize input to prevent injection attacks
- Validate file uploads (type, size, content)

**Image Upload Validation:**
- Allowed formats: JPG, PNG, PDF
- Maximum file size: 10MB
- Image dimension limits: 10000x10000 pixels
- Malware scanning on upload

### Model Security

**Model Integrity:**
- Verify model checksums on installation
- Prevent model tampering
- Secure model loading process

---

## Deployment Strategy

### Installation Package Structure

```
GuruAI-Installer/
├── app/
│   ├── backend/
│   │   ├── app.py
│   │   ├── models/
│   │   ├── services/
│   │   └── utils/
│   ├── frontend/
│   │   ├── index.html
│   │   ├── static/
│   │   │   ├── css/
│   │   │   ├── js/
│   │   │   └── images/
│   └── templates/
├── data/
│   ├── ncert/
│   │   ├── vector_store/
│   │   ├── diagrams/
│   │   └── content/
│   ├── previous_papers/
│   └── models/
│       ├── llm/
│       ├── embeddings/
│       └── ocr/
├── config/
│   └── default_config.json
├── install.sh (Linux/Mac)
├── install.bat (Windows)
└── README.txt
```

### Installation Process

**Step 1: Pre-Installation Checks**
- Verify system requirements (RAM, disk space, OS)
- Check Python 3.10+ installation
- Verify required dependencies

**Step 2: Content Extraction**
- Extract NCERT content to data directory
- Extract AI models
- Extract previous papers database
- Verify checksums for all files

**Step 3: Environment Setup**
- Create Python virtual environment
- Install required packages from requirements.txt
- Configure database connections
- Initialize vector store

**Step 4: First-Time Setup**
- Run database migrations
- Create default configuration
- Verify model loading
- Run system health check

**Step 5: Launch**
- Start Flask backend server
- Open web interface in default browser
- Display welcome screen

### Update Strategy

**Version Updates:**
- Check for updates on startup (optional)
- Download delta updates only
- Backup user data before update
- Rollback capability if update fails

**Content Updates:**
- NCERT content updates (when new editions release)
- Previous papers updates (annually)
- Model updates (for performance improvements)

---

## Monitoring and Logging

### Logging Strategy

**Log Levels:**
- DEBUG: Detailed diagnostic information
- INFO: General informational messages
- WARNING: Warning messages for potential issues
- ERROR: Error messages for failures
- CRITICAL: Critical failures requiring immediate attention

**Log Categories:**
- Application logs: General application flow
- Query logs: User queries and responses
- Model logs: Model loading/unloading events
- Error logs: Exceptions and failures
- Performance logs: Response times and resource usage

**Log Storage:**
- Logs stored in `logs/` directory
- Rotating file handler (max 10MB per file, keep 5 files)
- Separate log files for different categories
- User-specific logs for debugging

### Health Monitoring

**System Health Checks:**
- Model availability check
- Database connectivity check
- Disk space check
- Memory usage check
- Vector store integrity check

**Performance Monitoring:**
- Track average response times
- Monitor memory usage trends
- Track model load/unload frequency
- Monitor error rates

---

## Future Enhancements

### Planned Features (v2.0)

**Mobile Applications:**
- Android app with offline sync
- iOS app with offline sync
- Cross-platform progress synchronization

**Advanced Features:**
- Video explanations for complex topics
- Study group features
- Spaced repetition system for revision
- Collaborative learning sessions
- Integration with physical smart pens

**Content Expansion:**
- Additional reference books beyond NCERT
- State board content integration
- Foundation course content (Class 9-10)
- Olympiad preparation content

### Technical Improvements

**Performance:**
- GPU acceleration for faster inference
- Model quantization for smaller footprint
- Improved caching strategies
- Parallel processing for batch operations

**AI Capabilities:**
- Fine-tuned models specifically for JEE/NEET content
- Better diagram generation capabilities
- Improved question generation diversity
- Adaptive difficulty adjustment

**User Experience:**
- Dark mode support
- Customizable themes
- Keyboard shortcuts
- Accessibility improvements (screen reader support)
