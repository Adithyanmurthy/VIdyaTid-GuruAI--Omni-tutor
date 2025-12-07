# GuruAI - The AI Omni-Tutor for JEE & NEET

> **Your Personal AI Study Partner - Completely Offline, 100% NCERT-Based**

GuruAI is a revolutionary offline-first AI learning companion designed specifically for Indian students preparing for JEE and NEET competitive exams. Unlike other learning platforms, GuruAI works completely offline, requires no API keys, and is based entirely on NCERT textbooks - the foundation of all competitive exams in India.

---

## 🎯 The Problem We Solve

Indian students preparing for JEE and NEET face several critical challenges:

1. **Fragmented Learning Experience**: Students switch between multiple apps for doubt-solving, practice questions, and concept revision
2. **Passive Learning**: Most digital tools offer passive content consumption without interactive engagement
3. **Internet Dependency**: Many students in Tier 2/3 cities have limited or unreliable internet access
4. **Non-NCERT Content**: Most platforms provide content that doesn't align with NCERT, creating confusion
5. **Expensive Solutions**: Premium learning platforms require costly subscriptions and API access

**GuruAI solves all these problems in one integrated, offline, conversational platform.**

---

## ✨ What Makes GuruAI Unique

### 1. **100% Offline Operation**
- No internet required after installation
- All AI models run locally on your computer
- Complete privacy - your data never leaves your device
- Works in areas with poor connectivity

### 2. **Pure NCERT Content**
- Based entirely on NCERT textbooks (Classes 11-12)
- Physics, Chemistry, Mathematics, and Biology
- Includes all diagrams and illustrations from NCERT books
- Ensures alignment with JEE/NEET syllabus

### 3. **Conversational Learning**
- Ask questions naturally in English or Hinglish
- Get step-by-step explanations
- Interactive quizzes after every concept
- Like having a personal tutor available 24/7

### 4. **Snap & Solve**
- Take a photo of any problem from your textbook
- AI recognizes and solves it step-by-step
- Works with diagrams, equations, and text
- No need to type complex mathematical expressions

### 5. **20 Years of Previous Papers**
- Complete question bank from past JEE Main, JEE Advanced, and NEET exams
- Practice with real exam patterns
- Detailed solutions for every question
- Custom question paper generation

### 6. **Smart Resource Management**
- Dynamic AI model loading - uses memory only when needed
- Runs smoothly on student laptops (4-8GB RAM)
- Battery-efficient design
- Configurable performance settings

---

## 🚀 Core Features

### 📚 NCERT-Based Question Answering
Ask any question about Physics, Chemistry, Maths, or Biology concepts:
- **Query**: "Explain photoelectric effect"
- **Response**: Detailed explanation from NCERT Physics Class 12, Chapter 11
- **Includes**: Relevant diagrams, formulas, and examples
- **Follow-up**: Adaptive quiz questions to test understanding

### 📸 Image Recognition & Problem Solving
Stuck on a problem? Just snap a picture:
- Upload image of any problem from your textbook
- AI extracts text and visual elements using OCR
- Provides step-by-step solution based on NCERT methodology
- Works with mathematical equations, chemical reactions, and biological diagrams

### 🎯 Practice Question Generation
Generate custom practice tests:
- Select specific chapters or topics
- Choose difficulty level (easy, medium, hard)
- Get questions matching JEE/NEET patterns
- Instant feedback with detailed solutions

### 🔮 AI-Powered Question Predictions
Prepare smarter with AI predictions based on 20 years of NEET analysis:
- **Subject-wise Predictions**: Generate predicted papers for Physics, Chemistry, or Biology
- **Chapter Analysis**: View probability scores for each chapter based on historical patterns
- **Complete NEET Papers**: Generate full 200-question predicted papers for upcoming years
- **Smart Paper Generation**: Create personalized practice papers targeting your weak areas
- **Prediction Insights**: Understand topic frequency, difficulty trends, and high-probability areas
- **Accuracy Tracking**: View historical accuracy of predictions vs actual papers

### 📊 Progress Tracking
Monitor your preparation journey:
- Track topics studied and questions attempted
- View accuracy percentage by subject and chapter
- Identify weak areas automatically
- Get personalized recommendations

### 🔍 Smart Search
Find information across all NCERT content:
- Semantic search - understands meaning, not just keywords
- Search by topic, concept, or keyword
- Results show subject, class, chapter, and page number
- Includes diagrams and illustrations in search results

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 20 GB free space
- **Processor**: Intel i3 / AMD Ryzen 3 or better
- **Internet**: Only for initial download

### Recommended Requirements
- **RAM**: 8 GB or more
- **Storage**: 30 GB free space (SSD preferred)
- **Processor**: Intel i5 / AMD Ryzen 5 or better
- **GPU**: Optional (speeds up image processing)

---

## 📥 Installation Guide

### Windows Installation

#### Step 1: Install Python
1. Download Python 3.10+ from python.org
2. During installation, check "Add Python to PATH"
3. Verify: `python --version`

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Install Poppler (for PDF processing)
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract to: `C:\Program Files\poppler`
3. Add to PATH: `C:\Program Files\poppler\Library\bin`
4. Verify: `pdftoppm -v`

**Quick Install (Alternative):**
```bash
choco install poppler
```

#### Step 4: Install Tesseract OCR (for image extraction)
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer (tesseract-ocr-w64-setup-v5.x.x.exe)
3. Note installation path (usually `C:\Program Files\Tesseract-OCR`)
4. Add to PATH or set in Python
5. Verify: `tesseract --version`

#### Step 5: Install llama-cpp-python
```bash
pip install llama-cpp-python==0.2.90
```

**Note**: Version 0.2.90 provides pre-built wheels for Windows. For newer versions, you'll need Visual Studio Build Tools.

#### Step 6: Download AI Model
```bash
python setup_local_llm.py
```

This downloads Llama 3.2 3B model (~2GB, one-time download).

#### Step 7: Initialize Database
```bash
python init_db.py
```

---

## 🎮 Quick Start

### 1. Setup NCERT Content

```bash
# Run NCERT pipeline setup
python setup_ncert_pipeline.py
```

**Add NCERT PDFs:**
- Place PDFs in `ncert_content/pdfs/`
- Use naming format: `Subject_Class_Chapter.pdf`
  - Example: `Physics_12_01.pdf`, `Chemistry_11_05.pdf`

**Process PDFs:**
```bash
python setup_ncert_pipeline.py
```

This will:
- Extract text from PDFs
- Create vector embeddings
- Store in local database
- Enable semantic search

**Verify:**
```bash
python test_ncert_setup.py
```

### 2. Setup Previous Papers

```bash
# Initialize previous papers database
python setup_previous_papers.py
```

**Import Questions:**

From JSON:
```bash
python utils/question_importer.py import --file questions.json --format json
```

From CSV:
```bash
python utils/question_importer.py import --file questions.csv --format csv
```

From directory:
```bash
python utils/question_importer.py import --directory ./previous_papers --format json
```

**Generate Templates:**
```bash
# JSON template
python utils/question_importer.py template --format json --output template.json

# CSV template
python utils/question_importer.py template --format csv --output template.csv
```

### 3. Extract Previous Year Papers

**For Arihant PDFs (text-based):**
```bash
python improved_arihant_extractor.py
```

**For Disha PDFs (image-based, requires OCR):**
```bash
# First install Tesseract OCR (see installation guide)
python ocr_disha_chemistry.py
```

**Auto-extract NEET papers:**
```bash
python auto_extract_neet_papers.py
```

### 4. Start the Application

```bash
python app.py
```

Open browser: `http://localhost:5000`

---

## 🔮 Using AI Question Predictions

GuruAI's prediction system analyzes 20 years of NEET papers to identify patterns and predict high-probability topics for upcoming exams.

### Accessing Predictions

1. **Navigate to Predictions Page**
   - Click "Predictions" in the main navigation menu
   - Or visit: `http://localhost:5000/predictions`

2. **Authentication Required**
   - You must be logged in to access predictions
   - Some advanced features may require specific subscription tiers

### Prediction Features

#### 1. Subject-wise Predictions

Generate predicted papers for individual subjects:

1. Select subject (Physics/Chemistry/Biology)
2. Choose target year (2026-2028)
3. Click "Generate Prediction"
4. View predicted paper with confidence scores

**What you get:**
- 50 questions for Physics/Chemistry
- 100 questions for Biology
- Confidence score for each prediction (0-100%)
- High-probability topics highlighted
- Difficulty distribution chart

#### 2. Chapter Analysis

View chapter-wise prediction probabilities:

1. Click on subject tab (Physics/Chemistry/Biology)
2. View chapters sorted by probability
3. See prediction confidence for each chapter
4. Filter by class (11/12)

**Insights provided:**
- Probability score for each chapter (0-100%)
- Historical frequency data
- Recommended focus areas
- Class 11 vs Class 12 weightage

#### 3. Complete NEET Predictions

Generate full 200-question predicted papers:

1. Select target year
2. Click "Generate Complete NEET Paper"
3. View complete paper with all subjects

**NEET 2026 Pattern (New):**
- Physics: 45 questions (all compulsory)
- Chemistry: 45 questions (all compulsory)
- Biology: 90 questions (all compulsory)
- Total: 180 questions (all compulsory)
- Duration: 180 minutes (3 hours)
- Overall confidence score displayed

**NEET 2025 Pattern (Legacy):**
- Physics: 50 questions (45 to attempt)
- Chemistry: 50 questions (45 to attempt)
- Biology: 100 questions (90 to attempt)
- Total: 200 questions (180 to attempt)
- Duration: 200 minutes (3 hours 20 minutes)

#### 4. Smart Paper Generation

Create personalized practice papers:

1. Select subject
2. Choose focus chapters (multi-select)
3. Set difficulty level (easy/medium/hard/mixed)
4. Click "Generate Smart Paper"

**Benefits:**
- Targets your weak areas
- Combines high-probability topics with your needs
- Adaptive difficulty based on your performance
- Optimized for efficient learning

#### 5. Prediction Insights

Understand the reasoning behind predictions:

1. Select a subject
2. Click "View Insights"
3. Explore detailed analysis

**Insights include:**
- Topic frequency trends over years
- Difficulty distribution (Easy/Medium/Hard)
- Increasing/decreasing topic trends
- Historical pattern analysis
- Data confidence metrics

#### 6. Accuracy Tracking

View prediction system reliability:

- Overall accuracy percentage
- Subject-wise accuracy breakdown
- Year-over-year accuracy trends
- Number of papers analyzed
- Confidence in predictions

### Understanding Confidence Scores

Predictions include confidence scores to help you assess reliability:

- **Green (75-100%)**: High confidence - strong historical pattern
- **Yellow (50-74%)**: Medium confidence - moderate pattern
- **Red (0-49%)**: Low confidence - weak or emerging pattern

### Best Practices

1. **Use predictions as a guide**, not the only study material
2. **Focus on high-confidence predictions** for efficient preparation
3. **Combine predictions with NCERT fundamentals** for comprehensive coverage
4. **Track accuracy metrics** to understand prediction reliability
5. **Generate smart papers** regularly to target weak areas
6. **Review chapter analysis** to prioritize study topics

### API Integration

Predictions are powered by backend APIs:

```python
# Available endpoints
/api/prediction/predict-paper          # Subject-wise predictions
/api/prediction/insights/<subject>     # Prediction insights
/api/prediction/chapter-analysis/<subject>  # Chapter probabilities
/api/prediction/complete-neet/<year>   # Complete NEET paper
/api/prediction/full-prediction/<year> # Full prediction with all data
/api/prediction/smart-paper            # Personalized smart papers
```

See [API Documentation](#api-usage) for detailed integration guide.

---

## 🏗️ How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface                         │
│              (HTML/CSS/JavaScript)                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Flask Backend (Python)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Query      │  │    Image     │  │   Question   │  │
│  │  Processing  │  │  Processing  │  │  Generation  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              RAG System (Retrieval)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ChromaDB Vector Store                           │   │
│  │  - NCERT Textbooks (Embedded)                    │   │
│  │  - Previous Papers (Indexed)                     │   │
│  │  - Diagrams (Image Embeddings)                   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Local AI Models                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   LLM        │  │  Embedding   │  │     OCR      │  │
│  │ (Llama/Phi)  │  │    Model     │  │   (Tesseract)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Configuration

### Model Settings (config.py)

```python
# AI Models
MODEL_DIR = BASE_DIR / 'ai_models'
LLM_MODEL_PATH = MODEL_DIR / 'llama-3.2-3b-instruct-q4_k_m.gguf'

# Model Manager Settings
MODEL_IDLE_TIMEOUT = 600  # 10 minutes
MODEL_MEMORY_LIMIT = 4 * 1024 * 1024 * 1024  # 4GB

# LLM Configuration
LLM_N_CTX = 2048  # Context window
LLM_N_GPU_LAYERS = 0  # CPU only (set to 35 for GPU)
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
```

### Performance Optimization

**For GPU acceleration (NVIDIA):**
```python
config = {
    'n_gpu_layers': 35,  # Offload layers to GPU
}
```

**Adjust context window:**
```python
config = {
    'n_ctx': 2048,  # Default: good balance
    # 'n_ctx': 4096,  # Larger: more context, more memory
    # 'n_ctx': 1024,  # Smaller: less memory, less context
}
```

---

## 📚 Usage Examples

### Python API

```python
from services.model_manager import ModelManagerSingleton
from services.ncert_processor import NCERTProcessor
from config import Config

# Initialize NCERT processor
processor = NCERTProcessor(
    str(Config.NCERT_PDF_DIR),
    str(Config.VECTOR_STORE_DIR)
)

# Query NCERT content
results = processor.query_vector_store("Newton's laws of motion", top_k=3)

# Get model manager
manager = ModelManagerSingleton.get_instance(
    model_path=str(Config.LLM_MODEL_PATH),
    config={
        'idle_timeout': Config.MODEL_IDLE_TIMEOUT,
        'n_ctx': 2048,
        'temperature': 0.7
    }
)

# Generate response
result = manager.generate("Explain Newton's laws of motion")
print(result['text'])
```

### Query Previous Papers

```python
from models import Question
from models.database import SessionLocal

db = SessionLocal()

# Get JEE Physics questions
jee_physics = Question.get_by_filters(
    db, 
    exam='JEE Main',
    subject='Physics',
    difficulty='medium',
    limit=10
)

# Get random practice questions
random_questions = Question.get_random_questions(
    db,
    count=5,
    subject='Chemistry',
    difficulty='hard'
)

# Get statistics
stats = Question.get_statistics(db, exam='NEET', subject='Biology')
print(f"Total questions: {stats['total_questions']}")

db.close()
```

### Using Prediction APIs

```python
import requests

# Base URL
BASE_URL = "http://localhost:5000"

# Predict subject-specific paper
response = requests.post(
    f"{BASE_URL}/api/prediction/predict-paper",
    json={
        "subject": "Physics",
        "year": 2026,
        "use_ai": True
    },
    headers={"Content-Type": "application/json"}
)
predicted_paper = response.json()

# Get chapter analysis
response = requests.get(
    f"{BASE_URL}/api/prediction/chapter-analysis/Chemistry"
)
chapter_analysis = response.json()

# Get prediction insights
response = requests.get(
    f"{BASE_URL}/api/prediction/insights/Biology"
)
insights = response.json()

# Generate complete NEET paper
response = requests.get(
    f"{BASE_URL}/api/prediction/complete-neet/2026"
)
complete_paper = response.json()

# Generate smart paper
response = requests.post(
    f"{BASE_URL}/api/prediction/smart-paper",
    json={
        "subject": "Physics",
        "focus_chapters": ["Electrostatics", "Current Electricity"],
        "difficulty": "medium",
        "question_count": 20
    }
)
smart_paper = response.json()

# Response structure
print(f"Paper confidence: {predicted_paper['paper_info']['prediction_confidence']}")
print(f"Questions: {len(predicted_paper['questions'])}")
print(f"High probability chapters: {insights['high_probability_chapters']}")
```

---

## 🔧 Troubleshooting

### Model Download Fails
```bash
# Force re-download
python setup_local_llm.py --force
```

### Out of Memory
1. Use 4-bit quantization instead of 8-bit
2. Reduce context window: `n_ctx=1024`
3. Close other applications
4. Reduce `max_tokens` in generation

### PDF Extraction Returns Empty
- Ensure PDFs have selectable text (not scanned images)
- For scanned PDFs, use OCR extraction
- Verify Poppler is installed and in PATH

### OCR Not Working
1. Install Tesseract OCR
2. Add to PATH or set path in Python:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
3. Install Python packages: `pip install pytesseract pillow`

### Slow Inference
1. Use GPU acceleration if available
2. Reduce `max_tokens`
3. Use smaller model
4. Ensure no other heavy processes running

---

## 📊 Database Schema

### Questions Table
```python
class Question:
    question_id: str          # Unique identifier (UUID)
    source: str              # e.g., "JEE Main 2023"
    year: int                # Year of exam
    exam: str                # JEE Main, JEE Advanced, NEET
    subject: str             # Physics, Chemistry, Mathematics, Biology
    chapter: str             # Chapter name
    topic: str               # Specific topic
    difficulty: str          # easy, medium, hard
    question_text: str       # The question
    question_type: str       # MCQ, Numerical, Integer, etc.
    options: list            # List of options (for MCQs)
    correct_answer: str      # Correct answer
    solution: str            # Detailed solution with steps
    ncert_reference: str     # NCERT chapter and page reference
    marks: int               # Marks allocated
```

---

## 🎓 Project Structure

```
GuruAI-Omni Tutor/
├── ai_models/                  # AI model files
├── ncert_content/              # NCERT PDFs and processed data
│   └── pdfs/                   # Place NCERT PDFs here
├── previous_papers/            # Previous year papers
│   ├── JEE/
│   └── NEET/
├── vector_store/               # ChromaDB vector database
├── services/                   # Core services
│   ├── model_manager.py        # AI model management
│   ├── ncert_processor.py      # NCERT content processing
│   ├── question_generator.py   # Question generation
│   └── query_handler.py        # Query processing
├── routes/                     # Flask routes
├── templates/                  # HTML templates
├── static/                     # CSS, JS, images
├── models/                     # Database models
├── utils/                      # Utility functions
├── app.py                      # Main Flask application
├── config.py                   # Configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- **Content**: Add more previous year questions
- **Models**: Optimize AI models for better performance
- **Features**: Suggest and implement new features
- **Documentation**: Improve guides and tutorials
- **Testing**: Report bugs and test on different systems

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Note**: NCERT content is freely available for educational use as per Government of India guidelines.

---

## 👥 Team

- **Ajay Kumar T N** - Lead Developer
- **Adithya N Murthy** - AI Specialist
- **Abhishek Biradar** - UI/UX Designer

---

## 🎯 Roadmap

### Current Version (v1.0)
- ✅ Offline NCERT-based Q&A
- ✅ Image recognition and problem solving
- ✅ 20 years previous papers
- ✅ AI-powered question predictions
- ✅ Chapter-wise prediction analysis
- ✅ Smart paper generation
- ✅ Progress tracking
- ✅ Custom question generation
- ✅ Voice input/output (Hinglish support)
- ✅ Handwriting recognition
- ✅ Deep explanations for complex topics
- ✅ Peer comparison (anonymous)

### Upcoming Features (v2.0)
- 🔄 Mobile app (Android/iOS)
- 🔄 Video explanations for complex topics
- 🔄 Collaborative study groups
- 🔄 Adaptive learning paths
- 🔄 Integration with physical smart pens

---

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) folder:

- **[User Guides](./docs/USER_SUBSCRIPTION_GUIDE.md)** - Subscription tiers, payments, and troubleshooting
- **[API Documentation](./docs/API_DOCUMENTATION.md)** - Complete API reference
- **[Setup Guides](./docs/)** - Razorpay, SendGrid, Cloudflare setup
- **[Deployment Guides](./docs/DEPLOYMENT_GUIDE.md)** - Production deployment instructions
- **[Implementation Details](./docs/)** - Technical implementation documentation

See the [Documentation Index](./docs/README.md) for a complete list of available guides.

---

## 🙏 Acknowledgments

- **NCERT** for providing free educational content
- **Open Source Community** for amazing AI models
- **Students** who provided feedback during development

---

**Made with ❤️ for Indian Students**

*GuruAI - Because every student deserves a personal tutor*
