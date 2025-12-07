# Predictions API Integration Guide

## Overview

This guide provides comprehensive documentation for integrating with GuruAI's prediction APIs. These APIs enable developers to access AI-powered NEET question predictions, chapter analysis, and smart paper generation capabilities.

---

## Table of Contents

1. [Authentication](#authentication)
2. [API Endpoints](#api-endpoints)
3. [Request/Response Formats](#requestresponse-formats)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)

---

## Authentication

### Session-Based Authentication

All prediction APIs require user authentication via session cookies.

**Login Flow:**
```python
import requests

session = requests.Session()

# Login
login_response = session.post(
    'http://localhost:5000/auth/login',
    json={
        'email': 'user@example.com',
        'password': 'password123'
    }
)

# Session cookie is automatically stored
# Use same session for subsequent requests
```

**Headers:**
```python
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}
```

---

## API Endpoints

### 1. Predict Paper

Generate predicted paper for a specific subject.

**Endpoint:** `POST /api/prediction/predict-paper`

**Request Body:**
```json
{
  "subject": "Physics",
  "year": 2026,
  "use_ai": true
}
```

**Parameters:**
- `subject` (string, required): "Physics", "Chemistry", or "Biology"
- `year` (integer, required): Target year (2026-2028)
- `use_ai` (boolean, optional): Use AI enhancement (default: true)

**Response:**
```json
{
  "paper_info": {
    "exam_type": "NEET_PREDICTED",
    "year": 2026,
    "subject": "Physics",
    "question_count": 50,
    "to_attempt": 45,
    "duration_minutes": 60,
    "total_marks": 180,
    "prediction_confidence": 0.85,
    "based_on_years": 20,
    "metadata": {
      "generated_at": "2024-11-25T10:30:00Z",
      "model_version": "1.0"
    }
  },
  "questions": [
    {
      "question_number": 1,
      "question_text": "A particle moves with constant acceleration...",
      "question_type": "MCQ",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B",
      "solution": "Step-by-step solution...",
      "difficulty": "medium",
      "topic": "Kinematics",
      "chapter": "Motion in a Straight Line",
      "marks": 4,
      "negative_marks": -1,
      "prediction_score": 0.92,
      "is_predicted": true
    }
  ]
}
```

---

### 2. Get Insights

Retrieve prediction insights for a subject.

**Endpoint:** `GET /api/prediction/insights/<subject>`

**Parameters:**
- `subject` (path parameter): "Physics", "Chemistry", or "Biology"

**Response:**
```json
{
  "subject": "Physics",
  "high_probability_chapters": [
    "Electrostatics",
    "Current Electricity",
    "Magnetic Effects of Current"
  ],
  "recommended_focus": [
    "Capacitors",
    "Kirchhoff's Laws",
    "Electromagnetic Induction"
  ],
  "difficulty_trend": {
    "easy": 0.25,
    "medium": 0.50,
    "hard": 0.25
  },
  "topic_frequency": {
    "Electrostatics": 18,
    "Current Electricity": 16,
    "Magnetism": 14
  },
  "data_confidence": 0.88,
  "total_questions_analyzed": 1000
}
```

---

### 3. Chapter Analysis

Get chapter-wise prediction probabilities.

**Endpoint:** `GET /api/prediction/chapter-analysis/<subject>`

**Parameters:**
- `subject` (path parameter): "Physics", "Chemistry", or "Biology"

**Response:**
```json
{
  "subject": "Physics",
  "analysis": {
    "chapters": [
      {
        "name": "Electrostatics",
        "class": 12,
        "frequency": 18,
        "probability": 0.90,
        "trend": "increasing",
        "recommended": true,
        "last_appeared": 2024,
        "average_questions": 3.2
      },
      {
        "name": "Current Electricity",
        "class": 12,
        "frequency": 16,
        "probability": 0.85,
        "trend": "stable",
        "recommended": true,
        "last_appeared": 2024,
        "average_questions": 2.8
      }
    ],
    "total_analyzed": 20,
    "confidence": 0.87
  }
}
```

---

### 4. Complete NEET Paper

Generate complete 200-question NEET paper.

**Endpoint:** `GET /api/prediction/complete-neet/<year>`

**Parameters:**
- `year` (path parameter): Target year (2026-2028)

**Response:**
```json
{
  "paper_info": {
    "exam_type": "NEET_PREDICTED",
    "year": 2026,
    "total_questions": 200,
    "to_attempt": 180,
    "duration_minutes": 200,
    "total_marks": 720,
    "prediction_confidence": 0.83,
    "based_on_years": 20,
    "subjects": {
      "Physics": {
        "questions": 50,
        "to_attempt": 45,
        "confidence": 0.85
      },
      "Chemistry": {
        "questions": 50,
        "to_attempt": 45,
        "confidence": 0.82
      },
      "Biology": {
        "questions": 100,
        "to_attempt": 90,
        "confidence": 0.83
      }
    }
  },
  "questions": [
    // 200 questions (same format as predict-paper)
  ]
}
```

---

### 5. Full Prediction

Get comprehensive prediction data for a year.

**Endpoint:** `GET /api/prediction/full-prediction/<year>`

**Parameters:**
- `year` (path parameter): Target year (2026-2028)

**Response:**
```json
{
  "year": 2026,
  "overall_confidence": 0.83,
  "subjects": {
    "Physics": {
      "confidence": 0.85,
      "high_probability_chapters": [...],
      "insights": {...}
    },
    "Chemistry": {
      "confidence": 0.82,
      "high_probability_chapters": [...],
      "insights": {...}
    },
    "Biology": {
      "confidence": 0.83,
      "high_probability_chapters": [...],
      "insights": {...}
    }
  },
  "paper": {
    // Complete paper data
  }
}
```

---

### 6. Smart Paper Generation

Generate personalized practice paper.

**Endpoint:** `POST /api/prediction/smart-paper`

**Request Body:**
```json
{
  "subject": "Physics",
  "focus_chapters": ["Electrostatics", "Current Electricity"],
  "difficulty": "medium",
  "question_count": 20,
  "include_weak_areas": true
}
```

**Parameters:**
- `subject` (string, required): Subject name
- `focus_chapters` (array, required): List of chapter names
- `difficulty` (string, optional): "easy", "medium", "hard", or "mixed"
- `question_count` (integer, optional): Number of questions (10-50)
- `include_weak_areas` (boolean, optional): Include user's weak areas

**Response:**
```json
{
  "paper_info": {
    "exam_type": "SMART_PAPER",
    "subject": "Physics",
    "question_count": 20,
    "difficulty": "medium",
    "focus_chapters": ["Electrostatics", "Current Electricity"],
    "personalization_score": 0.92,
    "weak_areas_included": true
  },
  "questions": [
    // Questions array
  ],
  "targeting_info": {
    "weak_chapters_covered": ["Electrostatics"],
    "high_probability_topics": ["Capacitors", "Kirchhoff's Laws"],
    "difficulty_distribution": {
      "easy": 5,
      "medium": 10,
      "hard": 5
    }
  }
}
```

---

## Request/Response Formats

### Common Data Types

#### Question Object
```typescript
interface Question {
  question_number: number;
  question_text: string;
  question_type: "MCQ" | "Numerical" | "Integer";
  options: string[];
  correct_answer: string;
  solution: string;
  difficulty: "easy" | "medium" | "hard";
  topic: string;
  chapter: string;
  marks: number;
  negative_marks: number;
  prediction_score: number;  // 0.0 - 1.0
  is_predicted: boolean;
}
```

#### Paper Info Object
```typescript
interface PaperInfo {
  exam_type: string;
  year: number;
  subject?: string;
  question_count: number;
  to_attempt: number;
  duration_minutes: number;
  total_marks: number;
  prediction_confidence: number;  // 0.0 - 1.0
  based_on_years: number;
  metadata?: object;
}
```

#### Chapter Analysis Object
```typescript
interface ChapterAnalysis {
  name: string;
  class: 11 | 12;
  frequency: number;
  probability: number;  // 0.0 - 1.0
  trend: "increasing" | "stable" | "decreasing";
  recommended: boolean;
  last_appeared: number;
  average_questions: number;
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_SUBJECT",
    "message": "Invalid subject. Must be Physics, Chemistry, or Biology.",
    "details": {
      "provided": "Maths",
      "allowed": ["Physics", "Chemistry", "Biology"]
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | User not authenticated |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `INVALID_SUBJECT` | 400 | Invalid subject parameter |
| `INVALID_YEAR` | 400 | Invalid year parameter |
| `INVALID_DIFFICULTY` | 400 | Invalid difficulty level |
| `INSUFFICIENT_DATA` | 422 | Not enough data for prediction |
| `GENERATION_FAILED` | 500 | Paper generation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

### Error Handling Example

```python
try:
    response = session.post(
        'http://localhost:5000/api/prediction/predict-paper',
        json={'subject': 'Physics', 'year': 2026}
    )
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Authentication required")
    elif e.response.status_code == 403:
        print("Insufficient permissions")
    elif e.response.status_code == 400:
        error_data = e.response.json()
        print(f"Validation error: {error_data['error']['message']}")
    else:
        print(f"Error: {e}")
```

---

## Rate Limiting

### Limits

- **Predict Paper**: 10 requests per minute
- **Chapter Analysis**: 20 requests per minute
- **Insights**: 20 requests per minute
- **Complete NEET**: 5 requests per minute
- **Smart Paper**: 10 requests per minute

### Rate Limit Headers

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1700000000
```

### Handling Rate Limits

```python
import time

def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = session.get(url)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        return response
    
    raise Exception("Max retries exceeded")
```

---

## Code Examples

### Python

#### Complete Workflow Example

```python
import requests
from typing import Dict, List

class PredictionClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def login(self, email: str, password: str) -> bool:
        """Authenticate user"""
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={'email': email, 'password': password}
        )
        return response.status_code == 200
    
    def predict_paper(self, subject: str, year: int, use_ai: bool = True) -> Dict:
        """Generate predicted paper"""
        response = self.session.post(
            f"{self.base_url}/api/prediction/predict-paper",
            json={
                'subject': subject,
                'year': year,
                'use_ai': use_ai
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_chapter_analysis(self, subject: str) -> Dict:
        """Get chapter-wise analysis"""
        response = self.session.get(
            f"{self.base_url}/api/prediction/chapter-analysis/{subject}"
        )
        response.raise_for_status()
        return response.json()
    
    def get_insights(self, subject: str) -> Dict:
        """Get prediction insights"""
        response = self.session.get(
            f"{self.base_url}/api/prediction/insights/{subject}"
        )
        response.raise_for_status()
        return response.json()
    
    def generate_complete_neet(self, year: int) -> Dict:
        """Generate complete NEET paper"""
        response = self.session.get(
            f"{self.base_url}/api/prediction/complete-neet/{year}"
        )
        response.raise_for_status()
        return response.json()
    
    def generate_smart_paper(
        self,
        subject: str,
        focus_chapters: List[str],
        difficulty: str = "medium",
        question_count: int = 20
    ) -> Dict:
        """Generate smart paper"""
        response = self.session.post(
            f"{self.base_url}/api/prediction/smart-paper",
            json={
                'subject': subject,
                'focus_chapters': focus_chapters,
                'difficulty': difficulty,
                'question_count': question_count
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
client = PredictionClient()

# Login
if client.login('user@example.com', 'password123'):
    # Get chapter analysis
    analysis = client.get_chapter_analysis('Physics')
    print(f"High probability chapters: {len(analysis['analysis']['chapters'])}")
    
    # Generate predicted paper
    paper = client.predict_paper('Physics', 2026)
    print(f"Generated {len(paper['questions'])} questions")
    print(f"Confidence: {paper['paper_info']['prediction_confidence']:.2%}")
    
    # Get insights
    insights = client.get_insights('Chemistry')
    print(f"Recommended topics: {insights['recommended_focus']}")
    
    # Generate smart paper
    smart_paper = client.generate_smart_paper(
        subject='Biology',
        focus_chapters=['Cell Biology', 'Genetics'],
        difficulty='medium',
        question_count=25
    )
    print(f"Smart paper generated with {len(smart_paper['questions'])} questions")
```

### JavaScript

#### Frontend Integration Example

```javascript
class PredictionAPI {
  constructor(baseURL = '') {
    this.baseURL = baseURL;
  }

  async predictPaper(subject, year, useAI = true) {
    const response = await fetch(`${this.baseURL}/api/prediction/predict-paper`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({ subject, year, use_ai: useAI })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async getChapterAnalysis(subject) {
    const response = await fetch(
      `${this.baseURL}/api/prediction/chapter-analysis/${subject}`,
      { credentials: 'include' }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async getInsights(subject) {
    const response = await fetch(
      `${this.baseURL}/api/prediction/insights/${subject}`,
      { credentials: 'include' }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async generateCompleteNEET(year) {
    const response = await fetch(
      `${this.baseURL}/api/prediction/complete-neet/${year}`,
      { credentials: 'include' }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async generateSmartPaper(config) {
    const response = await fetch(`${this.baseURL}/api/prediction/smart-paper`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify(config)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

// Usage
const api = new PredictionAPI();

// Generate prediction
try {
  const paper = await api.predictPaper('Physics', 2026);
  console.log(`Generated ${paper.questions.length} questions`);
  console.log(`Confidence: ${(paper.paper_info.prediction_confidence * 100).toFixed(1)}%`);
} catch (error) {
  console.error('Failed to generate prediction:', error);
}

// Get chapter analysis
try {
  const analysis = await api.getChapterAnalysis('Chemistry');
  analysis.analysis.chapters.forEach(chapter => {
    console.log(`${chapter.name}: ${(chapter.probability * 100).toFixed(1)}%`);
  });
} catch (error) {
  console.error('Failed to get analysis:', error);
}
```

---

## Best Practices

### 1. Caching

Cache prediction results to reduce API calls:

```python
from functools import lru_cache
import time

class CachedPredictionClient(PredictionClient):
    def __init__(self, *args, cache_ttl=3600, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_ttl = cache_ttl
        self._cache = {}
    
    def _get_cache_key(self, method, *args):
        return f"{method}:{':'.join(map(str, args))}"
    
    def _is_cache_valid(self, timestamp):
        return time.time() - timestamp < self.cache_ttl
    
    def get_chapter_analysis(self, subject):
        cache_key = self._get_cache_key('chapter_analysis', subject)
        
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                return data
        
        data = super().get_chapter_analysis(subject)
        self._cache[cache_key] = (data, time.time())
        return data
```

### 2. Error Handling

Implement robust error handling:

```python
def safe_api_call(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    time.sleep(retry_after)
                    continue
                raise
        
        raise Exception("Max retries exceeded")
    
    return wrapper
```

### 3. Batch Processing

Process multiple subjects efficiently:

```python
from concurrent.futures import ThreadPoolExecutor

def get_all_chapter_analyses(client):
    subjects = ['Physics', 'Chemistry', 'Biology']
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(client.get_chapter_analysis, subject): subject
            for subject in subjects
        }
        
        results = {}
        for future in futures:
            subject = futures[future]
            try:
                results[subject] = future.result()
            except Exception as e:
                print(f"Failed to get analysis for {subject}: {e}")
        
        return results
```

### 4. Progress Tracking

Track long-running operations:

```python
def generate_complete_neet_with_progress(client, year):
    print("Generating complete NEET paper...")
    print("This may take 15-30 seconds...")
    
    import sys
    for i in range(30):
        sys.stdout.write('\r')
        sys.stdout.write(f"Progress: [{'=' * i}{' ' * (30-i)}] {i*100//30}%")
        sys.stdout.flush()
        time.sleep(0.5)
    
    paper = client.generate_complete_neet(year)
    print("\n✓ Paper generated successfully!")
    return paper
```

### 5. Data Validation

Validate responses:

```python
def validate_paper_response(paper):
    required_fields = ['paper_info', 'questions']
    
    for field in required_fields:
        if field not in paper:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(paper['questions'], list):
        raise ValueError("Questions must be a list")
    
    if len(paper['questions']) == 0:
        raise ValueError("Paper has no questions")
    
    confidence = paper['paper_info'].get('prediction_confidence', 0)
    if not 0 <= confidence <= 1:
        raise ValueError(f"Invalid confidence score: {confidence}")
    
    return True
```

---

## Support

For API support and questions:

- **Documentation**: [Main README](../README.md)
- **User Guide**: [Predictions User Guide](./PREDICTIONS_USER_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Email**: api-support@guruai.com

---

**Last Updated:** November 2024
**API Version:** 1.0
