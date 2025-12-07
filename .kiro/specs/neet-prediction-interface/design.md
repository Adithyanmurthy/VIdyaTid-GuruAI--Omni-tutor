# Design Document

## Overview

This design document outlines the frontend interface implementation for the existing NEET question prediction backend system. The system will provide students with AI-powered predictions based on 20 years of NEET analysis, chapter-wise insights, accuracy tracking, and smart paper generation capabilities. The frontend will integrate seamlessly with existing backend APIs and follow the established dark theme design patterns used throughout the GuruAI platform.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Predictions  │  │ Question     │  │ Visualizations│      │
│  │ Page         │  │ Paper Page   │  │ (Charts.js)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend APIs (Existing)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ /api/prediction/predict-paper                        │   │
│  │ /api/prediction/insights/<subject>                   │   │
│  │ /api/prediction/smart-paper                          │   │
│  │ /api/prediction/chapter-analysis/<subject>           │   │
│  │ /api/prediction/complete-neet/<year>                 │   │
│  │ /api/prediction/full-prediction/<year>               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Services Layer (Existing)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ QuestionPredictor Service                            │   │
│  │ - Pattern Analysis                                   │   │
│  │ - Confidence Calculation                             │   │
│  │ - Smart Paper Generation                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Structure

1. **Predictions Page** (`templates/predictions.html`)
   - Main landing page for all prediction features
   - Dashboard-style layout with feature cards
   - Navigation integration

2. **Prediction JavaScript Module** (`static/js/predictions.js`)
   - API communication layer
   - State management for predictions
   - Chart rendering and data visualization
   - Event handling

3. **Prediction Styles** (`static/css/predictions.css`)
   - Dark theme styling matching existing pages
   - Responsive design for mobile/tablet
   - Chart and visualization styling

4. **Question Paper Integration**
   - Enhanced existing question-paper.html
   - AI Predictions section added
   - Seamless integration with existing paper generation

## Components and Interfaces

### 1. Predictions Page Components

#### 1.1 Prediction Dashboard
- **Purpose**: Central hub for all prediction features
- **Components**:
  - Feature cards for each prediction type
  - Quick access buttons
  - Recent predictions history
  - Accuracy metrics display

#### 1.2 Subject-wise Prediction Panel
- **Purpose**: Generate predictions for individual subjects
- **Inputs**:
  - Subject selector (Physics/Chemistry/Biology)
  - Year selector (2026-2028)
  - Generate button
- **Outputs**:
  - Predicted paper with confidence scores
  - High-probability topics list
  - Difficulty distribution chart

#### 1.3 Chapter Analysis Viewer
- **Purpose**: Display chapter-wise prediction probabilities
- **Components**:
  - Subject tabs (Physics/Chemistry/Biology)
  - Chapter list with probability bars
  - Sortable by probability/name
  - Filter by class (11/12)
- **Visualization**: Horizontal bar chart showing probability percentages

#### 1.4 Complete NEET Prediction Panel
- **Purpose**: Generate full 200-question predicted papers
- **Inputs**:
  - Year selector
  - Generate button
- **Outputs**:
  - Complete paper with all subjects
  - Overall confidence score
  - Subject-wise breakdown
  - Start test button

#### 1.5 Prediction Insights Panel
- **Purpose**: Show detailed analysis and reasoning
- **Components**:
  - Topic frequency charts
  - Trend analysis graphs
  - Difficulty distribution pie chart
  - Historical pattern visualization

#### 1.6 Accuracy Tracker
- **Purpose**: Display historical accuracy metrics
- **Components**:
  - Overall accuracy percentage
  - Subject-wise accuracy breakdown
  - Year-over-year comparison chart
  - Papers analyzed count

#### 1.7 Smart Paper Generator
- **Purpose**: Create personalized practice papers
- **Inputs**:
  - Subject selection
  - Focus chapters (multi-select)
  - Difficulty level (easy/medium/hard/mixed)
  - Generate button
- **Outputs**:
  - Customized practice paper
  - Weak area targeting info

### 2. API Integration Layer

#### 2.1 PredictionAPI Class
```javascript
class PredictionAPI {
  // Predict subject-specific paper
  async predictPaper(subject, year, useAI = true)
  
  // Get prediction insights
  async getInsights(subject)
  
  // Generate smart paper
  async generateSmartPaper(subject, focusChapters, difficultyLevel)
  
  // Get chapter analysis
  async getChapterAnalysis(subject)
  
  // Get complete NEET prediction
  async getCompleteNEET(year)
  
  // Get full prediction (all subjects)
  async getFullPrediction(year)
}
```

#### 2.2 Error Handling
- Network error handling with retry logic
- Permission error handling (subscription tier checks)
- User-friendly error messages
- Upgrade prompts for premium features

### 3. Visualization Components

#### 3.1 Chart Types
- **Bar Charts**: Chapter probability distribution
- **Line Charts**: Accuracy trends over years
- **Pie Charts**: Difficulty distribution
- **Progress Bars**: Confidence scores

#### 3.2 Chart.js Integration
- Responsive charts
- Dark theme color palette
- Interactive tooltips
- Animation on load

### 4. Question Paper Page Integration

#### 4.1 AI Predictions Section
- New card in paper-type-cards grid
- "AI Powered" badge
- Quick access to predictions
- Confidence indicator

#### 4.2 Enhanced Configuration
- Prediction options in existing config flow
- Subject/year selectors
- Generate predicted paper button

## Data Models

### Frontend Data Structures

#### PredictionPaper
```javascript
{
  paper_info: {
    exam_type: string,           // "NEET_PREDICTED"
    year: number,                // 2026
    subject: string,             // "Physics"
    question_count: number,      // 50
    to_attempt: number,          // 45
    duration_minutes: number,    // 60
    total_marks: number,         // 180
    prediction_confidence: number, // 0.85
    based_on_years: number,      // 5
    metadata: object
  },
  questions: [
    {
      question_number: number,
      question_text: string,
      question_type: string,     // "MCQ"
      options: string[],
      correct_answer: string,
      solution: string,
      difficulty: string,        // "easy"|"medium"|"hard"
      topic: string,
      chapter: string,
      marks: number,
      negative_marks: number,
      prediction_score: number,  // 0.0-1.0
      is_predicted: boolean
    }
  ]
}
```

#### PredictionInsights
```javascript
{
  subject: string,
  high_probability_chapters: string[],
  recommended_focus: string[],
  difficulty_trend: {
    easy: number,
    medium: number,
    hard: number
  },
  data_confidence: number,
  total_questions_analyzed: number
}
```

#### ChapterAnalysis
```javascript
{
  analysis: {
    chapters: [
      {
        name: string,
        frequency: number,
        probability: number,      // 0.0-1.0
        recommended: boolean
      }
    ],
    total_analyzed: number
  }
}
```

#### CompletePaper
```javascript
{
  paper_info: {
    exam_type: string,           // "NEET_PREDICTED"
    year: number,
    total_questions: number,     // 200
    to_attempt: number,          // 180
    duration_minutes: number,    // 200
    total_marks: number,         // 720
    prediction_confidence: number,
    based_on_years: number,
    subjects: {
      Physics: { questions: string, to_attempt: number, confidence: number },
      Chemistry: { questions: string, to_attempt: number, confidence: number },
      Biology: { questions: string, to_attempt: number, confidence: number }
    }
  },
  questions: Question[]          // 200 questions
}
```

#### AccuracyMetrics
```javascript
{
  overall_accuracy: number,      // 0.80 (80%)
  subject_accuracy: {
    Physics: number,
    Chemistry: number,
    Biology: number
  },
  yearly_accuracy: [
    { year: number, accuracy: number }
  ],
  papers_analyzed: number
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: API Response Validation
*For any* API call to prediction endpoints, the response should contain all required fields as defined in the data models, and confidence scores should be between 0 and 1.
**Validates: Requirements 1.2, 1.3, 2.1, 5.1**

### Property 2: Confidence Score Display Consistency
*For any* prediction displayed to the user, the confidence score visualization (color coding) should match the numerical value (green for ≥0.75, yellow for 0.5-0.74, red for <0.5).
**Validates: Requirements 1.3, 8.2**

### Property 3: Chapter Probability Ordering
*For any* chapter analysis display, chapters should be ordered by probability from highest to lowest when default sort is applied.
**Validates: Requirements 2.3**

### Property 4: Complete Paper Structure
*For any* complete NEET paper generated, it should contain exactly 200 questions with the correct subject distribution (Physics: 50, Chemistry: 50, Biology: 100).
**Validates: Requirements 6.2, 6.3**

### Property 5: Navigation Integration
*For any* authenticated user, the predictions page link should be visible in the main navigation menu and clicking it should load the predictions page.
**Validates: Requirements 7.1, 7.2**

### Property 6: Responsive Visualization Adaptation
*For any* viewport width, charts and visualizations should adapt their dimensions to fit the screen without horizontal scrolling.
**Validates: Requirements 8.4**

### Property 7: Authentication Requirement
*For any* unauthenticated user attempting to access predictions, the system should redirect to the login page.
**Validates: Requirements 7.5**

### Property 8: Error Message Display
*For any* API error response, the system should display a user-friendly error message and not expose technical details.
**Validates: Requirements 1.2, 5.1**

### Property 9: Prediction Paper Format Consistency
*For any* predicted paper displayed, it should use the same format and styling as regular practice papers.
**Validates: Requirements 4.5**

### Property 10: Chart Color Scheme Consistency
*For any* multiple visualizations on the same page, they should use consistent color schemes matching the dark theme palette.
**Validates: Requirements 8.5**

## Error Handling

### Error Categories

#### 1. Network Errors
- **Scenario**: API request fails due to network issues
- **Handling**:
  - Display "Connection error" message
  - Provide retry button
  - Log error for debugging
  - Maintain UI state

#### 2. Permission Errors (403)
- **Scenario**: User doesn't have access to feature
- **Handling**:
  - Display upgrade prompt modal
  - Show feature benefits
  - Provide upgrade link
  - Gracefully disable feature

#### 3. Validation Errors (400)
- **Scenario**: Invalid input parameters
- **Handling**:
  - Highlight invalid fields
  - Display specific error messages
  - Prevent form submission
  - Guide user to correct input

#### 4. Server Errors (500)
- **Scenario**: Backend processing error
- **Handling**:
  - Display generic error message
  - Provide contact support option
  - Log error details
  - Offer alternative actions

#### 5. Data Loading Errors
- **Scenario**: Failed to load prediction data
- **Handling**:
  - Show loading skeleton
  - Display error state
  - Provide reload button
  - Fallback to cached data if available

### Error Display Patterns

```javascript
// Toast notifications for non-critical errors
showToast('Failed to load insights. Please try again.', 'error');

// Modal dialogs for critical errors
showErrorModal({
  title: 'Prediction Failed',
  message: 'Unable to generate prediction. Please try again later.',
  actions: ['Retry', 'Cancel']
});

// Inline error messages for form validation
showFieldError('subject-select', 'Please select a subject');
```

## Testing Strategy

### Unit Testing

#### JavaScript Unit Tests
- **Framework**: Jest
- **Coverage**:
  - API client methods
  - Data transformation functions
  - Chart configuration builders
  - Error handling logic
  - State management functions

**Example Tests**:
```javascript
describe('PredictionAPI', () => {
  test('predictPaper returns valid paper structure', async () => {
    const paper = await api.predictPaper('Physics', 2026);
    expect(paper).toHaveProperty('paper_info');
    expect(paper).toHaveProperty('questions');
    expect(paper.questions).toBeInstanceOf(Array);
  });
  
  test('confidence scores are between 0 and 1', async () => {
    const paper = await api.predictPaper('Physics', 2026);
    expect(paper.paper_info.prediction_confidence).toBeGreaterThanOrEqual(0);
    expect(paper.paper_info.prediction_confidence).toBeLessThanOrEqual(1);
  });
});
```

### Integration Testing

#### Frontend-Backend Integration
- **Test Scenarios**:
  - Complete prediction workflow (select subject → generate → display)
  - Chapter analysis loading and display
  - Smart paper generation with custom parameters
  - Error handling for permission denied
  - Navigation between prediction features

**Example Integration Test**:
```javascript
describe('Prediction Workflow', () => {
  test('user can generate and view predicted paper', async () => {
    // Select subject
    await selectSubject('Physics');
    
    // Select year
    await selectYear(2026);
    
    // Generate prediction
    await clickGenerateButton();
    
    // Verify paper is displayed
    expect(await getPaperTitle()).toContain('Physics');
    expect(await getQuestionCount()).toBe(50);
    expect(await getConfidenceScore()).toBeGreaterThan(0);
  });
});
```

### End-to-End Testing

#### User Journey Tests
- **Framework**: Playwright or Cypress
- **Scenarios**:
  1. User navigates to predictions page
  2. User generates subject-specific prediction
  3. User views chapter analysis
  4. User generates complete NEET paper
  5. User starts predicted test
  6. User views accuracy metrics

### Visual Regression Testing

#### Chart Rendering Tests
- Verify charts render correctly
- Check responsive behavior
- Validate color schemes
- Test animation performance

### Accessibility Testing

#### WCAG Compliance
- Keyboard navigation
- Screen reader compatibility
- Color contrast ratios
- Focus indicators
- ARIA labels

### Performance Testing

#### Metrics to Monitor
- Page load time (<2s)
- API response time (<1s)
- Chart rendering time (<500ms)
- Smooth animations (60fps)

## Implementation Notes

### Technology Stack
- **Frontend**: Vanilla JavaScript (ES6+)
- **Styling**: CSS3 with CSS Variables
- **Charts**: Chart.js v3.x
- **HTTP Client**: Fetch API
- **Build**: No build step (direct browser execution)

### Browser Support
- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Mobile browsers (iOS Safari, Chrome Mobile)

### Performance Optimizations
- Lazy load charts only when visible
- Cache prediction results in sessionStorage
- Debounce API calls
- Use CSS transforms for animations
- Minimize DOM manipulations

### Security Considerations
- All API calls use credentials: 'include'
- CSRF token validation (handled by backend)
- Input sanitization before display
- No sensitive data in localStorage
- Secure error messages (no stack traces)

### Accessibility Features
- Semantic HTML structure
- ARIA labels for interactive elements
- Keyboard navigation support
- Focus management
- Screen reader announcements
- High contrast mode support

### Mobile Responsiveness
- Touch-friendly button sizes (min 44x44px)
- Swipe gestures for navigation
- Responsive charts
- Collapsible sections
- Mobile-optimized modals

### Dark Theme Integration
- Use existing CSS variables
- Consistent color palette
- Proper contrast ratios
- Smooth theme transitions
- Chart colors match theme
