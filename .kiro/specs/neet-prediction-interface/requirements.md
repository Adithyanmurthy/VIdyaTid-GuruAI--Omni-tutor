# Requirements Document

## Introduction

This feature adds a comprehensive frontend interface for the existing NEET question prediction backend system. The backend already provides prediction APIs that analyze previous year patterns, predict high-probability chapters, and generate smart practice papers. This feature will create user-facing interfaces to access these capabilities, visualize prediction insights, track accuracy metrics, and integrate predictions into the existing question paper generation workflow.

## Glossary

- **Prediction System**: The backend service that analyzes historical NEET papers and predicts future question patterns
- **Smart Paper**: A practice paper generated based on user's weak areas and predicted high-probability topics
- **Confidence Score**: A numerical value (0-100) indicating the prediction system's confidence in a particular prediction
- **Chapter Analysis**: Subject-wise breakdown showing prediction probability for each chapter
- **Prediction Accuracy**: Historical metric comparing past predictions with actual NEET papers
- **High Probability Topics**: Chapters or topics predicted to have high likelihood of appearing in upcoming exams
- **User**: A student using the GuruAI platform to prepare for NEET exams
- **Frontend Interface**: The web-based user interface components (HTML, CSS, JavaScript)

## Requirements

### Requirement 1

**User Story:** As a NEET student, I want to access AI-predicted practice papers, so that I can focus my preparation on high-probability topics.

#### Acceptance Criteria

1. WHEN a user navigates to the predictions page THEN the system SHALL display a prediction interface with options for different prediction types
2. WHEN a user requests a predicted paper THEN the system SHALL call the backend prediction API and display the generated paper
3. WHEN a predicted paper is generated THEN the system SHALL show confidence scores for each predicted topic
4. WHEN a user views predictions THEN the system SHALL display the NEET exam pattern (Physics: 50, Chemistry: 50, Biology: 100 questions)
5. WHERE the user has completed previous practice tests THEN the system SHALL generate smart papers based on weak areas

### Requirement 2

**User Story:** As a NEET student, I want to view chapter-wise prediction analysis, so that I can understand which chapters are most likely to appear in the exam.

#### Acceptance Criteria

1. WHEN a user selects a subject THEN the system SHALL display chapter-wise prediction probabilities for that subject
2. WHEN displaying chapter analysis THEN the system SHALL show prediction confidence scores as visual indicators (progress bars or charts)
3. WHEN a user views chapter analysis THEN the system SHALL organize chapters by prediction probability (highest to lowest)
4. WHEN chapter data is displayed THEN the system SHALL include class 11 and class 12 weightage information
5. WHEN a user clicks on a high-probability chapter THEN the system SHALL provide option to generate practice questions from that chapter

### Requirement 3

**User Story:** As a NEET student, I want to see prediction accuracy metrics, so that I can trust the reliability of the prediction system.

#### Acceptance Criteria

1. WHEN a user views the predictions page THEN the system SHALL display historical accuracy metrics comparing predictions with actual papers
2. WHEN displaying accuracy metrics THEN the system SHALL show subject-wise accuracy percentages
3. WHEN accuracy data is available THEN the system SHALL visualize accuracy trends over multiple years
4. WHEN a user views accuracy information THEN the system SHALL display the number of past papers analyzed
5. IF no accuracy data exists THEN the system SHALL display a message indicating predictions are based on pattern analysis

### Requirement 4

**User Story:** As a NEET student, I want the question paper page to show AI-predicted papers, so that I can easily access prediction-based practice without navigating to a separate page.

#### Acceptance Criteria

1. WHEN a user visits the question paper page THEN the system SHALL display an "AI Predicted Papers" section
2. WHEN the AI Predicted Papers section is displayed THEN the system SHALL show available predicted papers for upcoming exam years
3. WHEN a user selects an AI predicted paper THEN the system SHALL generate and display the paper using the prediction API
4. WHEN displaying predicted papers THEN the system SHALL indicate prediction confidence level
5. WHEN a user generates a predicted paper THEN the system SHALL maintain the same paper format as regular practice papers

### Requirement 5

**User Story:** As a NEET student, I want to view prediction insights for each subject, so that I can understand the reasoning behind predictions.

#### Acceptance Criteria

1. WHEN a user requests prediction insights for a subject THEN the system SHALL call the insights API endpoint
2. WHEN displaying insights THEN the system SHALL show topic frequency analysis from previous years
3. WHEN insights are displayed THEN the system SHALL highlight topics with increasing frequency trends
4. WHEN a user views insights THEN the system SHALL show difficulty distribution (Easy/Medium/Hard) for predicted topics
5. WHEN insights include multiple data points THEN the system SHALL present information using charts or visual representations

### Requirement 6

**User Story:** As a NEET student, I want to generate a complete predicted NEET paper for a specific year, so that I can simulate the actual exam experience with predicted questions.

#### Acceptance Criteria

1. WHEN a user selects a target year for prediction THEN the system SHALL generate a complete 200-question NEET paper
2. WHEN generating a complete paper THEN the system SHALL follow the exact NEET pattern (180 questions to attempt, 20 optional)
3. WHEN a complete paper is generated THEN the system SHALL include all three subjects with correct distribution
4. WHEN displaying the complete paper THEN the system SHALL show overall prediction confidence score
5. WHEN a user starts the predicted paper THEN the system SHALL provide the same test-taking interface as regular papers

### Requirement 7

**User Story:** As a NEET student, I want a dedicated predictions page with navigation access, so that I can easily find and use prediction features.

#### Acceptance Criteria

1. WHEN the predictions page is created THEN the system SHALL add a navigation link in the main menu
2. WHEN a user clicks the predictions navigation link THEN the system SHALL load the predictions page
3. WHEN the predictions page loads THEN the system SHALL display all available prediction features in organized sections
4. WHEN displaying prediction features THEN the system SHALL use the same dark theme styling as other pages
5. WHEN a user is not authenticated THEN the system SHALL redirect to login before accessing predictions

### Requirement 8

**User Story:** As a NEET student, I want responsive and visually appealing prediction visualizations, so that I can easily understand complex prediction data.

#### Acceptance Criteria

1. WHEN prediction data is displayed THEN the system SHALL use charts and graphs for numerical data
2. WHEN displaying confidence scores THEN the system SHALL use color-coded indicators (green for high, yellow for medium, red for low)
3. WHEN showing chapter probabilities THEN the system SHALL use progress bars or similar visual elements
4. WHEN the page is viewed on mobile devices THEN the system SHALL adapt visualizations to smaller screens
5. WHEN multiple data visualizations are present THEN the system SHALL maintain consistent styling and color schemes
