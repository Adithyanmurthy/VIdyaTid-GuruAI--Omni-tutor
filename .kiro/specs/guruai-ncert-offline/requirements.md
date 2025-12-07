# Requirements Document

## Introduction

GuruAI is an offline-first, NCERT-based AI learning companion designed specifically for JEE and NEET aspirants. The system addresses the critical need for an integrated, accessible learning platform that works without requiring API keys or internet connectivity for core functionality. By consolidating NCERT textbooks (including diagrams), 20 years of previous exam papers, and intelligent question generation into a single conversational interface, GuruAI eliminates the fragmented learning experience that hinders student success in competitive exams.

## Glossary

- **System**: The GuruAI application
- **User**: A JEE or NEET student using the application
- **NCERT Content**: National Council of Educational Research and Training textbooks for Physics, Chemistry, Mathematics, and Biology (Classes 11-12, Latest 2024-25 Edition)
- **Vector Store**: A local database storing embedded representations of NCERT content for semantic search
- **RAG System**: Retrieval-Augmented Generation system that retrieves relevant context before generating responses
- **Previous Papers**: Past 20 years of JEE Main, JEE Advanced, and NEET examination questions
- **Diagram**: Visual representations from NCERT textbooks including scientific illustrations, graphs, and chemical structures
- **Question Paper Generator**: Component that creates practice tests based on NCERT content and previous exam patterns
- **Offline Mode**: Operation without internet connectivity using locally stored models and data
- **Embedding Model**: A local machine learning model that converts text into vector representations
- **LLM**: Large Language Model running locally for generating explanations and responses
- **Hinglish**: Code-mixed language combining Hindi and English commonly used by Indian students
- **Voice Input**: Speech-based query input using microphone
- **Text-to-Speech (TTS)**: Technology that converts text responses to spoken audio
- **Handwriting Recognition**: Technology that converts handwritten text and mathematical expressions to digital format
- **Deep Explanation**: Comprehensive, multi-layered explanation covering fundamentals, applications, and advanced aspects of a concept
- **Peer Comparison**: Anonymous performance comparison feature allowing users to see their relative standing
- **Percentile Ranking**: Statistical measure showing the percentage of peers a user has outperformed

## Requirements

### Requirement 1

**User Story:** As a JEE/NEET student, I want to ask questions about NCERT concepts and receive accurate answers based solely on NCERT content, so that I can trust the information aligns with my exam syllabus.

#### Acceptance Criteria

1. WHEN a user submits a query about any Physics, Chemistry, Mathematics, or Biology concept THEN the system SHALL retrieve relevant context from the local NCERT vector store
2. WHEN the system generates a response THEN the system SHALL base the answer exclusively on retrieved NCERT content
3. WHEN the retrieved context contains multiple relevant passages THEN the system SHALL synthesize information from all relevant passages into a coherent explanation
4. WHEN a query matches content from multiple NCERT chapters THEN the system SHALL indicate which chapters the information comes from
5. WHEN the system cannot find relevant NCERT content for a query THEN the system SHALL inform the user that the topic is not covered in NCERT textbooks

### Requirement 2

**User Story:** As a student studying from physical NCERT books, I want to snap a picture of a diagram or problem and get explanations, so that I can understand visual content without typing complex descriptions.

#### Acceptance Criteria

1. WHEN a user uploads an image containing a diagram from an NCERT textbook THEN the system SHALL extract text and visual elements from the image
2. WHEN the system processes an uploaded diagram THEN the system SHALL match it against stored NCERT diagrams to identify the concept
3. WHEN a matching diagram is found THEN the system SHALL retrieve the associated explanation from the NCERT content
4. WHEN a user uploads an image of a problem THEN the system SHALL extract the problem text and provide a step-by-step solution based on NCERT methodology
5. WHEN the image quality is insufficient for processing THEN the system SHALL request the user to upload a clearer image

### Requirement 3

**User Story:** As a student preparing for competitive exams, I want to practice with questions similar to previous JEE/NEET papers, so that I can familiarize myself with exam patterns and difficulty levels.

#### Acceptance Criteria

1. WHEN a user requests practice questions on a specific topic THEN the system SHALL generate questions based on previous 20 years of exam patterns
2. WHEN generating practice questions THEN the system SHALL ensure questions align with NCERT content coverage
3. WHEN a user completes a practice question THEN the system SHALL provide detailed solutions referencing NCERT concepts
4. WHEN a user requests a full-length practice test THEN the system SHALL generate a paper matching the structure and difficulty distribution of actual JEE/NEET exams
5. WHEN displaying previous year questions THEN the system SHALL indicate the year and exam (JEE Main/Advanced/NEET) the question appeared in

### Requirement 4

**User Story:** As a student with limited internet access, I want the system to work completely offline, so that I can study anytime without connectivity constraints.

#### Acceptance Criteria

1. WHEN the system is initialized THEN the system SHALL load all required models and data from local storage
2. WHEN a user interacts with the system without internet connectivity THEN the system SHALL provide full functionality using local resources
3. WHEN the system generates responses THEN the system SHALL use a locally running LLM without external API calls
4. WHEN the system performs semantic search THEN the system SHALL use a local embedding model to encode queries
5. WHEN the system processes images THEN the system SHALL use local OCR and vision models without cloud services

### Requirement 5

**User Story:** As a student studying Biology and Chemistry, I want to see diagrams and visual explanations for complex structures, so that I can better understand spatial relationships and molecular structures.

#### Acceptance Criteria

1. WHEN explaining a concept that has an associated NCERT diagram THEN the system SHALL display the relevant diagram from local storage
2. WHEN a user asks about a biological structure or chemical compound THEN the system SHALL retrieve and display the corresponding NCERT illustration
3. WHEN multiple diagrams are relevant to a concept THEN the system SHALL display all relevant diagrams in sequence
4. WHEN displaying a diagram THEN the system SHALL include the figure caption and page reference from the NCERT textbook
5. WHEN a diagram contains labeled parts THEN the system SHALL explain each labeled component in the response

### Requirement 6

**User Story:** As a student, I want to receive step-by-step solutions for mathematical and physics problems, so that I can understand the problem-solving methodology.

#### Acceptance Criteria

1. WHEN a user asks about a numerical problem THEN the system SHALL provide a solution broken into clear, sequential steps
2. WHEN solving a problem THEN the system SHALL reference the relevant NCERT formulas and concepts used in each step
3. WHEN a problem involves multiple concepts THEN the system SHALL explain how each concept applies to the solution
4. WHEN a problem has multiple solution approaches THEN the system SHALL present the NCERT-recommended method first
5. WHEN displaying mathematical expressions THEN the system SHALL format equations clearly using proper notation

### Requirement 7

**User Story:** As a student, I want to generate custom question papers based on specific chapters or topics, so that I can focus my practice on areas where I need improvement.

#### Acceptance Criteria

1. WHEN a user selects specific chapters or topics THEN the system SHALL generate a question paper covering only those areas
2. WHEN generating a custom paper THEN the system SHALL distribute questions across difficulty levels (easy, medium, hard) based on previous exam patterns
3. WHEN a user specifies the number of questions THEN the system SHALL generate exactly that many questions
4. WHEN generating questions THEN the system SHALL ensure no duplicate questions appear in the same paper
5. WHEN a custom paper is generated THEN the system SHALL provide an answer key with detailed solutions

### Requirement 8

**User Story:** As a student, I want the system to track which topics I've studied and which questions I've attempted, so that I can monitor my preparation progress.

#### Acceptance Criteria

1. WHEN a user completes a practice question THEN the system SHALL record the topic, difficulty, and whether the answer was correct
2. WHEN a user views their progress THEN the system SHALL display statistics for each subject and chapter
3. WHEN a user has attempted questions from a topic THEN the system SHALL show the accuracy percentage for that topic
4. WHEN a user requests recommendations THEN the system SHALL suggest topics based on areas with lower accuracy
5. WHEN progress data is stored THEN the system SHALL persist the data locally without requiring cloud synchronization

### Requirement 9

**User Story:** As a student, I want to search for specific topics or keywords across all NCERT content, so that I can quickly find relevant information without browsing multiple textbooks.

#### Acceptance Criteria

1. WHEN a user enters a search query THEN the system SHALL return all relevant sections from NCERT textbooks ranked by relevance
2. WHEN displaying search results THEN the system SHALL show the subject, class, chapter, and page number for each result
3. WHEN a user selects a search result THEN the system SHALL display the full context including surrounding paragraphs
4. WHEN a search query matches diagram captions THEN the system SHALL include those diagrams in the search results
5. WHEN no exact matches are found THEN the system SHALL suggest related topics using semantic similarity

### Requirement 10

**User Story:** As a student, I want to receive quiz questions after learning a concept, so that I can immediately test my understanding and reinforce learning.

#### Acceptance Criteria

1. WHEN the system completes an explanation THEN the system SHALL generate 2-4 multiple-choice questions on the explained concept
2. WHEN generating quiz questions THEN the system SHALL ensure questions test conceptual understanding rather than rote memorization
3. WHEN a user answers a quiz question THEN the system SHALL provide immediate feedback indicating correctness
4. WHEN a user answers incorrectly THEN the system SHALL explain why the answer is wrong and provide the correct reasoning
5. WHEN quiz questions are generated THEN the system SHALL vary difficulty based on the complexity of the explained concept

### Requirement 11

**User Story:** As a student with limited computing resources, I want the system to manage AI model loading efficiently, so that I can run the application on my personal computer without performance issues.

#### Acceptance Criteria

1. WHEN the application starts THEN the system SHALL load the local LLM into memory only when first needed
2. WHEN the system is idle for a configurable period THEN the system SHALL unload the LLM from memory to free resources
3. WHEN a user submits a query after the LLM has been unloaded THEN the system SHALL reload the model and display a loading indicator
4. WHEN the system loads or unloads models THEN the system SHALL display the current status to the user
5. WHEN the user configures system settings THEN the system SHALL allow adjustment of memory usage limits and model persistence duration

### Requirement 12

**User Story:** As a student sharing the GuruAI system with other users, I want to have my own account with personalized progress tracking, so that my study data remains private and my learning experience is tailored to my needs.

#### Acceptance Criteria

1. WHEN a new user accesses the system THEN the system SHALL provide options to create an account or log in
2. WHEN a user creates an account THEN the system SHALL store user credentials securely using local encryption
3. WHEN a user logs in THEN the system SHALL load that user's personal progress data, bookmarks, and preferences
4. WHEN multiple users are using the system THEN the system SHALL keep each user's progress data isolated and private
5. WHEN a user logs out THEN the system SHALL clear the session and require authentication for the next access

### Requirement 13

**User Story:** As a product distributor, I want all NCERT content and AI models bundled with the installation package, so that students can use the system immediately after installation without additional downloads.

#### Acceptance Criteria

1. WHEN the installation package is created THEN the system SHALL include all NCERT textbooks for Classes 11 and 12 (Latest 2024-25 Edition) in pre-processed vector format
2. WHEN the installation package is created THEN the system SHALL include all NCERT diagrams and illustrations as indexed image files
3. WHEN the installation package is created THEN the system SHALL include 20 years of JEE and NEET previous papers with solutions
4. WHEN the installation package is created THEN the system SHALL include all required AI models for offline operation
5. WHEN a student installs the application THEN the system SHALL verify all content is properly installed and display installation status

### Requirement 14

**User Story:** As a student who prefers speaking over typing, I want to ask questions using voice input in English or Hinglish, so that I can interact with the system more naturally and quickly.

#### Acceptance Criteria

1. WHEN a user activates voice input THEN the system SHALL capture audio using the device microphone
2. WHEN the system receives voice input THEN the system SHALL transcribe the audio to text using local speech recognition
3. WHEN transcribing voice input THEN the system SHALL support both English and Hinglish (Hindi-English code-mixed) speech
4. WHEN transcription is complete THEN the system SHALL process the transcribed text as a query
5. WHEN the system generates a response THEN the system SHALL provide an option to read the response aloud using text-to-speech

### Requirement 15

**User Story:** As a student who prefers auditory learning, I want to hear explanations spoken aloud, so that I can learn while doing other activities or when reading is inconvenient.

#### Acceptance Criteria

1. WHEN a user requests voice output THEN the system SHALL convert the text response to speech using local text-to-speech
2. WHEN generating voice output THEN the system SHALL support both English and Hinglish pronunciation
3. WHEN speaking a response THEN the system SHALL provide playback controls (play, pause, stop, speed adjustment)
4. WHEN voice output is active THEN the system SHALL highlight the currently spoken text in the interface
5. WHEN the user configures voice settings THEN the system SHALL allow selection of voice type, speed, and language preference

### Requirement 16

**User Story:** As a student who writes notes by hand, I want to upload handwritten problems or notes and get them recognized, so that I can get help with my handwritten work without retyping.

#### Acceptance Criteria

1. WHEN a user uploads an image containing handwritten text THEN the system SHALL extract the handwritten content using handwriting recognition
2. WHEN processing handwritten content THEN the system SHALL recognize both English and Hindi Devanagari script
3. WHEN handwritten mathematical expressions are detected THEN the system SHALL convert them to digital format
4. WHEN handwriting recognition is uncertain THEN the system SHALL highlight ambiguous characters and request user confirmation
5. WHEN handwritten content is successfully recognized THEN the system SHALL process it as a query or problem to solve

### Requirement 17

**User Story:** As a student studying complex topics, I want to access deep, comprehensive explanations with multiple examples and analogies, so that I can develop thorough understanding of difficult NCERT concepts.

#### Acceptance Criteria

1. WHEN a user requests a deep explanation for a concept THEN the system SHALL provide a multi-layered explanation covering fundamentals, applications, and advanced aspects
2. WHEN generating deep explanations THEN the system SHALL include multiple examples from NCERT textbooks demonstrating the concept
3. WHEN explaining complex topics THEN the system SHALL provide analogies and real-world connections to aid understanding
4. WHEN a deep explanation involves prerequisites THEN the system SHALL identify and explain prerequisite concepts first
5. WHEN displaying deep explanations THEN the system SHALL organize content into progressive sections (basic, intermediate, advanced) based on NCERT content

### Requirement 18

**User Story:** As a competitive student, I want to compare my performance anonymously with other users, so that I can understand where I stand and stay motivated without revealing my identity.

#### Acceptance Criteria

1. WHEN a user opts into peer comparison THEN the system SHALL anonymize the user's identity using a randomly generated identifier
2. WHEN displaying peer comparisons THEN the system SHALL show percentile rankings for each subject and overall performance
3. WHEN comparing performance THEN the system SHALL display statistics such as average accuracy, topics mastered, and questions attempted relative to peers
4. WHEN a user views peer comparison THEN the system SHALL ensure no personally identifiable information is visible
5. WHEN peer comparison data is stored THEN the system SHALL keep individual performance data private and only share aggregated anonymous statistics
