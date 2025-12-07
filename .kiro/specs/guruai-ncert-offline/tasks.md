# Implementation Plan

- [x] 1. Set up project structure and development environment

  - Create directory structure for backend (models, services, utils, routes)
  - Create directory structure for frontend (static/css, static/js, templates)
  - Set up Python virtual environment
  - Create requirements.txt with all dependencies
  - Initialize Git repository with .gitignore
  - _Requirements: 4.1, 13.4_

- [x] 2. Implement data models and database schema




  - Create SQLite database schema for User, Progress, Session models
  - Implement User model with password hashing (bcrypt)
  - Implement Progress model with subject/chapter tracking
  - Implement Session model for authentication
  - Create database migration scripts
  - _Requirements: 12.2, 8.1, 12.5_

- [ ]* 2.1 Write property test for User model
  - **Property 40: Password Encryption**
  - **Validates: Requirements 12.2**

- [ ]* 2.2 Write property test for Progress model
  - **Property 26: Accuracy Calculation Correctness**
  - **Validates: Requirements 8.3**

- [x] 3. Set up NCERT content processing pipeline


  - Create scripts to extract text from NCERT PDFs
  - Implement text chunking strategy (500-word chunks with overlap)
  - Set up sentence-transformers embedding model
  - Generate embeddings for all NCERT content
  - Initialize ChromaDB vector store with embeddings
  - _Requirements: 1.1, 13.1_

- [x] 4. Implement diagram indexing system

  - Extract diagrams from NCERT PDFs
  - Create diagram metadata (subject, chapter, figure number, caption)
  - Generate visual embeddings for diagrams
  - Store diagrams in organized directory structure
  - Create diagram index database
  - _Requirements: 5.1, 13.2_

- [x] 5. Set up local AI model infrastructure

  - Download and configure Llama 3.2 3B model (or Phi-3 Mini)
  - Set up llama.cpp for local inference
  - Configure model quantization (4-bit or 8-bit)
  - Test model loading and inference
  - _Requirements: 4.3, 13.4_

- [x] 6. Implement Model Manager with dynamic loading




  - Create ModelManager class with load/unload methods
  - Implement idle timeout mechanism (configurable)
  - Add memory monitoring and status reporting
  - Implement loading indicator state management
  - Create background task for idle checking
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ]* 6.1 Write property test for Model Manager
  - **Property 35: Model Lazy Loading**
  - **Property 36: Model Idle Unloading**
  - **Validates: Requirements 11.1, 11.2**

- [x] 7. Implement RAG System for context retrieval


  - Create RAGSystem class with embedding and retrieval methods
  - Implement semantic search using ChromaDB
  - Add result reranking by relevance
  - Implement multi-chapter detection and reference extraction
  - Handle out-of-scope queries gracefully
  - _Requirements: 1.1, 1.2, 1.4, 1.5_

- [ ]* 7.1 Write property test for RAG System
  - **Property 1: Context Retrieval Completeness**
  - **Property 3: Multi-Chapter Reference Inclusion**
  - **Validates: Requirements 1.1, 1.4**

- [x] 8. Implement Query Handler for text-based queries



  - Create QueryHandler class integrating RAG and LLM
  - Implement prompt engineering for NCERT-grounded responses
  - Add diagram retrieval based on query context
  - Implement response formatting with references
  - Add error handling for failed retrievals
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ]* 8.1 Write property test for Query Handler
  - **Property 2: Response Grounding in NCERT Content**
  - **Property 13: Diagram Display for Relevant Concepts**
  - **Validates: Requirements 1.2, 5.1_

- [x] 9. Implement adaptive quiz generation



  - Create quiz generation logic based on explained concepts
  - Implement MCQ format with 4 options
  - Generate 2-4 questions per explanation
  - Create answer validation and feedback system
  - Add incorrect answer explanations
  - _Requirements: 10.1, 10.3, 10.4_

- [ ]* 9.1 Write property test for quiz generation
  - **Property 32: Quiz Generation Count**
  - **Property 33: Quiz Feedback Provision**
  - **Validates: Requirements 10.1, 10.3**

- [x] 10. Implement image processing pipeline



  - Set up Tesseract and EasyOCR for text extraction
  - Create ImageProcessor class with OCR methods
  - Implement image preprocessing (resize, denoise, contrast)
  - Add diagram matching against stored NCERT diagrams
  - Implement content type classification (problem vs diagram)
  - _Requirements: 2.1, 2.2, 4.5_

- [ ]* 10.1 Write property test for image processing
  - **Property 4: OCR Text Extraction Success**
  - **Property 5: Diagram Matching Accuracy**
  - **Validates: Requirements 2.1, 2.2**

- [ ] 11. Implement problem solving from images




  - Extract problem text from uploaded images
  - Generate step-by-step solutions using LLM
  - Format solutions with proper mathematical notation
  - Add NCERT formula references to each step
  - Handle poor quality images with user feedback
  - _Requirements: 2.4, 2.5, 6.1, 6.2_

- [ ]* 11.1 Write property test for problem solving
  - **Property 16: Solution Step Structure**
  - **Property 17: Formula Reference in Solutions**
  - **Validates: Requirements 6.1, 6.2**
- [x] 12. Set up previous papers database





- [ ] 12. Set up previous papers database

  - Create database schema for Question model
  - Import 20 years of JEE Main/Advanced/NEET questions
  - Add metadata (year, exam, subject, chapter, difficulty)
  - Create solutions for all questions with NCERT references
  - Index questions for efficient retrieval
  - _Requirements: 3.1, 3.5, 13.3_

- [x] 13. Implement question paper generator




  - Create QuestionGenerator class
  - Implement topic-based question selection
  - Add difficulty distribution logic
  - Ensure no duplicate questions in papers
  - Generate answer keys with detailed solutions
  - Implement full-length test generation matching exam structure
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 3.4_

- [x]* 13.1 Write property test for question generator

  - **Property 19: Question Paper Topic Coverage**
  - **Property 20: Question Paper Difficulty Distribution**
  - **Property 21: Question Paper Count Accuracy**
  - **Property 22: Question Paper Uniqueness**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [ ] 14. Implement progress tracking system



  - Create progress recording logic for attempted questions
  - Implement accuracy calculation per topic/chapter
  - Add statistics aggregation by subject
  - Create recommendation engine based on weak areas
  - Implement local data persistence
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ]* 14.1 Write property test for progress tracking
  - **Property 24: Progress Record Completeness**
  - **Property 27: Recommendation Prioritization**
  - **Property 28: Progress Data Persistence**
  - **Validates: Requirements 8.1, 8.4, 8.5**

- [x] 15. Implement search functionality

  - Create semantic search across NCERT content
  - Add metadata extraction (subject, class, chapter, page)
  - Implement result ranking by relevance
  - Add context display with surrounding paragraphs
  - Include diagrams in search results when captions match
  - Handle no-results with semantic suggestions
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ]* 15.1 Write property test for search
  - **Property 29: Search Result Metadata Completeness**
  - **Property 31: Diagram Search Inclusion**
  - **Validates: Requirements 9.2, 9.4**

- [x] 16. Implement user authentication system



  - Create user registration endpoint with password hashing
  - Implement login endpoint with session creation
  - Add logout endpoint with session clearing
  - Implement session validation middleware
  - Add account lockout after failed attempts
  - _Requirements: 12.1, 12.2, 12.5_

- [ ]* 16.1 Write property test for authentication
  - **Property 41: User Data Loading**
  - **Property 42: User Data Isolation**
  - **Property 43: Session Clearing on Logout**
  - **Validates: Requirements 12.3, 12.4, 12.5**

- [x] 17. Build Flask backend API endpoints



  - Create /api/ask endpoint for text queries
  - Create /api/solve-image endpoint for image uploads
  - Create /api/generate-paper endpoint for question papers
  - Create /api/previous-papers endpoint for past papers
  - Create /api/progress endpoints (GET and POST)
  - Create /api/search endpoint
  - Create /api/auth endpoints (register, login, logout)
  - Add CORS configuration
  - _Requirements: All API-related requirements_

- [ ]* 17.1 Write integration tests for API endpoints
  - Test complete request-response cycles
  - Test error handling for invalid inputs
  - Test authentication flow
  - Test offline operation (no external calls)
-

- [x] 18. Implement frontend HTML structure




  - Create main index.html with chat interface
  - Add progress dashboard page
  - Add search interface page
  - Add question paper generator page
  - Add login/registration forms
  - Add settings page for configuration
  - _Requirements: UI-related requirements_

- [x] 19. Implement frontend CSS styling





  - Create responsive layout for all screen sizes
  - Style chat interface with user/bot message distinction
  - Add loading indicators and animations
  - Style progress charts and statistics
  - Create dark theme with modern design
  - Add mobile-friendly responsive design
  - _Requirements: UI-related requirements_

- [x] 20. Implement frontend JavaScript functionality





  - Create chat interface logic (send query, display messages)
  - Implement image upload and preview
  - Add quiz interaction (answer selection, feedback)
  - Create progress dashboard with charts
  - Implement search interface with filters
  - Add question paper generator UI logic
  - Implement authentication flow (login, logout)
  - Add settings management
  - _Requirements: All frontend interaction requirements_

- [ ]* 20.1 Write unit tests for JavaScript functions
  - Test message display functions
  - Test API call functions
  - Test quiz answer validation
  - Test form validation

- [x] 21. Implement diagram display system




  - Create diagram retrieval from local storage
  - Add diagram rendering in responses
  - Display figure captions and page references
  - Implement labeled part explanations
  - Add diagram zoom and pan functionality
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 21.1 Write property test for diagram display
  - **Property 14: Diagram Metadata Completeness**
  - **Property 15: Labeled Diagram Explanation**
  - **Validates: Requirements 5.4, 5.5**
-

- [x] 22. Implement mathematical expression formatting


  - Add LaTeX rendering support (MathJax or KaTeX)
  - Format equations in responses
  - Handle inline and display math modes
  - Test rendering of complex expressions
  - _Requirements: 6.5_

- [ ]* 22.1 Write property test for math formatting
  - **Property 18: Mathematical Expression Formatting**
  - **Validates: Requirements 6.5**
-

- [x] 23. Implement settings and configuration



  - Create settings page UI
  - Add memory limit configuration
  - Add idle timeout configuration
  - Implement settings persistence
  - Add settings validation
  - _Requirements: 11.5_

- [ ]* 23.1 Write property test for settings
  - **Property 39: Settings Configurability**
  - **Validates: Requirements 11.5**

- [x] 24. Implement error handling and logging



  - Add comprehensive error handling to all endpoints
  - Create error response format
  - Implement logging system with rotation
  - Add user-friendly error messages
  - Create error recovery mechanisms
  - _Requirements: Error handling requirements_

- [x] 25. Implement offline verification system


  - Add network call monitoring
  - Verify no external API calls during operation
  - Test all features without internet
  - Add offline status indicator
  - _Requirements: 4.2, 4.3, 4.4, 4.5_

- [ ]* 25.1 Write property test for offline operation
  - **Property 12: Offline Operation Completeness**
  - **Validates: Requirements 4.2, 4.3, 4.4, 4.5**

- [ ] 26. Create installation package
  - Bundle all NCERT content (2024-25 edition)
  - Bundle all diagrams and illustrations
  - Bundle 20 years of previous papers
  - Bundle AI models (LLM, embeddings, OCR)
  - Create installation scripts (Windows, Linux, macOS)
  - Add installation verification
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 27. Implement installation verification
  - Verify all content files present
  - Verify model checksums
  - Test database initialization
  - Test model loading
  - Display installation status
  - _Requirements: 13.5_

- [ ] 28. Performance optimization
  - Optimize vector search with ANN algorithms
  - Implement caching for frequent queries
  - Optimize image processing pipeline
  - Add batch processing where applicable
  - Profile and optimize slow operations
  - _Requirements: Performance requirements_

- [ ] 29. Security hardening
  - Implement input validation for all endpoints
  - Add file upload validation (size, type, content)
  - Implement rate limiting
  - Add SQL injection prevention
  - Test authentication security
  - _Requirements: Security requirements_

- [ ] 30. Documentation and README
  - Update README with installation instructions
  - Create user guide
  - Document API endpoints
  - Add troubleshooting guide
  - Create developer documentation
  - _Requirements: Documentation requirements_

- [ ] 31. Final checkpoint - Ensure all tests pass
  - Run all unit tests
  - Run all property-based tests
  - Run all integration tests
  - Verify offline operation
  - Test on different operating systems
  - Ask the user if questions arise
