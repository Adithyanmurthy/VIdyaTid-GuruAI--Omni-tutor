// Question Paper Generator - Create and manage practice tests
document.addEventListener('DOMContentLoaded', () => {
    let currentPaper = null;
    let currentTest = null;
    let testStartTime = null;
    let timerInterval = null;
    let currentQuestionIndex = 0;
    let userAnswers = {};

    // Initialize
    function initialize() {
        setupEventListeners();
        loadPreviousYears();
    }

    initialize();

    // Setup event listeners
    function setupEventListeners() {
        // Paper type selection
        document.querySelectorAll('.select-type-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const card = e.target.closest('.paper-type-card');
                const type = card.dataset.type;
                showConfigSection(type);
            });
        });

        // AI Prediction buttons
        document.getElementById('generate-subject-prediction-btn')?.addEventListener('click', generateSubjectPrediction);
        document.getElementById('generate-complete-prediction-btn')?.addEventListener('click', generateCompletePrediction);
        document.getElementById('cancel-prediction-btn')?.addEventListener('click', () => hideAllConfigs());

        // Custom paper configuration
        document.getElementById('generate-paper-btn')?.addEventListener('click', generateCustomPaper);
        document.getElementById('cancel-config-btn')?.addEventListener('click', () => hideAllConfigs());

        // Full-length test configuration
        document.getElementById('generate-full-test-btn')?.addEventListener('click', generateFullTest);
        document.getElementById('cancel-full-test-btn')?.addEventListener('click', () => hideAllConfigs());

        // Previous papers
        document.getElementById('cancel-previous-btn')?.addEventListener('click', () => hideAllConfigs());

        // Difficulty sliders
        setupDifficultySliders();

        // Subject checkboxes
        document.querySelectorAll('input[name="subject"]').forEach(checkbox => {
            checkbox.addEventListener('change', updateChapterSelection);
        });

        // Paper actions
        document.getElementById('download-paper-btn')?.addEventListener('click', downloadPaper);
        document.getElementById('print-paper-btn')?.addEventListener('click', printPaper);
        document.getElementById('start-test-btn')?.addEventListener('click', startTest);
        document.getElementById('new-paper-btn')?.addEventListener('click', () => {
            hideAllConfigs();
            document.getElementById('generated-paper').classList.add('hidden');
        });

        // Test controls
        document.getElementById('prev-question-btn')?.addEventListener('click', () => navigateQuestion(-1));
        document.getElementById('next-question-btn')?.addEventListener('click', () => navigateQuestion(1));
        document.getElementById('mark-review-btn')?.addEventListener('click', markForReview);
        document.getElementById('clear-response-btn')?.addEventListener('click', clearResponse);
        document.getElementById('submit-test-btn')?.addEventListener('click', submitTest);

        // Results actions
        document.getElementById('view-solutions-btn')?.addEventListener('click', viewSolutions);
        document.getElementById('retry-test-btn')?.addEventListener('click', retryTest);
        document.getElementById('back-to-generator-btn')?.addEventListener('click', backToGenerator);
    }

    // Show configuration section
    function showConfigSection(type) {
        hideAllConfigs();
        
        if (type === 'custom') {
            document.getElementById('custom-config').classList.remove('hidden');
            updateChapterSelection();
        } else if (type === 'full-length') {
            document.getElementById('full-length-config').classList.remove('hidden');
        } else if (type === 'previous') {
            document.getElementById('previous-papers-browser').classList.remove('hidden');
            loadPreviousPapers();
        } else if (type === 'ai-prediction') {
            showPredictionConfig();
        }
    }

    // Hide all config sections
    function hideAllConfigs() {
        document.getElementById('custom-config')?.classList.add('hidden');
        document.getElementById('full-length-config')?.classList.add('hidden');
        document.getElementById('previous-papers-browser')?.classList.add('hidden');
        document.getElementById('ai-prediction-config')?.classList.add('hidden');
    }

    // Show prediction configuration
    function showPredictionConfig() {
        hideAllConfigs();
        document.getElementById('ai-prediction-config')?.classList.remove('hidden');
        loadPredictionInsights();
    }

    // Load prediction insights when config is shown
    async function loadPredictionInsights() {
        const insightsGrid = document.getElementById('insights-grid');
        if (!insightsGrid) return;

        try {
            // Get the selected subject (default to Physics)
            const subject = document.getElementById('prediction-subject')?.value || 'Physics';
            
            // Call prediction API for insights
            const response = await fetch(`/api/prediction/insights/${subject}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to load insights');
            }

            const data = await response.json();
            displayPredictionInsights(data, insightsGrid);

        } catch (error) {
            console.error('Error loading prediction insights:', error);
            insightsGrid.innerHTML = `
                <div class="insight-card error">
                    <h4>Unable to load insights</h4>
                    <p>Please try again later</p>
                </div>
            `;
        }
    }

    // Display prediction insights
    function displayPredictionInsights(data, container) {
        const highProbChapters = data.high_probability_chapters || [];
        const recommendedFocus = data.recommended_focus || [];
        const dataConfidence = data.data_confidence || 0;

        container.innerHTML = `
            <div class="insight-card">
                <h4>📚 High Probability Chapters</h4>
                <div class="chapter-tags">
                    ${highProbChapters.slice(0, 5).map(ch => `<span class="chapter-tag">${ch}</span>`).join('')}
                </div>
            </div>
            <div class="insight-card">
                <h4>🎯 Accuracy Track Record</h4>
                <p>Our predictions: ${Math.round(dataConfidence * 100)}% accurate</p>
                <p class="insight-note">Based on ${data.total_questions_analyzed || 0} questions analyzed</p>
            </div>
            <div class="insight-card">
                <h4>⭐ Recommended Focus</h4>
                <div class="focus-list">
                    ${recommendedFocus.slice(0, 3).map(topic => `<div class="focus-item">• ${topic}</div>`).join('')}
                </div>
            </div>
        `;
    }

    // Update insights when subject changes
    document.getElementById('prediction-subject')?.addEventListener('change', loadPredictionInsights);

    // Setup difficulty sliders
    function setupDifficultySliders() {
        const easySlider = document.getElementById('easy-slider');
        const mediumSlider = document.getElementById('medium-slider');
        const hardSlider = document.getElementById('hard-slider');

        [easySlider, mediumSlider, hardSlider].forEach(slider => {
            slider?.addEventListener('input', () => {
                updateDifficultyValues();
                normalizeDifficulty();
            });
        });
    }

    // Update difficulty values
    function updateDifficultyValues() {
        const easy = parseInt(document.getElementById('easy-slider').value);
        const medium = parseInt(document.getElementById('medium-slider').value);
        const hard = parseInt(document.getElementById('hard-slider').value);

        document.getElementById('easy-value').textContent = easy;
        document.getElementById('medium-value').textContent = medium;
        document.getElementById('hard-value').textContent = hard;
        document.getElementById('total-percentage').textContent = easy + medium + hard;
    }

    // Normalize difficulty to 100%
    function normalizeDifficulty() {
        const easy = parseInt(document.getElementById('easy-slider').value);
        const medium = parseInt(document.getElementById('medium-slider').value);
        const hard = parseInt(document.getElementById('hard-slider').value);
        const total = easy + medium + hard;

        if (total !== 100) {
            const factor = 100 / total;
            document.getElementById('easy-slider').value = Math.round(easy * factor);
            document.getElementById('medium-slider').value = Math.round(medium * factor);
            document.getElementById('hard-slider').value = Math.round(hard * factor);
            updateDifficultyValues();
        }
    }

    // Update chapter selection based on selected subjects
    async function updateChapterSelection() {
        const selectedSubjects = Array.from(document.querySelectorAll('input[name="subject"]:checked'))
            .map(cb => cb.value);

        const chapterSelection = document.getElementById('chapter-selection');
        chapterSelection.innerHTML = '';

        if (selectedSubjects.length === 0) {
            chapterSelection.innerHTML = '<p>Please select at least one subject</p>';
            return;
        }

        for (const subject of selectedSubjects) {
            const subjectDiv = document.createElement('div');
            subjectDiv.className = 'subject-chapters';
            subjectDiv.innerHTML = `<h4>${subject}</h4>`;

            try {
                const response = await fetch(`/api/chapters?subject=${subject}`, {
                    credentials: 'include'
                });
                const data = await response.json();
                const chapters = data.chapters || [];

                const chaptersGrid = document.createElement('div');
                chaptersGrid.className = 'chapters-grid';

                chapters.forEach(chapter => {
                    const label = document.createElement('label');
                    label.className = 'checkbox-label';
                    label.innerHTML = `
                        <input type="checkbox" name="chapter" value="${subject}:${chapter.number}" checked>
                        <span>Ch ${chapter.number}</span>
                    `;
                    chaptersGrid.appendChild(label);
                });

                subjectDiv.appendChild(chaptersGrid);
            } catch (error) {
                console.error(`Error loading chapters for ${subject}:`, error);
            }

            chapterSelection.appendChild(subjectDiv);
        }
    }

    // Generate custom paper
    async function generateCustomPaper() {
        const selectedChapters = Array.from(document.querySelectorAll('input[name="chapter"]:checked'))
            .map(cb => cb.value);

        if (selectedChapters.length === 0) {
            alert('Please select at least one chapter');
            return;
        }

        const config = {
            type: 'custom',
            chapters: selectedChapters,
            difficulty: {
                easy: parseInt(document.getElementById('easy-slider').value),
                medium: parseInt(document.getElementById('medium-slider').value),
                hard: parseInt(document.getElementById('hard-slider').value)
            },
            count: parseInt(document.getElementById('question-count').value)
        };

        await generatePaper(config);
    }

    // Generate full-length test
    async function generateFullTest() {
        const examType = document.querySelector('input[name="exam-type"]:checked')?.value;

        if (!examType) {
            alert('Please select an exam type');
            return;
        }

        const config = {
            type: 'full-length',
            exam_type: examType
        };

        await generatePaper(config);
    }

    // Generate paper (common function)
    async function generatePaper(config) {
        showLoading(true);

        try {
            const response = await fetch('/api/generate-paper', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config),
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to generate paper');
            }

            const data = await response.json();
            
            // Handle response format - API returns {success: true, paper: {...}}
            const paper = data.paper || data;
            
            // Ensure paper has required fields
            if (!paper.questions) {
                paper.questions = [];
            }
            if (!paper.title) {
                paper.title = 'Question Paper';
            }
            
            currentPaper = paper;
            displayGeneratedPaper(paper);

        } catch (error) {
            console.error('Error generating paper:', error);
            alert('Failed to generate question paper. Please try again.');
        } finally {
            showLoading(false);
        }
    }

    // Display generated paper
    function displayGeneratedPaper(paper) {
        hideAllConfigs();
        
        const generatedPaper = document.getElementById('generated-paper');
        const paperTitle = document.getElementById('paper-title');
        const paperContent = document.getElementById('paper-content');

        // Check if this is a predicted paper
        const isPredicted = paper.paper_info && paper.paper_info.exam_type && 
                           paper.paper_info.exam_type.includes('PREDICTED');
        
        // Set title with prediction indicator
        let titleText = paper.title || paper.paper_info?.exam_type || 'Question Paper';
        if (isPredicted && paper.paper_info) {
            const confidence = paper.paper_info.prediction_confidence || 0;
            const confidencePercent = Math.round(confidence * 100);
            titleText = `🤖 ${paper.paper_info.subject || 'NEET'} ${paper.paper_info.year || ''} - AI Predicted Paper`;
            
            // Add confidence indicator to title
            const confidenceBadge = getConfidenceBadge(confidence);
            paperTitle.innerHTML = `${titleText} ${confidenceBadge}`;
        } else {
            paperTitle.textContent = titleText;
        }

        paperContent.innerHTML = '';

        // Add prediction info header if it's a predicted paper
        if (isPredicted && paper.paper_info) {
            const infoHeader = document.createElement('div');
            infoHeader.className = 'prediction-info-header';
            infoHeader.innerHTML = `
                <div class="prediction-stats-row">
                    <div class="stat-item">
                        <span class="stat-label">Questions:</span>
                        <span class="stat-value">${paper.paper_info.question_count || paper.questions.length}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Duration:</span>
                        <span class="stat-value">${paper.paper_info.duration_minutes || 180} min</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Confidence:</span>
                        <span class="stat-value">${Math.round((paper.paper_info.prediction_confidence || 0) * 100)}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Based on:</span>
                        <span class="stat-value">${paper.paper_info.based_on_years || 5} years</span>
                    </div>
                </div>
            `;
            paperContent.appendChild(infoHeader);
        }

        // Get questions array (handle both formats)
        const questions = paper.questions || [];

        questions.forEach((question, index) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'question-item';
            
            let optionsHTML = '';
            if (question.options) {
                optionsHTML = '<div class="question-options">';
                question.options.forEach((option, oIndex) => {
                    optionsHTML += `<div class="option">${String.fromCharCode(65 + oIndex)}. ${option}</div>`;
                });
                optionsHTML += '</div>';
            }

            // Add prediction confidence indicator if available
            let predictionIndicator = '';
            if (isPredicted && question.prediction_score !== undefined) {
                const predScore = question.prediction_score;
                const predPercent = Math.round(predScore * 100);
                const predLevel = predScore >= 0.75 ? 'high' : predScore >= 0.5 ? 'medium' : 'low';
                predictionIndicator = `<span class="prediction-confidence ${predLevel}" title="Prediction confidence">🎯 ${predPercent}%</span>`;
            }

            questionDiv.innerHTML = `
                <div class="question-number">Question ${index + 1}</div>
                <div class="question-text">${question.question_text}</div>
                ${optionsHTML}
                <div class="question-meta">
                    <span class="difficulty-badge ${question.difficulty}">${question.difficulty || 'medium'}</span>
                    <span class="subject-badge">${question.subject || paper.paper_info?.subject || ''}</span>
                    ${predictionIndicator}
                </div>
            `;

            paperContent.appendChild(questionDiv);
        });

        generatedPaper.classList.remove('hidden');
    }

    // Get confidence badge HTML
    function getConfidenceBadge(confidence) {
        const percent = Math.round(confidence * 100);
        let level = 'low';
        let color = '#ef4444';
        
        if (confidence >= 0.75) {
            level = 'high';
            color = '#10b981';
        } else if (confidence >= 0.5) {
            level = 'medium';
            color = '#f59e0b';
        }

        return `<span class="confidence-badge ${level}" style="background-color: ${color}20; color: ${color}; border: 1px solid ${color}40;">${percent}% Confidence</span>`;
    }

    // Start test
    function startTest() {
        if (!currentPaper) return;

        currentTest = {
            paper: currentPaper,
            answers: {},
            marked: new Set(),
            startTime: Date.now()
        };

        testStartTime = Date.now();
        currentQuestionIndex = 0;
        userAnswers = {};

        document.getElementById('generated-paper').classList.add('hidden');
        document.getElementById('test-interface').classList.remove('hidden');

        setupTestInterface();
        startTimer();
        displayQuestion(0);
    }

    // Setup test interface
    function setupTestInterface() {
        const questionGrid = document.getElementById('question-grid');
        questionGrid.innerHTML = '';

        currentPaper.questions.forEach((_, index) => {
            const btn = document.createElement('button');
            btn.className = 'question-number-btn unanswered';
            btn.textContent = index + 1;
            btn.addEventListener('click', () => displayQuestion(index));
            questionGrid.appendChild(btn);
        });
    }

    // Start timer
    function startTimer() {
        const duration = currentPaper.duration_minutes || 180; // Default 3 hours
        const endTime = testStartTime + (duration * 60 * 1000);

        timerInterval = setInterval(() => {
            const remaining = endTime - Date.now();
            
            if (remaining <= 0) {
                clearInterval(timerInterval);
                submitTest();
                return;
            }

            const hours = Math.floor(remaining / 3600000);
            const minutes = Math.floor((remaining % 3600000) / 60000);
            const seconds = Math.floor((remaining % 60000) / 1000);

            document.getElementById('timer-display').textContent = 
                `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }, 1000);
    }

    // Display question
    function displayQuestion(index) {
        currentQuestionIndex = index;
        const question = currentPaper.questions[index];
        const currentQuestion = document.getElementById('current-question');

        let optionsHTML = '';
        if (question.options) {
            optionsHTML = '<div class="test-options">';
            question.options.forEach((option, oIndex) => {
                const optionLetter = String.fromCharCode(65 + oIndex);
                const isSelected = userAnswers[index] === option;
                optionsHTML += `
                    <label class="test-option ${isSelected ? 'selected' : ''}">
                        <input type="radio" name="answer-${index}" value="${option}" ${isSelected ? 'checked' : ''}>
                        <span>${optionLetter}. ${option}</span>
                    </label>
                `;
            });
            optionsHTML += '</div>';
        }

        currentQuestion.innerHTML = `
            <div class="question-header">
                <h3>Question ${index + 1} of ${currentPaper.questions.length}</h3>
                <span class="subject-badge">${question.subject}</span>
            </div>
            <div class="question-text">${question.question_text}</div>
            ${optionsHTML}
        `;

        // Add event listeners to options
        currentQuestion.querySelectorAll('input[type="radio"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                userAnswers[index] = e.target.value;
                updateQuestionStatus(index, 'answered');
            });
        });

        // Update navigation buttons
        document.getElementById('prev-question-btn').disabled = index === 0;
        document.getElementById('next-question-btn').disabled = index === currentPaper.questions.length - 1;

        // Update question grid
        updateQuestionGrid();
    }

    // Navigate question
    function navigateQuestion(direction) {
        const newIndex = currentQuestionIndex + direction;
        if (newIndex >= 0 && newIndex < currentPaper.questions.length) {
            displayQuestion(newIndex);
        }
    }

    // Mark for review
    function markForReview() {
        updateQuestionStatus(currentQuestionIndex, 'marked');
    }

    // Clear response
    function clearResponse() {
        delete userAnswers[currentQuestionIndex];
        updateQuestionStatus(currentQuestionIndex, 'unanswered');
        displayQuestion(currentQuestionIndex);
    }

    // Update question status
    function updateQuestionStatus(index, status) {
        const btn = document.querySelectorAll('.question-number-btn')[index];
        if (btn) {
            btn.className = `question-number-btn ${status}`;
        }
    }

    // Update question grid
    function updateQuestionGrid() {
        document.querySelectorAll('.question-number-btn').forEach((btn, index) => {
            btn.classList.remove('current');
            if (index === currentQuestionIndex) {
                btn.classList.add('current');
            }
        });
    }

    // Submit test
    async function submitTest() {
        if (!confirm('Are you sure you want to submit the test?')) {
            return;
        }

        clearInterval(timerInterval);

        const timeTaken = Date.now() - testStartTime;
        const results = calculateResults();

        // Save progress
        await saveTestProgress(results);

        // Display results
        displayResults(results, timeTaken);
    }

    // Calculate results
    function calculateResults() {
        let correct = 0;
        let incorrect = 0;
        let unattempted = 0;

        currentPaper.questions.forEach((question, index) => {
            if (userAnswers[index]) {
                if (userAnswers[index] === question.correct_answer) {
                    correct++;
                } else {
                    incorrect++;
                }
            } else {
                unattempted++;
            }
        });

        return { correct, incorrect, unattempted };
    }

    // Save test progress
    async function saveTestProgress(results) {
        try {
            await fetch('/api/progress/record', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    paper_id: currentPaper.paper_id,
                    answers: userAnswers,
                    results: results
                }),
                credentials: 'include'
            });
        } catch (error) {
            console.error('Error saving progress:', error);
        }
    }

    // Display results
    function displayResults(results, timeTaken) {
        document.getElementById('test-interface').classList.add('hidden');
        document.getElementById('test-results').classList.remove('hidden');

        const total = currentPaper.questions.length;
        const percentage = Math.round((results.correct / total) * 100);

        document.getElementById('score-percentage').textContent = `${percentage}%`;
        document.getElementById('score-text').textContent = `${results.correct} / ${total} Correct`;
        document.getElementById('correct-count').textContent = results.correct;
        document.getElementById('incorrect-count').textContent = results.incorrect;
        document.getElementById('unattempted-count').textContent = results.unattempted;
        document.getElementById('time-taken').textContent = formatTime(timeTaken);
    }

    // View solutions
    function viewSolutions() {
        const solutionsDisplay = document.getElementById('solutions-display');
        solutionsDisplay.innerHTML = '';

        currentPaper.questions.forEach((question, index) => {
            const solutionDiv = document.createElement('div');
            solutionDiv.className = 'solution-item';

            const userAnswer = userAnswers[index];
            const isCorrect = userAnswer === question.correct_answer;
            const statusClass = !userAnswer ? 'unattempted' : (isCorrect ? 'correct' : 'incorrect');

            solutionDiv.innerHTML = `
                <div class="solution-header ${statusClass}">
                    <h4>Question ${index + 1}</h4>
                    <span class="status-badge">${!userAnswer ? 'Not Attempted' : (isCorrect ? 'Correct' : 'Incorrect')}</span>
                </div>
                <div class="solution-question">${question.question_text}</div>
                <div class="solution-answer">
                    <p><strong>Your Answer:</strong> ${userAnswer || 'Not answered'}</p>
                    <p><strong>Correct Answer:</strong> ${question.correct_answer}</p>
                </div>
                <div class="solution-explanation">
                    <h5>Explanation:</h5>
                    <p>${question.solution || 'No explanation available'}</p>
                </div>
                ${question.ncert_reference ? `<div class="solution-reference">Reference: ${question.ncert_reference}</div>` : ''}
            `;

            solutionsDisplay.appendChild(solutionDiv);
        });

        solutionsDisplay.classList.remove('hidden');
    }

    // Retry test
    function retryTest() {
        userAnswers = {};
        startTest();
    }

    // Back to generator
    function backToGenerator() {
        document.getElementById('test-results').classList.add('hidden');
        document.getElementById('generated-paper').classList.add('hidden');
        hideAllConfigs();
    }

    // Download paper
    function downloadPaper() {
        // Create downloadable content
        let content = `${currentPaper.title}\n\n`;
        currentPaper.questions.forEach((q, i) => {
            content += `${i + 1}. ${q.question_text}\n`;
            if (q.options) {
                q.options.forEach((opt, j) => {
                    content += `   ${String.fromCharCode(65 + j)}. ${opt}\n`;
                });
            }
            content += '\n';
        });

        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentPaper.title.replace(/\s+/g, '_')}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // Print paper
    function printPaper() {
        window.print();
    }

    // Load previous years for filter
    function loadPreviousYears() {
        const yearFilter = document.getElementById('prev-year-filter');
        if (!yearFilter) return;

        const currentYear = new Date().getFullYear();
        for (let year = currentYear; year >= currentYear - 20; year--) {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearFilter.appendChild(option);
        }
    }

    // Load previous papers
    async function loadPreviousPapers() {
        try {
            const response = await fetch('/api/previous-papers', {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to load previous papers');
            }

            const data = await response.json();
            displayPreviousPapers(data.papers || []);

        } catch (error) {
            console.error('Error loading previous papers:', error);
        }
    }

    // Display previous papers
    function displayPreviousPapers(papers) {
        const papersList = document.getElementById('previous-papers-list');
        papersList.innerHTML = '';

        if (papers.length === 0) {
            papersList.innerHTML = '<p>No previous papers available</p>';
            return;
        }

        papers.forEach(paper => {
            const paperDiv = document.createElement('div');
            paperDiv.className = 'previous-paper-item';
            paperDiv.innerHTML = `
                <div class="paper-info">
                    <h4>${paper.exam} ${paper.year}</h4>
                    <p>${paper.subject} | ${paper.question_count} questions</p>
                </div>
                <button class="load-paper-btn" data-paper-id="${paper.paper_id}">Load Paper</button>
            `;
            papersList.appendChild(paperDiv);
        });

        // Add event listeners
        document.querySelectorAll('.load-paper-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const paperId = e.target.dataset.paperId;
                await loadPreviousPaper(paperId);
            });
        });
    }

    // Load specific previous paper
    async function loadPreviousPaper(paperId) {
        showLoading(true);

        try {
            const response = await fetch(`/api/previous-papers/${paperId}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to load paper');
            }

            const data = await response.json();
            currentPaper = data;
            displayGeneratedPaper(data);

        } catch (error) {
            console.error('Error loading paper:', error);
            alert('Failed to load paper. Please try again.');
        } finally {
            showLoading(false);
        }
    }

    // Generate subject-wise prediction
    async function generateSubjectPrediction() {
        const subject = document.getElementById('prediction-subject')?.value;
        const year = parseInt(document.getElementById('prediction-year')?.value);
        const btn = document.getElementById('generate-subject-prediction-btn');

        if (!subject || !year) {
            alert('Please select both subject and year');
            return;
        }

        showLoading(true);
        btn.disabled = true;
        btn.textContent = 'Generating...';

        try {
            const response = await fetch('/api/prediction/predict-paper', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include',
                body: JSON.stringify({
                    subject: subject,
                    year: year,
                    use_ai: true
                })
            });

            if (!response.ok) {
                if (response.status === 403) {
                    throw new Error('PERMISSION_DENIED');
                }
                throw new Error('Failed to generate prediction');
            }

            const data = await response.json();
            currentPaper = data;
            displayGeneratedPaper(data);

        } catch (error) {
            console.error('Prediction error:', error);
            if (error.message === 'PERMISSION_DENIED') {
                alert('This feature requires a premium subscription. Please upgrade your plan.');
            } else {
                alert('Failed to generate prediction. Please try again.');
            }
        } finally {
            showLoading(false);
            btn.disabled = false;
            btn.textContent = 'Generate Subject Prediction';
        }
    }

    // Generate complete NEET prediction
    async function generateCompletePrediction() {
        const year = parseInt(document.getElementById('complete-prediction-year')?.value);
        const btn = document.getElementById('generate-complete-prediction-btn');

        if (!year) {
            alert('Please select a year');
            return;
        }

        showLoading(true);
        btn.disabled = true;
        btn.textContent = 'Generating...';

        try {
            const response = await fetch(`/api/prediction/complete-neet/${year}`, {
                method: 'GET',
                credentials: 'include'
            });

            if (!response.ok) {
                if (response.status === 403) {
                    throw new Error('PERMISSION_DENIED');
                }
                throw new Error('Failed to generate complete prediction');
            }

            const data = await response.json();
            currentPaper = data;
            displayGeneratedPaper(data);

        } catch (error) {
            console.error('Complete prediction error:', error);
            if (error.message === 'PERMISSION_DENIED') {
                alert('This feature requires a premium subscription. Please upgrade your plan.');
            } else {
                alert('Failed to generate complete prediction. Please try again.');
            }
        } finally {
            showLoading(false);
            btn.disabled = false;
            btn.textContent = 'Generate Complete Prediction';
        }
    }

    // Format time
    function formatTime(ms) {
        const hours = Math.floor(ms / 3600000);
        const minutes = Math.floor((ms % 3600000) / 60000);
        const seconds = Math.floor((ms % 60000) / 1000);
        return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    // Show/hide loading
    function showLoading(show) {
        document.getElementById('qp-loading')?.classList.toggle('hidden', !show);
    }
});
