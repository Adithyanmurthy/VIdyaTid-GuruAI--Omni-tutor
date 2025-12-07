/**
 * NEET Mock Exam System
 * Professional exam interface with anti-cheating measures
 */

class NEETExam {
    constructor(examData) {
        this.examData = examData;
        this.questions = examData.questions || [];
        this.currentQuestionIndex = 0;
        this.answers = {};
        this.markedForReview = new Set();
        this.visitedQuestions = new Set();
        this.timeRemaining = examData.paper_info?.duration_minutes * 60 || 10800; // seconds
        this.timerInterval = null;
        this.tabSwitchCount = 0;
        this.isFullScreen = false;
        
        this.init();
    }
    
    init() {
        // Setup instructions screen
        this.setupInstructionsScreen();
        
        // Setup anti-cheating measures
        this.setupAntiCheating();
        
        // Disable right-click
        document.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Disable certain keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Disable F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U
            if (e.keyCode === 123 || 
                (e.ctrlKey && e.shiftKey && (e.keyCode === 73 || e.keyCode === 74)) ||
                (e.ctrlKey && e.keyCode === 85)) {
                e.preventDefault();
                return false;
            }
        });
    }
    
    setupInstructionsScreen() {
        const acceptTerms = document.getElementById('accept-terms');
        const startBtn = document.getElementById('start-exam-btn');
        
        acceptTerms.addEventListener('change', () => {
            startBtn.disabled = !acceptTerms.checked;
        });
        
        startBtn.addEventListener('click', () => {
            this.startExam();
        });
    }
    
    setupAntiCheating() {
        // Monitor tab switching
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.timerInterval) {
                this.tabSwitchCount++;
                if (this.tabSwitchCount >= 3) {
                    this.showWarning('Multiple tab switches detected! Your exam may be flagged for review.');
                } else {
                    this.showWarning(`Warning: Tab switching detected (${this.tabSwitchCount}/3). Stay on this tab!`);
                }
            }
        });
        
        // Monitor window blur
        window.addEventListener('blur', () => {
            if (this.timerInterval) {
                this.showWarning('Please stay focused on the exam window!');
            }
        });
        
        // Monitor fullscreen exit
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement && this.timerInterval) {
                this.showWarning('Please return to fullscreen mode!');
                this.requestFullScreen();
            }
        });
    }
    
    requestFullScreen() {
        const elem = document.documentElement;
        if (elem.requestFullscreen) {
            elem.requestFullscreen().catch(err => {
                console.log('Fullscreen request failed:', err);
            });
        }
    }
    
    startExam() {
        // Request fullscreen
        this.requestFullScreen();
        
        // Switch to exam screen
        document.getElementById('instructions-screen').classList.remove('active');
        document.getElementById('exam-screen').classList.add('active');
        
        // Set candidate name
        document.getElementById('candidate-name').textContent = 'Candidate'; // Get from session
        
        // Initialize exam
        this.initializeExam();
        
        // Start timer
        this.startTimer();
    }
    
    initializeExam() {
        // Create question palette
        this.createQuestionPalette();
        
        // Load first question
        this.loadQuestion(0);
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Mark first question as visited
        this.visitedQuestions.add(0);
        this.updatePalette();
    }
    
    createQuestionPalette() {
        const physicsGrid = document.getElementById('physics-grid');
        const chemistryGrid = document.getElementById('chemistry-grid');
        const biologyGrid = document.getElementById('biology-grid');
        
        // Physics: 1-45
        for (let i = 0; i < 45; i++) {
            const btn = this.createQuestionButton(i + 1);
            physicsGrid.appendChild(btn);
        }
        
        // Chemistry: 46-90
        for (let i = 45; i < 90; i++) {
            const btn = this.createQuestionButton(i + 1);
            chemistryGrid.appendChild(btn);
        }
        
        // Biology: 91-180
        for (let i = 90; i < 180; i++) {
            const btn = this.createQuestionButton(i + 1);
            biologyGrid.appendChild(btn);
        }
    }
    
    createQuestionButton(number) {
        const btn = document.createElement('button');
        btn.className = 'question-btn not-visited';
        btn.textContent = number;
        btn.dataset.questionIndex = number - 1;
        
        btn.addEventListener('click', () => {
            this.loadQuestion(parseInt(btn.dataset.questionIndex));
        });
        
        return btn;
    }
    
    loadQuestion(index) {
        if (index < 0 || index >= this.questions.length) return;
        
        this.currentQuestionIndex = index;
        const question = this.questions[index];
        
        // Mark as visited
        this.visitedQuestions.add(index);
        
        // Update question display
        document.getElementById('current-question-number').textContent = index + 1;
        document.getElementById('current-subject').textContent = this.getSubject(index);
        document.getElementById('question-content').innerHTML = question.question_text;
        
        // Load options
        this.loadOptions(question, index);
        
        // Update palette
        this.updatePalette();
        
        // Update navigation buttons
        this.updateNavigationButtons();
    }
    
    getSubject(index) {
        if (index < 45) return 'Physics';
        if (index < 90) return 'Chemistry';
        return 'Biology';
    }
    
    loadOptions(question, questionIndex) {
        const container = document.getElementById('options-container');
        container.innerHTML = '';
        
        const options = question.options || [];
        const labels = ['A', 'B', 'C', 'D'];
        
        options.forEach((option, i) => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'option-item';
            if (this.answers[questionIndex] === i) {
                optionDiv.classList.add('selected');
            }
            
            optionDiv.innerHTML = `
                <div class="option-radio"></div>
                <span class="option-label">${labels[i]}.</span>
                <span class="option-text">${option}</span>
            `;
            
            optionDiv.addEventListener('click', () => {
                this.selectOption(questionIndex, i);
            });
            
            container.appendChild(optionDiv);
        });
    }
    
    selectOption(questionIndex, optionIndex) {
        this.answers[questionIndex] = optionIndex;
        this.loadOptions(this.questions[questionIndex], questionIndex);
        this.updatePalette();
    }
    
    updatePalette() {
        const buttons = document.querySelectorAll('.question-btn');
        
        buttons.forEach((btn, index) => {
            btn.classList.remove('answered', 'not-answered', 'marked', 'not-visited', 'current');
            
            if (index === this.currentQuestionIndex) {
                btn.classList.add('current');
            }
            
            if (this.markedForReview.has(index)) {
                btn.classList.add('marked');
            } else if (this.answers.hasOwnProperty(index)) {
                btn.classList.add('answered');
            } else if (this.visitedQuestions.has(index)) {
                btn.classList.add('not-answered');
            } else {
                btn.classList.add('not-visited');
            }
        });
        
        // Update summary
        this.updateSummary();
    }
    
    updateSummary() {
        const answered = Object.keys(this.answers).length - this.markedForReview.size;
        const marked = this.markedForReview.size;
        const notAnswered = this.visitedQuestions.size - Object.keys(this.answers).length;
        const notVisited = this.questions.length - this.visitedQuestions.size;
        
        document.getElementById('answered-count').textContent = answered;
        document.getElementById('marked-count').textContent = marked;
        document.getElementById('not-answered-count').textContent = notAnswered;
        document.getElementById('not-visited-count').textContent = notVisited;
    }
    
    updateNavigationButtons() {
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        
        prevBtn.disabled = this.currentQuestionIndex === 0;
        nextBtn.disabled = this.currentQuestionIndex === this.questions.length - 1;
    }
    
    setupEventListeners() {
        // Save & Next
        document.getElementById('save-next-btn').addEventListener('click', () => {
            if (this.currentQuestionIndex < this.questions.length - 1) {
                this.loadQuestion(this.currentQuestionIndex + 1);
            }
        });
        
        // Mark for Review
        document.getElementById('mark-review-btn').addEventListener('click', () => {
            this.markedForReview.add(this.currentQuestionIndex);
            this.updatePalette();
            if (this.currentQuestionIndex < this.questions.length - 1) {
                this.loadQuestion(this.currentQuestionIndex + 1);
            }
        });
        
        // Clear Response
        document.getElementById('clear-response-btn').addEventListener('click', () => {
            delete this.answers[this.currentQuestionIndex];
            this.markedForReview.delete(this.currentQuestionIndex);
            this.loadOptions(this.questions[this.currentQuestionIndex], this.currentQuestionIndex);
            this.updatePalette();
        });
        
        // Previous
        document.getElementById('prev-btn').addEventListener('click', () => {
            if (this.currentQuestionIndex > 0) {
                this.loadQuestion(this.currentQuestionIndex - 1);
            }
        });
        
        // Next
        document.getElementById('next-btn').addEventListener('click', () => {
            if (this.currentQuestionIndex < this.questions.length - 1) {
                this.loadQuestion(this.currentQuestionIndex + 1);
            }
        });
        
        // Submit
        document.getElementById('submit-exam-btn').addEventListener('click', () => {
            this.showSubmitConfirmation();
        });
        
        // Modal buttons
        document.getElementById('cancel-submit-btn').addEventListener('click', () => {
            this.hideModal('submit-modal');
        });
        
        document.getElementById('confirm-submit-btn').addEventListener('click', () => {
            this.submitExam();
        });
        
        document.getElementById('warning-ok-btn').addEventListener('click', () => {
            this.hideModal('warning-modal');
        });
    }
    
    startTimer() {
        this.updateTimerDisplay();
        
        this.timerInterval = setInterval(() => {
            this.timeRemaining--;
            this.updateTimerDisplay();
            
            // Warning at 10 minutes
            if (this.timeRemaining === 600) {
                this.showWarning('Only 10 minutes remaining!');
                document.getElementById('exam-timer').classList.add('warning');
            }
            
            // Auto-submit at 0
            if (this.timeRemaining <= 0) {
                clearInterval(this.timerInterval);
                this.showWarning('Time is up! Exam will be submitted automatically.');
                setTimeout(() => {
                    this.submitExam();
                }, 2000);
            }
        }, 1000);
    }
    
    updateTimerDisplay() {
        const hours = Math.floor(this.timeRemaining / 3600);
        const minutes = Math.floor((this.timeRemaining % 3600) / 60);
        const seconds = this.timeRemaining % 60;
        
        const display = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        document.getElementById('exam-timer').textContent = display;
    }
    
    showSubmitConfirmation() {
        const answered = Object.keys(this.answers).length - this.markedForReview.size;
        const marked = this.markedForReview.size;
        const notAnswered = this.questions.length - Object.keys(this.answers).length;
        
        document.getElementById('modal-answered').textContent = answered;
        document.getElementById('modal-marked').textContent = marked;
        document.getElementById('modal-not-answered').textContent = notAnswered;
        
        this.showModal('submit-modal');
    }
    
    submitExam() {
        // Stop timer
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        
        // Hide modal
        this.hideModal('submit-modal');
        
        // Calculate results
        const results = this.calculateResults();
        
        // Show results
        this.showResults(results);
    }
    
    calculateResults() {
        let correct = 0;
        let incorrect = 0;
        let unattempted = 0;
        
        const subjectResults = {
            Physics: { correct: 0, incorrect: 0, unattempted: 0 },
            Chemistry: { correct: 0, incorrect: 0, unattempted: 0 },
            Biology: { correct: 0, incorrect: 0, unattempted: 0 }
        };
        
        this.questions.forEach((question, index) => {
            const subject = this.getSubject(index);
            
            if (this.answers.hasOwnProperty(index)) {
                const selectedOption = question.options[this.answers[index]];
                if (selectedOption === question.correct_answer) {
                    correct++;
                    subjectResults[subject].correct++;
                } else {
                    incorrect++;
                    subjectResults[subject].incorrect++;
                }
            } else {
                unattempted++;
                subjectResults[subject].unattempted++;
            }
        });
        
        const totalMarks = (correct * 4) - (incorrect * 1);
        const percentage = ((correct / this.questions.length) * 100).toFixed(2);
        
        return {
            correct,
            incorrect,
            unattempted,
            totalMarks,
            percentage,
            subjectResults
        };
    }
    
    showResults(results) {
        // Exit fullscreen
        if (document.fullscreenElement) {
            document.exitFullscreen();
        }
        
        // Switch to results screen
        document.getElementById('exam-screen').classList.remove('active');
        document.getElementById('results-screen').classList.add('active');
        
        // Update results display
        document.getElementById('score-percentage').textContent = results.percentage + '%';
        document.getElementById('correct-answers').textContent = results.correct;
        document.getElementById('incorrect-answers').textContent = results.incorrect;
        document.getElementById('unattempted-answers').textContent = results.unattempted;
        document.getElementById('total-marks').textContent = results.totalMarks + ' / 720';
        
        // Update score circle
        const circle = document.getElementById('score-circle');
        const circumference = 2 * Math.PI * 90;
        const offset = circumference - (results.percentage / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        
        // Update subject-wise results
        Object.keys(results.subjectResults).forEach(subject => {
            const data = results.subjectResults[subject];
            const marks = (data.correct * 4) - (data.incorrect * 1);
            
            document.getElementById(`${subject.toLowerCase()}-correct`).textContent = data.correct;
            document.getElementById(`${subject.toLowerCase()}-incorrect`).textContent = data.incorrect;
            document.getElementById(`${subject.toLowerCase()}-marks`).textContent = marks;
        });
        
        // Setup view solutions button
        document.getElementById('view-solutions-btn').addEventListener('click', () => {
            this.showSolutions();
        });
    }
    
    showSolutions() {
        // TODO: Implement solutions view
        alert('Solutions view will be implemented');
    }
    
    showModal(modalId) {
        document.getElementById(modalId).classList.add('active');
    }
    
    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }
    
    showWarning(message) {
        document.getElementById('warning-message').textContent = message;
        this.showModal('warning-modal');
    }
}

// Toggle section collapse
function toggleSection(sectionId) {
    const section = document.querySelector(`#${sectionId}-grid`).parentElement;
    section.classList.toggle('collapsed');
}

// Initialize exam when page loads
document.addEventListener('DOMContentLoaded', () => {
    if (typeof examData !== 'undefined') {
        window.neetExam = new NEETExam(examData);
    } else {
        console.error('Exam data not found!');
    }
});
