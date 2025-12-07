// Chat Interface - Main functionality for GuruAI
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');
    const loadingIndicator = document.getElementById('loading-indicator');
    const recordBtn = document.getElementById('record-btn');
    const imageUploadInput = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const removeImageBtn = document.getElementById('remove-image');
    const modelStatus = document.getElementById('model-status');
    const quizContainer = document.getElementById('quiz-container');

    let currentImage = null;
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];

    // Initialize chat
    function initializeChat() {
        chatWindow.innerHTML = '';
        userInput.placeholder = "Ask about NCERT concepts, upload diagrams, or paste problems...";
        
        const welcomeMessage = "Welcome to GuruAI! I'm your offline study companion for JEE & NEET preparation. Ask me anything about Physics, Chemistry, Mathematics, or Biology from NCERT textbooks.";
        displayMessage(welcomeMessage, 'bot');
        
        // Auto-resize textarea
        userInput.addEventListener('input', autoResizeTextarea);
        
        // Handle Enter key to submit (Shift+Enter for new line)
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
        
        // Check authentication
        checkAuth();
    }

    initializeChat();

    // Check if user is authenticated
    async function checkAuth() {
        try {
            const response = await fetch('/api/auth/check', {
                credentials: 'include'
            });
            const data = await response.json();
            
            if (data.authenticated) {
                updateUserProfile(data.username);
            } else {
                // Authentication not required for demo mode
                // Redirect disabled to allow viewing offline verification
                // if (window.location.pathname !== '/auth') {
                //     window.location.href = '/auth';
                // }
            }
        } catch (error) {
            console.error('Auth check failed:', error);
        }
    }

    // Update user profile display
    function updateUserProfile(username) {
        const profileLink = document.getElementById('user-profile');
        if (profileLink) {
            profileLink.textContent = username;
            profileLink.href = '#';
            profileLink.addEventListener('click', (e) => {
                e.preventDefault();
                showProfileMenu();
            });
        }
    }

    // Show profile menu
    function showProfileMenu() {
        const menu = document.createElement('div');
        menu.className = 'profile-menu';
        menu.innerHTML = `
            <button id="logout-btn">Logout</button>
        `;
        document.body.appendChild(menu);
        
        document.getElementById('logout-btn').addEventListener('click', logout);
        
        // Close menu on outside click
        setTimeout(() => {
            document.addEventListener('click', function closeMenu(e) {
                if (!menu.contains(e.target)) {
                    menu.remove();
                    document.removeEventListener('click', closeMenu);
                }
            });
        }, 100);
    }

    // Logout function
    async function logout() {
        try {
            await fetch('/api/auth/logout', {
                method: 'POST',
                credentials: 'include'
            });
            window.location.href = '/auth';
        } catch (error) {
            console.error('Logout failed:', error);
        }
    }

    // Auto-resize textarea
    function autoResizeTextarea() {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
    }

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        
        if (!query && !currentImage) return;

        // Check trial limit before processing
        if (window.trialManager && !window.trialManager.canAskQuestion()) {
            window.trialManager.showUpgradeModal();
            return;
        }

        // Clear previous quiz when new query is submitted
        quizContainer.classList.add('hidden');
        quizContainer.innerHTML = '';

        if (currentImage) {
            await handleImageUpload(query);
        } else {
            await sendQuery(query);
        }
        
        // Increment trial count after successful query
        if (window.trialManager) {
            window.trialManager.incrementTrialCount();
        }
        
        userInput.value = '';
        autoResizeTextarea();
    });

    // Send text query
    async function sendQuery(query) {
        displayMessage(query, 'user');
        toggleLoading(true);
        showModelStatus('Processing your query...');

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query,
                    include_quiz: true,  // Enable quiz with Cloudflare AI
                    include_diagrams: true
                }),
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to get response');
            }

            const data = await response.json();
            
            // Display explanation
            displayMessage(data.explanation, 'bot', data.references);
            
            // Display diagrams if available
            if (data.diagrams && data.diagrams.length > 0) {
                displayDiagrams(data.diagrams);
            }
            
            // Display quiz if available
            console.log('Quiz data:', data.quiz);
            if (data.quiz && data.quiz.questions && data.quiz.questions.length > 0) {
                console.log('Displaying quiz with', data.quiz.questions.length, 'questions');
                displayQuiz(data.quiz);
            } else {
                console.log('No quiz to display');
            }

        } catch (error) {
            console.error('Error:', error);
            displayMessage('Sorry, I encountered an error processing your query. Please try again.', 'bot');
        } finally {
            toggleLoading(false);
            hideModelStatus();
        }
    }

    // Handle image upload
    async function handleImageUpload(additionalQuery = '') {
        if (!currentImage) return;

        displayMessage(additionalQuery || 'Analyzing uploaded image...', 'user');
        
        if (currentImage) {
            displayImageInChat(currentImage);
        }
        
        toggleLoading(true);
        showModelStatus('Processing image...');

        try {
            const formData = new FormData();
            formData.append('image', currentImage);
            if (additionalQuery) {
                formData.append('query', additionalQuery);
            }

            const response = await fetch('/api/solve-image', {
                method: 'POST',
                body: formData,
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to process image');
            }

            const data = await response.json();
            
            // Display confidence warning for complex problems
            if (data.confidence === 'low' || data.confidence === 'medium') {
                displayMessage(
                    '⚠️ Note: This is a complex problem. The AI solution may contain errors. Please verify the answer independently.',
                    'bot'
                );
            }
            
            // Display solution or explanation
            if (data.solution) {
                displayMessage(data.solution, 'bot');
            }
            
            // Display steps if available
            if (data.steps && data.steps.length > 0) {
                displaySteps(data.steps);
            }
            
            // Display matched diagram info
            if (data.matched_diagram) {
                displayMessage(`This appears to be from NCERT ${data.matched_diagram.subject}, Chapter ${data.matched_diagram.chapter}`, 'bot');
            }

        } catch (error) {
            console.error('Error:', error);
            displayMessage('Sorry, I had trouble processing the image. Please ensure it\'s clear and try again.', 'bot');
        } finally {
            toggleLoading(false);
            hideModelStatus();
            clearImagePreview();
        }
    }

    // Display message in chat
    function displayMessage(message, sender, references = null) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message ${sender}-message message-animated`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Convert markdown-style formatting (but preserve LaTeX)
        let formattedMessage = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
        
        messageContent.innerHTML = `<p>${formattedMessage}</p>`;
        
        // Render mathematical expressions using KaTeX
        if (window.MathRenderer) {
            MathRenderer.renderMath(messageContent);
        }
        
        // Add references if available
        if (references && references.length > 0) {
            const refsDiv = document.createElement('div');
            refsDiv.className = 'message-references';
            refsDiv.innerHTML = '<strong>References:</strong> ' + references.map(ref => 
                `<span class="reference-tag">${ref}</span>`
            ).join(' ');
            messageContent.appendChild(refsDiv);
        }
        
        // Add voice output button for bot messages
        if (sender === 'bot') {
            const voiceBtn = document.createElement('button');
            voiceBtn.className = 'voice-output-btn';
            voiceBtn.title = 'Listen to response';
            voiceBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                </svg>
            `;
            voiceBtn.onclick = () => speakText(message);
            messageContent.appendChild(voiceBtn);
        }
        
        messageWrapper.appendChild(messageContent);
        chatWindow.appendChild(messageWrapper);
        scrollToBottom();
    }
    
    // Speak text using ElevenLabs
    async function speakText(text) {
        try {
            // Remove HTML tags and get plain text
            const plainText = text.replace(/<[^>]*>/g, '').replace(/\n/g, ' ').trim();
            
            if (!plainText) return;
            
            // Show loading indicator
            showModelStatus('Generating speech...');
            
            const response = await fetch('/api/voice/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: plainText }),
                credentials: 'include'
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate speech');
            }
            
            // Get audio blob
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Play audio
            const audio = new Audio(audioUrl);
            audio.play();
            
            hideModelStatus();
            
            // Clean up URL after playing
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
            };
            
        } catch (error) {
            console.error('Speech synthesis error:', error);
            hideModelStatus();
            // Silently fail - don't show error to user
        }
    }

    // Display image in chat
    function displayImageInChat(file) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message user-message message-animated';
        
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.className = 'chat-image';
        img.style.maxWidth = '300px';
        img.style.borderRadius = '8px';
        
        messageWrapper.appendChild(img);
        chatWindow.appendChild(messageWrapper);
        scrollToBottom();
    }

    // Display diagrams using the diagram viewer
    function displayDiagrams(diagrams) {
        if (!diagrams || diagrams.length === 0) return;
        
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message bot-message message-animated';
        
        // Use the diagram viewer to display diagrams
        if (typeof diagramViewer !== 'undefined') {
            diagramViewer.displayMultipleDiagrams(diagrams, messageWrapper);
        } else {
            // Fallback to simple display if diagram viewer not loaded
            const diagramsContainer = document.createElement('div');
            diagramsContainer.className = 'diagrams-container';
            
            diagrams.forEach(diagram => {
                const diagramDiv = document.createElement('div');
                diagramDiv.className = 'diagram-item';
                diagramDiv.innerHTML = `
                    <img src="/api/diagrams/file/${diagram.page_id}" alt="Figure ${diagram.figures.join(', ')}" class="diagram-image">
                    <p class="diagram-reference">${diagram.reference}</p>
                `;
                diagramsContainer.appendChild(diagramDiv);
            });
            
            messageWrapper.appendChild(diagramsContainer);
        }
        
        chatWindow.appendChild(messageWrapper);
        scrollToBottom();
    }

    // Display solution steps
    function displaySteps(steps) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message bot-message message-animated';
        
        const stepsContainer = document.createElement('div');
        stepsContainer.className = 'steps-container';
        stepsContainer.innerHTML = '<h4>Step-by-Step Solution:</h4>';
        
        steps.forEach((step, index) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'solution-step';
            
            // Handle both string steps and object steps with content property
            const stepContent = typeof step === 'string' ? step : (step.content || JSON.stringify(step));
            const stepNumber = typeof step === 'object' && step.step_number ? step.step_number : (index + 1);
            
            stepDiv.innerHTML = `
                <div class="step-number">Step ${stepNumber}</div>
                <div class="step-content">${stepContent}</div>
            `;
            stepsContainer.appendChild(stepDiv);
        });
        
        // Render math in all steps
        if (window.MathRenderer) {
            MathRenderer.renderMath(stepsContainer);
        }
        
        messageWrapper.appendChild(stepsContainer);
        chatWindow.appendChild(messageWrapper);
        scrollToBottom();
    }

    // Display quiz
    function displayQuiz(quiz) {
        // Clear and show quiz container
        quizContainer.classList.remove('hidden');
        quizContainer.innerHTML = '';
        
        // Create section header
        const sectionHeader = document.createElement('div');
        sectionHeader.className = 'quiz-section-header';
        
        const headerTitle = document.createElement('h3');
        headerTitle.innerHTML = '📝 Test Your Understanding';
        
        const headerSubtitle = document.createElement('p');
        headerSubtitle.textContent = `${quiz.questions.length} questions to reinforce your learning`;
        
        // Close button in header
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '✕';
        closeBtn.style.cssText = 'position: absolute; top: 20px; right: 20px; background: rgba(255,255,255,0.2); color: white; border: none; width: 36px; height: 36px; border-radius: 50%; cursor: pointer; font-size: 20px; transition: all 0.3s;';
        closeBtn.onmouseover = () => closeBtn.style.background = 'rgba(255,255,255,0.3)';
        closeBtn.onmouseout = () => closeBtn.style.background = 'rgba(255,255,255,0.2)';
        closeBtn.onclick = () => {
            quizContainer.classList.add('hidden');
            quizContainer.innerHTML = '';
        };
        
        sectionHeader.appendChild(headerTitle);
        sectionHeader.appendChild(headerSubtitle);
        sectionHeader.appendChild(closeBtn);
        quizContainer.appendChild(sectionHeader);
        
        // Create quiz wrapper
        const quizWrapper = document.createElement('div');
        quizWrapper.className = 'quiz-wrapper';
        
        // Add each question
        quiz.questions.forEach((q, qIndex) => {
            const quizDiv = document.createElement('div');
            quizDiv.className = 'quiz-question-wrapper';
            
            let optionsHTML = '';
            q.options.forEach((option, oIndex) => {
                optionsHTML += `
                    <label class="quiz-option">
                        <input type="radio" name="quiz-${qIndex}" value="${option}">
                        <span>${option}</span>
                    </label>
                `;
            });
            
            quizDiv.innerHTML = `
                <h4>Quick Check ${qIndex + 1}:</h4>
                <p class="quiz-question">${q.question}</p>
                <div class="quiz-options">${optionsHTML}</div>
                <button class="quiz-submit-btn" data-index="${qIndex}" data-answer="${q.correct_answer}">Check Answer</button>
                <div class="quiz-feedback hidden"></div>
            `;
            
            quizWrapper.appendChild(quizDiv);
        });
        
        quizContainer.appendChild(quizWrapper);
        
        // Add event listeners to quiz buttons
        document.querySelectorAll('.quiz-submit-btn').forEach(btn => {
            btn.addEventListener('click', handleQuizAnswer);
        });
    }

    // Handle quiz answer
    function handleQuizAnswer(e) {
        const btn = e.target;
        const index = btn.dataset.index;
        const correctAnswerData = btn.dataset.answer;
        const selectedOption = document.querySelector(`input[name="quiz-${index}"]:checked`);
        
        if (!selectedOption) {
            alert('Please select an answer');
            return;
        }
        
        // Get all options for this question
        const allOptions = btn.parentElement.querySelectorAll(`input[name="quiz-${index}"]`);
        const options = Array.from(allOptions).map(opt => opt.value);
        
        // Determine correct answer - could be index or text
        let correctAnswer;
        let correctAnswerText;
        
        // Check if correctAnswerData is a number (index)
        const answerIndex = parseInt(correctAnswerData);
        if (!isNaN(answerIndex) && answerIndex >= 0 && answerIndex < options.length) {
            // It's an index
            correctAnswer = options[answerIndex];
            correctAnswerText = correctAnswer;
        } else {
            // It's the actual answer text
            correctAnswer = correctAnswerData;
            correctAnswerText = correctAnswerData;
        }
        
        const feedback = btn.parentElement.querySelector('.quiz-feedback');
        const isCorrect = selectedOption.value === correctAnswer;
        
        feedback.classList.remove('hidden');
        feedback.className = `quiz-feedback ${isCorrect ? 'correct' : 'incorrect'}`;
        feedback.textContent = isCorrect 
            ? '✓ Correct! Well done.' 
            : `✗ Incorrect. The correct answer is: ${correctAnswerText}`;
        
        // Disable quiz after submission
        btn.parentElement.querySelectorAll('input, button').forEach(el => el.disabled = true);
        btn.classList.add('disabled');
    }

    // Image upload handling
    imageUploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            currentImage = file;
            previewImg.src = URL.createObjectURL(file);
            imagePreview.classList.remove('hidden');
        }
    });

    // Remove image preview
    removeImageBtn.addEventListener('click', clearImagePreview);

    function clearImagePreview() {
        currentImage = null;
        imagePreview.classList.add('hidden');
        previewImg.src = '';
        imageUploadInput.value = '';
    }

    // Voice recording using Web Speech API (browser-based, free)
    let recognition = null;
    
    // Initialize Web Speech API
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            autoResizeTextarea();
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line></svg>';
            recordBtn.title = 'Voice Input';
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line></svg>';
            recordBtn.title = 'Voice Input';
            
            if (event.error === 'no-speech') {
                displayMessage('No speech detected. Please try again.', 'bot');
            } else if (event.error === 'not-allowed') {
                alert('Microphone access denied. Please allow microphone access in your browser settings.');
            } else {
                displayMessage('Voice recognition error. Please try again.', 'bot');
            }
        };
        
        recognition.onend = () => {
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line></svg>';
            recordBtn.title = 'Voice Input';
        };
    }
    
    recordBtn.addEventListener('click', toggleRecording);

    function toggleRecording() {
        if (!recognition) {
            alert('Voice input is not supported in your browser. Please use Chrome, Edge, or Safari.');
            return;
        }
        
        if (!isRecording) {
            try {
                recognition.start();
                isRecording = true;
                recordBtn.classList.add('recording');
                // Change to stop icon (square)
                recordBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="2"></rect></svg>';
                recordBtn.title = '🔴 Recording... Click to stop';
            } catch (error) {
                console.error('Error starting recognition:', error);
                alert('Could not start voice recognition. Please try again.');
            }
        } else {
            recognition.stop();
            isRecording = false;
            recordBtn.classList.remove('recording');
            // Change back to microphone icon
            recordBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line></svg>';
            recordBtn.title = 'Voice Input';
        }
    }

    // Model status display
    function showModelStatus(message) {
        if (modelStatus) {
            modelStatus.classList.remove('hidden');
            modelStatus.querySelector('.status-text').textContent = message;
        }
    }

    function hideModelStatus() {
        if (modelStatus) {
            modelStatus.classList.add('hidden');
        }
    }

    // Loading indicator
    function toggleLoading(isLoading) {
        loadingIndicator.classList.toggle('hidden', !isLoading);
        if (isLoading) scrollToBottom();
    }

    // Scroll to bottom
    function scrollToBottom() {
        setTimeout(() => {
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }, 100);
    }
});

