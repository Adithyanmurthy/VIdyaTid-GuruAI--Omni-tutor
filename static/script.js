document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');
    const loadingIndicator = document.getElementById('loading-indicator');
    const recordBtn = document.getElementById('record-btn');
    const imageUploadInput = document.getElementById('image-upload');

    // --- JEE/NEET Specific Customizations ---
    function initializeChat() {
        // Clear any hardcoded HTML messages
        chatWindow.innerHTML = '';
        
        // Set a more targeted placeholder
        userInput.placeholder = "Ask about Rotational Motion, P-Block Elements...";

        // Display the new welcome message
        const welcomeMessage = "Gearing up for JEE or NEET? I'm here to help. Ask a tough concept from Physics, Chemistry, Maths, or Biology to get started!";
        displayMessage(welcomeMessage, 'bot');
    }

    initializeChat();
    // --- End of Customizations ---

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        displayMessage(query, 'user');
        userInput.value = '';
        toggleLoading(true);

        try {
            const formData = new FormData();
            formData.append('query', query);

            const response = await fetch('/api/ask', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Network response was not ok');
            }

            const data = await response.json();
            
            // Display explanation
            displayMessage(data.explanation, 'bot');
            
            // Display quiz if it exists
            if (data.quiz) {
                displayQuiz(data.quiz);
            }

        } catch (error) {
            console.error('Error:', error);
            displayMessage(`Oops! I'm having trouble connecting. Don't let this interrupt your study flow. Please check your network and try again.`, 'bot');
        } finally {
            toggleLoading(false);
        }
    });

    function displayMessage(message, sender) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message ${sender}-message message-animated`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = `<p>${message}</p>`; // Use innerHTML to render bold/other tags
        
        messageWrapper.appendChild(messageContent);
        chatWindow.appendChild(messageWrapper);
        scrollToBottom();
    }
    
    function displayQuiz(quiz) {
        const quizWrapper = document.createElement('div');
        quizWrapper.className = 'message bot-message message-animated';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        let optionsHTML = '';
        quiz.options.forEach((option, index) => {
            optionsHTML += `
                <label>
                    <input type="radio" name="quiz-option" value="${option}">
                    <span>${option}</span>
                </label>
            `;
        });

        messageContent.innerHTML = `
            <div class="quiz-container">
                <p class="quiz-question">${quiz.question}</p>
                <div class="quiz-options">${optionsHTML}</div>
                <button class="quiz-submit">Check Answer</button>
            </div>
        `;
        
        quizWrapper.appendChild(messageContent);
        chatWindow.appendChild(quizWrapper);
        
        const submitButton = quizWrapper.querySelector('.quiz-submit');
        submitButton.addEventListener('click', () => {
            const selectedOption = quizWrapper.querySelector('input[name="quiz-option"]:checked');
            if (selectedOption) {
                if(selectedOption.value === quiz.answer) {
                    displayMessage("That's correct! A solid concept, crucial for NEET/JEE. Well done.", 'bot');
                } else {
                    displayMessage(`Not quite. The correct answer is "${quiz.answer}". Understanding this distinction is key for competitive exams. Let's try another one!`, 'bot');
                }
                // Disable quiz after submission
                quizWrapper.querySelectorAll('input, button').forEach(el => el.disabled = true);
            } else {
                alert("Please select an option.");
            }
        });
        scrollToBottom();
    }

    function toggleLoading(isLoading) {
        loadingIndicator.classList.toggle('hidden', !isLoading);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
    
    // --- Placeholder Functionality ---
    recordBtn.addEventListener('click', () => {
        alert("Voice input is coming soon! You'll be able to ask complex Physics and Chemistry questions hands-free.");
    });

    imageUploadInput.addEventListener('change', (event) => {
         if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            alert(`Image selected: ${file.name}. \n\nIn the full version, I'll solve this Maths problem or Biology diagram for you!`);
            // You would then create a FormData object with the image and send it
        }
    });

});

