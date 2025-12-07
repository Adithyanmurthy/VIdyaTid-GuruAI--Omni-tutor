// VidyaTid Auth Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Form containers
    const loginContainer = document.getElementById('login-form-container');
    const registerContainer = document.getElementById('register-form-container');
    
    // Switch buttons
    const showRegisterBtn = document.getElementById('show-register');
    const showLoginBtn = document.getElementById('show-login');
    
    // Forms
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    
    // Toggle password visibility
    const togglePasswordButtons = document.querySelectorAll('.toggle-password');
    
    togglePasswordButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const input = document.getElementById(targetId);
            const eyeOpen = this.querySelector('.eye-open');
            const eyeClosed = this.querySelector('.eye-closed');
            
            if (input.type === 'password') {
                input.type = 'text';
                eyeOpen.classList.add('hidden');
                eyeClosed.classList.remove('hidden');
            } else {
                input.type = 'password';
                eyeOpen.classList.remove('hidden');
                eyeClosed.classList.add('hidden');
            }
        });
    });
    
    // Switch to register form
    showRegisterBtn.addEventListener('click', function(e) {
        e.preventDefault();
        loginContainer.classList.remove('active');
        registerContainer.classList.add('active');
    });
    
    // Switch to login form
    showLoginBtn.addEventListener('click', function(e) {
        e.preventDefault();
        registerContainer.classList.remove('active');
        loginContainer.classList.add('active');
    });
    
    // Login form submission
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const email = document.getElementById('login-username').value;  // Field is named login-username but contains email
        const password = document.getElementById('login-password').value;
        const rememberMe = document.getElementById('remember-me').checked;
        const errorDiv = document.getElementById('login-error');
        const submitBtn = this.querySelector('.submit-btn');
        const btnText = submitBtn.querySelector('.btn-text');
        const btnLoader = submitBtn.querySelector('.btn-loader');
        
        // Show loading state
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        submitBtn.disabled = true;
        errorDiv.classList.add('hidden');
        
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: email,  // Send as email instead of username
                    password: password,
                    remember_me: rememberMe
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Store session token and user data in localStorage
                if (data.session && data.session.session_token) {
                    localStorage.setItem('session_token', data.session.session_token);
                }
                
                if (data.user) {
                    localStorage.setItem('user_data', JSON.stringify({
                        user_id: data.user.user_id,
                        username: data.user.username,
                        email: data.user.email || '',
                        avatar: data.user.avatar || null
                    }));
                }
                
                // Success - redirect to intended page or chat
                const redirectTo = sessionStorage.getItem('redirect_after_login') || '/chat';
                sessionStorage.removeItem('redirect_after_login');
                window.location.href = redirectTo;
            } else {
                // Show error - handle both string and object error formats
                let errorMessage = 'Login failed. Please try again.';
                if (data.error) {
                    if (typeof data.error === 'string') {
                        errorMessage = data.error;
                    } else if (data.error.message) {
                        errorMessage = data.error.message;
                    }
                }
                errorDiv.textContent = errorMessage;
                errorDiv.classList.remove('hidden');
            }
        } catch (error) {
            errorDiv.textContent = 'Network error. Please check your connection.';
            errorDiv.classList.remove('hidden');
        } finally {
            // Reset button state
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });
    
    // Register form submission
    registerForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const username = document.getElementById('register-username').value;
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;
        const confirmPassword = document.getElementById('register-confirm-password').value;
        const errorDiv = document.getElementById('register-error');
        const submitBtn = this.querySelector('.submit-btn');
        const btnText = submitBtn.querySelector('.btn-text');
        const btnLoader = submitBtn.querySelector('.btn-loader');
        
        // Validate passwords match
        if (password !== confirmPassword) {
            errorDiv.textContent = 'Passwords do not match!';
            errorDiv.classList.remove('hidden');
            return;
        }
        
        // Show loading state
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        submitBtn.disabled = true;
        errorDiv.classList.add('hidden');
        
        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    email: email || null,
                    password: password
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Registration successful - show success message and switch to login
                const successDiv = document.createElement('div');
                successDiv.className = 'success-message';
                successDiv.style.cssText = 'background: #10b981; color: white; padding: 12px; border-radius: 8px; margin-bottom: 16px; text-align: center;';
                successDiv.textContent = 'Registration successful! Please login to continue.';
                
                // Insert success message
                registerForm.insertBefore(successDiv, registerForm.firstChild);
                
                // Clear form
                registerForm.reset();
                
                // Switch to login form after 2 seconds
                setTimeout(() => {
                    registerContainer.classList.remove('active');
                    loginContainer.classList.add('active');
                    
                    // Pre-fill email in login form (use email if provided, otherwise username)
                    document.getElementById('login-username').value = email || username;
                }, 2000);
            } else {
                // Show error - handle both string and object error formats
                let errorMessage = 'Registration failed. Please try again.';
                if (data.error) {
                    if (typeof data.error === 'string') {
                        errorMessage = data.error;
                    } else if (data.error.message) {
                        errorMessage = data.error.message;
                    }
                }
                errorDiv.textContent = errorMessage;
                errorDiv.classList.remove('hidden');
            }
        } catch (error) {
            errorDiv.textContent = 'Network error. Please check your connection.';
            errorDiv.classList.remove('hidden');
        } finally {
            // Reset button state
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });
    
    // Input validation and animations
    const inputs = document.querySelectorAll('input[type="text"], input[type="email"], input[type="password"]');
    
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });
});
