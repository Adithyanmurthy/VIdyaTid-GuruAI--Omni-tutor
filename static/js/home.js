// Check authentication state and update navigation
function updateNavigation() {
    const sessionToken = localStorage.getItem('session_token');
    const userData = localStorage.getItem('user_data');
    const authLink = document.getElementById('auth-link');
    const ctaLink = document.getElementById('cta-link');
    
    if (sessionToken && userData) {
        // User is logged in
        try {
            const user = JSON.parse(userData);
            
            // Update auth link to show Dashboard
            if (authLink) {
                authLink.textContent = 'Dashboard';
                authLink.href = '/chat';
                authLink.title = 'Go to Dashboard';
            }
            
            // Update CTA to "Dashboard" or "Logout"
            if (ctaLink) {
                ctaLink.textContent = 'Dashboard';
                ctaLink.href = '/chat';
            }
            
            // Add logout button
            const navLinks = document.getElementById('nav-links');
            if (navLinks && !document.getElementById('logout-btn')) {
                const logoutBtn = document.createElement('a');
                logoutBtn.id = 'logout-btn';
                logoutBtn.href = '#';
                logoutBtn.textContent = 'Logout';
                logoutBtn.className = 'btn-secondary';
                logoutBtn.style.marginLeft = '10px';
                logoutBtn.onclick = function(e) {
                    e.preventDefault();
                    logout();
                };
                navLinks.appendChild(logoutBtn);
            }
        } catch (e) {
            console.error('Error parsing user data:', e);
            // Clear invalid data
            localStorage.removeItem('session_token');
            localStorage.removeItem('user_data');
        }
    } else {
        // User is not logged in - show default links
        if (authLink) {
            authLink.textContent = 'Login';
            authLink.href = '/auth';
        }
        if (ctaLink) {
            ctaLink.textContent = 'Try Free';
            ctaLink.href = '/chat';
        }
        
        // Remove logout button if it exists
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.remove();
        }
    }
}

// Logout function
function logout() {
    // Clear local storage
    localStorage.removeItem('session_token');
    localStorage.removeItem('user_data');
    
    // Redirect to home page
    window.location.href = '/';
}

// Smooth scroll function
function scrollToDemo() {
    document.getElementById('how-it-works').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = 'none';
    }
});

// Continuous loop animation for "How It Works" steps
let currentStep = 0;
let stepAnimationInterval = null;
let isStepsVisible = false;

function animateSteps() {
    const stepCards = document.querySelectorAll('.step-card');
    if (stepCards.length === 0) return;
    
    // Remove active class from all
    stepCards.forEach(card => card.classList.remove('active'));
    
    // Add active class to current step
    stepCards[currentStep].classList.add('active');
    
    // Move to next step
    currentStep = (currentStep + 1) % stepCards.length;
}

// Observer for "How It Works" section
const stepsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !isStepsVisible) {
            isStepsVisible = true;
            // Start animation immediately
            animateSteps();
            // Continue animation every 3 seconds
            stepAnimationInterval = setInterval(animateSteps, 3000);
        } else if (!entry.isIntersecting && isStepsVisible) {
            isStepsVisible = false;
            // Stop animation when section is not visible
            if (stepAnimationInterval) {
                clearInterval(stepAnimationInterval);
                stepAnimationInterval = null;
            }
            // Reset to first step
            currentStep = 0;
            const stepCards = document.querySelectorAll('.step-card');
            stepCards.forEach(card => card.classList.remove('active'));
        }
    });
}, { threshold: 0.3 });

// Animate elements on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all feature cards and pricing cards
document.addEventListener('DOMContentLoaded', () => {
    // Update navigation based on auth state
    updateNavigation();
    
    const cards = document.querySelectorAll('.feature-card, .pricing-card, .stat-card');
    
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
    
    // Observe "How It Works" section for step animation
    const howItWorksSection = document.querySelector('.how-it-works');
    if (howItWorksSection) {
        stepsObserver.observe(howItWorksSection);
    }
});

// Counter animation for stats
function animateCounter(element, target, duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16);
    
    const timer = setInterval(() => {
        start += increment;
        if (start >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(start);
        }
    }, 16);
}

// Trigger counter animation when stats come into view
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
            entry.target.classList.add('animated');
            const number = entry.target.querySelector('.stat-number');
            const text = number.textContent;
            
            // Extract number from text (e.g., "10K+" -> 10)
            const value = parseInt(text.replace(/[^0-9]/g, ''));
            if (!isNaN(value)) {
                number.textContent = '0';
                setTimeout(() => {
                    animateCounter(number, value, 1500);
                    // Add back the suffix
                    setTimeout(() => {
                        number.textContent = text;
                    }, 1500);
                }, 200);
            }
        }
    });
}, { threshold: 0.5 });

document.addEventListener('DOMContentLoaded', () => {
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach(card => statsObserver.observe(card));
});
