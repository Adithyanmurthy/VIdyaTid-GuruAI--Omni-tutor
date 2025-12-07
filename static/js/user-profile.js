// User Profile Dropdown Handler

document.addEventListener('DOMContentLoaded', function() {
    const loginBtn = document.getElementById('login-btn');
    const profileDropdown = document.getElementById('user-profile-dropdown');
    const profileTrigger = document.getElementById('profile-trigger');
    const dropdownMenu = document.getElementById('dropdown-menu');
    const logoutBtn = document.getElementById('logout-btn');
    
    // Check if user is logged in
    checkAuthStatus();
    
    // Toggle dropdown
    if (profileTrigger) {
        profileTrigger.addEventListener('click', function(e) {
            e.stopPropagation();
            profileDropdown.classList.toggle('active');
        });
    }
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (profileDropdown && !profileDropdown.contains(e.target)) {
            profileDropdown.classList.remove('active');
        }
    });
    
    // Logout handler
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async function(e) {
            e.preventDefault();
            
            try {
                const response = await fetch('/api/auth/logout', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                if (response.ok) {
                    // Clear local storage
                    localStorage.removeItem('session_token');
                    localStorage.removeItem('user_data');
                    
                    // Redirect to home or auth page
                    window.location.href = '/auth';
                } else {
                    console.error('Logout failed');
                }
            } catch (error) {
                console.error('Logout error:', error);
            }
        });
    }
    
    // Check authentication status
    async function checkAuthStatus() {
        const sessionToken = localStorage.getItem('session_token');
        const userData = localStorage.getItem('user_data');
        
        if (sessionToken && userData) {
            try {
                const user = JSON.parse(userData);
                showProfileDropdown(user);
            } catch (error) {
                console.error('Error parsing user data:', error);
                showLoginButton();
            }
        } else {
            showLoginButton();
        }
    }
    
    // Show profile dropdown with user data
    function showProfileDropdown(user) {
        if (loginBtn) loginBtn.classList.add('hidden');
        if (profileDropdown) profileDropdown.classList.remove('hidden');
        
        // Update user info
        const userName = document.getElementById('user-name');
        const dropdownUsername = document.getElementById('dropdown-username');
        const dropdownEmail = document.getElementById('dropdown-email');
        
        if (userName) userName.textContent = user.username || 'User';
        if (dropdownUsername) dropdownUsername.textContent = user.username || 'User';
        if (dropdownEmail) {
            dropdownEmail.textContent = user.email || 'No email provided';
        }
        
        // Update avatar if user has one
        if (user.avatar) {
            const userAvatar = document.getElementById('user-avatar');
            const dropdownAvatar = document.querySelector('.dropdown-avatar');
            if (userAvatar) userAvatar.src = user.avatar;
            if (dropdownAvatar) dropdownAvatar.src = user.avatar;
        }
    }
    
    // Show login button
    function showLoginButton() {
        if (loginBtn) loginBtn.classList.remove('hidden');
        if (profileDropdown) profileDropdown.classList.add('hidden');
    }
});
