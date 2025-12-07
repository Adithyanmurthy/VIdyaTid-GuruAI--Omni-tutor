/**
 * User Profile Dropdown Component
 * Handles user profile display, dropdown menu, and logout functionality
 */

(function() {
    'use strict';

    // Initialize user profile dropdown
    function initUserProfileDropdown() {
        const loginBtn = document.getElementById('login-btn');
        const profileDropdown = document.getElementById('user-profile-dropdown');
        const profileTrigger = document.getElementById('profile-trigger');
        const dropdownMenu = document.getElementById('dropdown-menu');
        const logoutBtn = document.getElementById('logout-btn');
        const userAvatar = document.getElementById('user-avatar');
        const userName = document.getElementById('user-name');
        const dropdownUsername = document.getElementById('dropdown-username');
        const dropdownEmail = document.getElementById('dropdown-email');

        // Check authentication status
        const sessionToken = localStorage.getItem('session_token');
        const userDataStr = localStorage.getItem('user_data');
        
        if (sessionToken && userDataStr) {
            try {
                const userData = JSON.parse(userDataStr);
                
                // Hide login button, show profile dropdown
                if (loginBtn) loginBtn.classList.add('hidden');
                if (profileDropdown) profileDropdown.classList.remove('hidden');
                
                // Set user information
                const displayName = userData.username || userData.email || 'User';
                const email = userData.email || '';
                
                if (userName) userName.textContent = displayName;
                if (dropdownUsername) dropdownUsername.textContent = displayName;
                if (dropdownEmail) dropdownEmail.textContent = email;
                
                // Set user avatar (use first letter of username as fallback)
                if (userAvatar) {
                    if (userData.avatar_url) {
                        userAvatar.src = userData.avatar_url;
                    } else {
                        // Create a colored circle with first letter
                        const firstLetter = displayName.charAt(0).toUpperCase();
                        const canvas = document.createElement('canvas');
                        canvas.width = 40;
                        canvas.height = 40;
                        const ctx = canvas.getContext('2d');
                        
                        // Generate color based on username
                        const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a'];
                        const colorIndex = displayName.charCodeAt(0) % colors.length;
                        
                        ctx.fillStyle = colors[colorIndex];
                        ctx.fillRect(0, 0, 40, 40);
                        ctx.fillStyle = '#ffffff';
                        ctx.font = 'bold 20px Inter, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(firstLetter, 20, 20);
                        
                        userAvatar.src = canvas.toDataURL();
                    }
                }
                
                // Copy avatar to dropdown
                const dropdownAvatar = document.querySelector('.dropdown-avatar');
                if (dropdownAvatar && userAvatar) {
                    dropdownAvatar.src = userAvatar.src;
                }
                
            } catch (e) {
                console.error('Error parsing user data:', e);
                showLoginButton();
            }
        } else {
            showLoginButton();
        }
        
        // Toggle dropdown menu
        if (profileTrigger && dropdownMenu) {
            profileTrigger.addEventListener('click', function(e) {
                e.stopPropagation();
                dropdownMenu.classList.toggle('show');
            });
            
            // Close dropdown when clicking outside
            document.addEventListener('click', function(e) {
                if (!profileDropdown.contains(e.target)) {
                    dropdownMenu.classList.remove('show');
                }
            });
        }
        
        // Handle logout
        if (logoutBtn) {
            logoutBtn.addEventListener('click', function(e) {
                e.preventDefault();
                handleLogout();
            });
        }
    }
    
    function showLoginButton() {
        const loginBtn = document.getElementById('login-btn');
        const profileDropdown = document.getElementById('user-profile-dropdown');
        
        if (loginBtn) loginBtn.classList.remove('hidden');
        if (profileDropdown) profileDropdown.classList.add('hidden');
    }
    
    function handleLogout() {
        // Clear authentication data
        localStorage.removeItem('session_token');
        localStorage.removeItem('user_data');
        
        // Show confirmation message
        const confirmed = confirm('Are you sure you want to logout?');
        if (confirmed) {
            // Redirect to home page
            window.location.href = '/';
        }
    }
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initUserProfileDropdown);
    } else {
        initUserProfileDropdown();
    }
    
    // Export for external use
    window.userProfileDropdown = {
        init: initUserProfileDropdown,
        logout: handleLogout
    };
})();
