# VidyaTid Authentication Routes

## Required Flask Routes

Add these routes to your Flask application (app.py or routes/auth.py):

```python
from flask import render_template, request, jsonify, session, redirect, url_for

# Auth page route
@app.route('/auth')
def auth_page():
    """Render the login/register page"""
    return render_template('auth.html')

# Login route
@app.route('/login', methods=['POST'])
def login():
    """Handle login requests"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    remember_me = data.get('remember_me', False)
    
    # Add your authentication logic here
    # Example:
    # user = authenticate_user(username, password)
    # if user:
    #     session['user_id'] = user.id
    #     session['username'] = user.username
    #     return jsonify({'success': True}), 200
    # else:
    #     return jsonify({'error': 'Invalid credentials'}), 401
    
    return jsonify({'success': True}), 200

# Register route
@app.route('/register', methods=['POST'])
def register():
    """Handle registration requests"""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # Add your registration logic here
    # Example:
    # if user_exists(username):
    #     return jsonify({'error': 'Username already exists'}), 400
    # 
    # user = create_user(username, email, password)
    # session['user_id'] = user.id
    # session['username'] = user.username
    # return jsonify({'success': True}), 200
    
    return jsonify({'success': True}), 200

# Logout route
@app.route('/logout')
def logout():
    """Handle logout"""
    session.clear()
    return redirect(url_for('auth_page'))
```

## Updated Links

All login/signup buttons now point to `/auth`:

### Home Page (home.html)
- Navigation "Login" button → `/auth`
- Pricing "Get Started" buttons → `/auth`

### Other Pages
- All pages can access auth via `/auth` route

## Files Created

1. `templates/auth.html` - Beautiful animated login/register page
2. `static/css/auth.css` - Styling with animations and transitions
3. `static/js/auth.js` - Form handling and validation

## Features

- Animated background with floating orbs and particles
- Split-screen design with branding on left, forms on right
- Smooth transitions between login and register forms
- Password visibility toggle
- Form validation
- Loading states
- Error message display
- Fully responsive design
- Matches VidyaTid color scheme (black and blue gradients)
