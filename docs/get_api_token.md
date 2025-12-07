# 🔑 Get Your Cloudflare API Token

## ✅ Account ID Already Set!
Your Account ID is: `109da621137c33043fc277630bac4a69`

## 📝 Now Get Your API Token:

### Step 1: Open Token Creation Page
Run this command to open the page:
```powershell
Start-Process 'https://dash.cloudflare.com/profile/api-tokens/create'
```

Or manually go to: https://dash.cloudflare.com/profile/api-tokens/create

### Step 2: Create Token
1. Click **"Create Custom Token"** or use **"Edit Cloudflare Workers"** template
2. Set these permissions:
   - **Account** > **Workers AI** > **Edit** ✓
   - **Account** > **Account Settings** > **Read** ✓
3. Click **"Continue to summary"**
4. Click **"Create Token"**
5. **COPY THE TOKEN** (you won't see it again!)

### Step 3: Add Token to .env
Once you have the token, run:
```powershell
notepad .env
```

Find this line:
```
CLOUDFLARE_API_TOKEN=
```

And paste your token after the `=`:
```
CLOUDFLARE_API_TOKEN=your_token_here
```

Save and close.

### Step 4: Test It!
```powershell
python test_cloudflare_ai.py
```

---

## 🚀 Quick Commands:

```powershell
# 1. Open token creation page
Start-Process 'https://dash.cloudflare.com/profile/api-tokens/create'

# 2. After getting token, edit .env
notepad .env

# 3. Test the setup
python test_cloudflare_ai.py

# 4. Start the app
python app.py
```

---

**Your Account ID is already configured!** ✅
Just need the API Token now! 🔑
