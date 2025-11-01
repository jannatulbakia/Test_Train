# Railway Deployment - Quick Start 🚀

## Quick Steps to Deploy

### Step 1: Prepare Your Repository

Make sure all files are ready:
```bash
# Check you have all required files
ls *.pkl  # Should show: best_model.pkl, scaler.pkl, label_encoders.pkl, feature_columns.pkl
```

### Step 2: Initialize Git (if not done)

```bash
git init
git add .
git commit -m "Initial commit for Railway deployment"
```

### Step 3: Push to GitHub

1. Create a new repository on GitHub
2. Push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Railway

#### Option A: Via Railway Website (Easiest)

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository
5. Railway will automatically:
   - Detect Python
   - Install dependencies
   - Deploy your app

#### Option B: Via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up
```

### Step 5: Get Your API URL

1. Go to your Railway project
2. Click on your service
3. Go to **Settings** → **Networking**
4. Click **"Generate Domain"**
5. Copy the URL (e.g., `https://your-app.railway.app`)

### Step 6: Test Your API

```bash
# Test health endpoint
curl https://your-app.railway.app/health

# Test prediction
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "city": "New York",
    "session_duration": 20.5,
    "pages_viewed": 8,
    "total_clicks": 15,
    "cart_value": 250.50,
    "discount_applied": 1,
    "payment_method": "Credit Card",
    "product_category": "Electronics",
    "device_type": "Desktop",
    "month": "Jan"
  }'
```

### Step 7: Update Your React App

Update your React project's `.env` file:
```
REACT_APP_API_URL=https://your-app.railway.app
```

## ✅ Checklist

- [ ] All `.pkl` files are in repository
- [ ] Code is pushed to GitHub
- [ ] Railway account created
- [ ] Project deployed on Railway
- [ ] API URL obtained
- [ ] API tested and working
- [ ] React app updated with API URL

## 🐛 Common Issues

**Issue**: Model files not found
- **Fix**: Make sure `*.pkl` files are not in `.gitignore`

**Issue**: Build fails
- **Fix**: Check `requirements.txt` has all dependencies

**Issue**: App doesn't start
- **Fix**: Verify `Procfile` has correct command: `web: gunicorn app:app`

For detailed instructions, see `RAILWAY_DEPLOYMENT.md`

