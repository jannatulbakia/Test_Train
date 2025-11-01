# Railway Deployment Guide

Step-by-step guide to deploy your ML Model API to Railway.

## 📋 Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Git Repository**: Your code should be in a Git repository
3. **Model Files**: Ensure all `.pkl` files are in the repository

## 🚀 Deployment Steps

### Method 1: Deploy via Railway CLI

#### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

Or using other package managers:
```bash
# Using Homebrew (Mac)
brew install railway

# Using Scoop (Windows)
scoop bucket add railway https://github.com/railwayapp/homebrew-railway
scoop install railway
```

#### Step 2: Login to Railway

```bash
railway login
```

This will open your browser to authenticate.

#### Step 3: Initialize Railway Project

```bash
railway init
```

This will:
- Create a new Railway project
- Link your local directory to the project

#### Step 4: Deploy

```bash
railway up
```

This will:
- Build your application
- Deploy it to Railway
- Provide you with a URL

### Method 2: Deploy via GitHub (Recommended)

#### Step 1: Push Code to GitHub

1. Create a new GitHub repository
2. Initialize git in your project (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Railway deployment"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main
   ```

#### Step 2: Connect GitHub to Railway

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will automatically detect it's a Python app

#### Step 3: Configure Build Settings

Railway will automatically:
- Detect Python from `requirements.txt`
- Use the `Procfile` to start the app
- Build and deploy your application

**Note**: Make sure your `Procfile` contains:
```
web: gunicorn app:app
```

#### Step 4: Set Environment Variables (Optional)

If needed, set environment variables in Railway dashboard:
- Go to your project settings
- Click "Variables"
- Add any required environment variables

#### Step 5: Deploy

Railway will automatically deploy when you push to the main branch. Or deploy manually:
1. Go to your project dashboard
2. Click "Deploy"

## 📁 Required Files for Railway

Make sure these files are in your repository:

```
├── app.py                 # Flask API server
├── requirements.txt       # Python dependencies
├── Procfile              # Process command (web: gunicorn app:app)
├── railway.json          # Railway configuration (optional)
├── best_model.pkl        # Trained model
├── scaler.pkl            # Feature scaler
├── label_encoders.pkl    # Categorical encoders
└── feature_columns.pkl   # Feature names
```

## 🔍 Important Configuration

### Port Configuration

The `app.py` file is configured to use the `PORT` environment variable:
- Railway automatically sets the `PORT` variable
- The app will use `PORT` if available, otherwise defaults to 5000
- This ensures it works both locally and on Railway

### Model Files

**Important**: Make sure all `.pkl` files are committed to Git:
- `best_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `feature_columns.pkl`

If your model files are large (>100MB), consider:
- Using Git LFS: `git lfs track "*.pkl"`
- Or storing them in cloud storage and downloading during deployment

### Build Process

Railway will:
1. Install Python dependencies from `requirements.txt`
2. Build your application
3. Run the command from `Procfile`
4. Start your API server

## 🌐 Getting Your API URL

After deployment:

1. Go to your Railway project dashboard
2. Click on your service
3. Go to "Settings" → "Networking"
4. Click "Generate Domain" or use your custom domain
5. Your API will be available at: `https://your-app-name.railway.app`

## 🧪 Testing Deployment

After deployment, test your API:

```bash
# Health check
curl https://your-app-name.railway.app/health

# Make a prediction
curl -X POST https://your-app-name.railway.app/predict \
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

## 🔧 Troubleshooting

### Build Fails

**Issue**: Build fails during dependency installation
- **Solution**: Check `requirements.txt` for correct package versions
- Ensure all packages are available on PyPI

### Model Not Found Error

**Issue**: API returns "Model not loaded" error
- **Solution**: 
  - Verify all `.pkl` files are in the repository
  - Check that files are committed to Git
  - Ensure files are in the root directory

### Port Issues

**Issue**: App doesn't start
- **Solution**: Verify `app.py` uses `os.environ.get('PORT', 5000)`
- Check `Procfile` has correct command

### CORS Errors

**Issue**: CORS errors from React app
- **Solution**: 
  - Verify `flask-cors` is installed
  - Check `CORS(app)` is in `app.py`
  - Update React app's API URL to Railway URL

### Large File Size

**Issue**: Git won't push large `.pkl` files
- **Solution**: 
  ```bash
  git lfs install
  git lfs track "*.pkl"
  git add .gitattributes
  git add *.pkl
  git commit -m "Add model files with LFS"
  ```

## 📝 Deployment Checklist

Before deploying:

- [ ] All `.pkl` model files are in repository
- [ ] `requirements.txt` has all dependencies
- [ ] `Procfile` exists with correct command
- [ ] `app.py` uses `PORT` environment variable
- [ ] Code is pushed to GitHub (if using GitHub deployment)
- [ ] Railway CLI is installed (if using CLI method)
- [ ] Railway account is set up

After deployment:

- [ ] API responds to `/health` endpoint
- [ ] API makes predictions successfully
- [ ] API URL is configured in React app
- [ ] CORS is working (test from React app)

## 🎯 Update React App

After Railway deployment, update your React app:

1. Create `.env` file in React project:
   ```
   REACT_APP_API_URL=https://your-app-name.railway.app
   ```

2. Or update your API service file:
   ```javascript
   const API_URL = process.env.REACT_APP_API_URL || 'https://your-app-name.railway.app';
   ```

3. Test the connection from your React app

## 💰 Railway Pricing

Railway offers:
- **Free tier**: $5 credit per month (great for testing)
- **Pay as you go**: Pay only for what you use
- Check [railway.app/pricing](https://railway.app/pricing) for current plans

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [Railway Python Guide](https://docs.railway.app/deploy/python)
- [Railway CLI Reference](https://docs.railway.app/develop/cli)

---

Happy Deploying! 🚀

