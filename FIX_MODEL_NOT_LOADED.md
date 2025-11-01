# Fix: Model Not Loaded Error on Railway

If you're getting `"error": "Model not loaded"` on Railway, the model files (`.pkl` files) are likely not committed to your Git repository or not being deployed to Railway.

## ✅ Quick Fix Steps

### Step 1: Check if Model Files are in Git

Run this command to see if `.pkl` files are tracked:

```bash
git ls-files | grep "\.pkl$"
```

If you see:
- `best_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `feature_columns.pkl`

Then files are tracked. If not, they need to be added.

### Step 2: Ensure Files Are NOT in .gitignore

Check your `.gitignore` file. Make sure `*.pkl` is **NOT** uncommented (it should be commented out with `#`).

Your `.gitignore` should have:
```
# Model files (optional - uncomment if you don't want to commit model files)
# *.pkl
```

**NOT**:
```
*.pkl  ❌ This will ignore all .pkl files
```

### Step 3: Add and Commit Model Files

If files are not tracked, add them:

```bash
# Check current status
git status

# Add all .pkl files
git add *.pkl

# Or add specific files
git add best_model.pkl
git add scaler.pkl
git add label_encoders.pkl
git add feature_columns.pkl

# Commit
git commit -m "Add model files for Railway deployment"

# Push to GitHub
git push
```

### Step 4: Verify Files are Pushed

Check your GitHub repository to confirm `.pkl` files are there:
- Go to your GitHub repo
- Look for `best_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_columns.pkl`
- If they're missing, they weren't pushed

### Step 5: Redeploy on Railway

After pushing to GitHub:
1. Railway should auto-redeploy
2. Or manually trigger a deploy in Railway dashboard
3. Check the build logs - you should see:
   ```
   ✓ Model loaded
   ✓ Scaler loaded
   ✓ Label encoders loaded
   ✓ Feature columns loaded
   ```

## 🔍 Troubleshooting

### Issue: Files are too large (>100MB)

If Git won't push large files:

**Option 1: Use Git LFS**
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add *.pkl
git commit -m "Add model files with Git LFS"
git push
```

**Option 2: Check Railway build logs**

In Railway dashboard:
1. Go to your project
2. Click on the latest deployment
3. Check the build logs
4. Look for:
   - "Current directory: ..."
   - "Files in current directory: ..."
   - Any error messages about missing files

### Issue: Files exist but still not loading

Check Railway logs for error messages:
1. Go to Railway dashboard
2. Click on your service
3. Click "View Logs"
4. Look for error messages

The updated `app.py` will now print:
- Current directory
- List of files in directory
- Which files are missing
- Detailed error messages

## ✅ Verification

After deploying, test:

```bash
# Health check (should show model_loaded: true)
curl https://your-app.railway.app/health

# Features (should work)
curl https://your-app.railway.app/features
```

If `model_loaded` is still `false`, check Railway logs for the detailed error message.

## 📋 Checklist

- [ ] `.pkl` files are NOT in `.gitignore`
- [ ] `.pkl` files are committed to Git (`git ls-files | grep .pkl`)
- [ ] `.pkl` files are visible on GitHub
- [ ] Files are pushed to main/master branch
- [ ] Railway has redeployed after push
- [ ] Railway logs show files being found
- [ ] `/health` endpoint shows `model_loaded: true`

## 🚀 Quick Command Reference

```bash
# Check if files are tracked
git ls-files | grep "\.pkl$"

# Add all model files
git add *.pkl

# Commit
git commit -m "Add model files"

# Push
git push

# Check what's in your repository
git ls-tree -r HEAD --name-only | grep "\.pkl$"
```

---

Once files are committed and pushed, Railway will redeploy and the model should load successfully!

