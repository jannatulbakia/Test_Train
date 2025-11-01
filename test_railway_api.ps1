# PowerShell script to test Railway API endpoints
# Replace YOUR_RAILWAY_URL with your actual Railway URL

$RailwayUrl = "https://web-production-e5415.up.railway.app"

Write-Host "Testing Railway API endpoints..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`n1. Root endpoint (GET /)" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$RailwayUrl/" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

Write-Host "`n2. Health check (GET /health)" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$RailwayUrl/health" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

Write-Host "`n3. Get features (GET /features)" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$RailwayUrl/features" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

Write-Host "`n4. Get model info (GET /model/info)" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$RailwayUrl/model/info" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

Write-Host "`n5. Make prediction (POST /predict)" -ForegroundColor Yellow
$body = @{
    age = 35
    gender = "Male"
    city = "New York"
    session_duration = 20.5
    pages_viewed = 8
    total_clicks = 15
    cart_value = 250.50
    discount_applied = 1
    payment_method = "Credit Card"
    product_category = "Electronics"
    device_type = "Desktop"
    month = "Jan"
} | ConvertTo-Json

Invoke-WebRequest -Uri "$RailwayUrl/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body `
    -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

Write-Host "`nDone!" -ForegroundColor Green

