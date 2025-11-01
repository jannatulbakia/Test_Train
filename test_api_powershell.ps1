# PowerShell script to test Railway API endpoints

$baseUrl = "https://web-production-e5415.up.railway.app"

Write-Host "=" -NoNewline
Write-Host ("=" * 79)
Write-Host "Testing Railway API Endpoints"
Write-Host "=" -NoNewline
Write-Host ("=" * 79)
Write-Host ""

# Test Health Endpoint
Write-Host "[1] Testing /health endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "$baseUrl/health" -UseBasicParsing
    Write-Host "✓ Status Code: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Yellow
    $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test Features Endpoint
Write-Host "[2] Testing /features endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "$baseUrl/features" -UseBasicParsing
    Write-Host "✓ Status Code: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Yellow
    $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test Model Info Endpoint
Write-Host "[3] Testing /model/info endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "$baseUrl/model/info" -UseBasicParsing
    Write-Host "✓ Status Code: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Yellow
    $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test Predict Endpoint (POST)
Write-Host "[4] Testing /predict endpoint..." -ForegroundColor Cyan
$predictionData = @{
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

try {
    $response = Invoke-WebRequest -Uri "$baseUrl/predict" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $predictionData `
        -UseBasicParsing
    Write-Host "✓ Status Code: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Yellow
    $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response Body:" -ForegroundColor Red
        Write-Host $responseBody
    }
}
Write-Host ""

Write-Host "=" -NoNewline
Write-Host ("=" * 79)
Write-Host "Testing Complete!"
Write-Host "=" -NoNewline
Write-Host ("=" * 79)

