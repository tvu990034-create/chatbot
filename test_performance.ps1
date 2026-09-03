# Performance Testing Script for Local Chatbot
# Tests 20 prompts for speed and intelligence

# Configuration
$apiUrl = "http://localhost:8080/api/v1/chat"
$model = "ollama/gemma2:2b"
$performanceMode = "speed"
$useCache = $true

# 40 Test Prompts with Categories and Estimated Speeds
$testPrompts = @(
    @{
        Prompt = "Hello"
        Category = "Simple Greeting"
        EstimatedTime = "3-5s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is 2+2?"
        Category = "Simple Math"
        EstimatedTime = "3-6s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is the capital of France?"
        Category = "Geography"
        EstimatedTime = "4-7s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "Explain what AI is"
        Category = "Technology"
        EstimatedTime = "6-10s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "Write a haiku about coding"
        Category = "Creative Writing"
        EstimatedTime = "5-8s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What are the primary colors?"
        Category = "Basic Knowledge"
        EstimatedTime = "3-6s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "How do you make coffee?"
        Category = "Instructions"
        EstimatedTime = "5-9s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is the meaning of life?"
        Category = "Philosophy"
        EstimatedTime = "7-12s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "Tell me a joke"
        Category = "Entertainment"
        EstimatedTime = "4-7s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is Python programming?"
        Category = "Technology"
        EstimatedTime = "5-9s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "Summarize the plot of Romeo and Juliet"
        Category = "Literature"
        EstimatedTime = "8-15s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What causes seasons?"
        Category = "Science"
        EstimatedTime = "5-10s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "How does a computer work?"
        Category = "Technology"
        EstimatedTime = "8-15s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is democracy?"
        Category = "Politics"
        EstimatedTime = "6-12s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "Explain photosynthesis"
        Category = "Science"
        EstimatedTime = "7-14s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is the difference between a virus and bacteria?"
        Category = "Biology"
        EstimatedTime = "8-16s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "How do I tie my shoes?"
        Category = "Instructions"
        EstimatedTime = "5-9s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is the most spoken language in the world?"
        Category = "Geography"
        EstimatedTime = "4-8s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "Write a short story about a robot"
        Category = "Creative Writing"
        EstimatedTime = "10-20s (first), 0.001s (cached)"
        Complexity = "High"
    },
    @{
        Prompt = "What are the benefits of exercise?"
        Category = "Health"
        EstimatedTime = "6-12s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    # Additional 20 prompts
    @{
        Prompt = "What is the speed of light?"
        Category = "Science"
        EstimatedTime = "3-6s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "Who wrote Romeo and Juliet?"
        Category = "Literature"
        EstimatedTime = "3-5s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is a black hole?"
        Category = "Science"
        EstimatedTime = "5-9s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "How do plants grow?"
        Category = "Science"
        EstimatedTime = "5-10s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is the largest ocean?"
        Category = "Geography"
        EstimatedTime = "3-6s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "Who was the first person on the moon?"
        Category = "History"
        EstimatedTime = "3-5s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is DNA?"
        Category = "Biology"
        EstimatedTime = "4-8s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "How does the internet work?"
        Category = "Technology"
        EstimatedTime = "7-14s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is climate change?"
        Category = "Environment"
        EstimatedTime = "6-12s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "Who invented the telephone?"
        Category = "History"
        EstimatedTime = "3-6s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is the periodic table?"
        Category = "Chemistry"
        EstimatedTime = "4-8s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "How do birds fly?"
        Category = "Biology"
        EstimatedTime = "5-10s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is gravity?"
        Category = "Physics"
        EstimatedTime = "5-9s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "Who painted the Mona Lisa?"
        Category = "Art"
        EstimatedTime = "3-5s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is electricity?"
        Category = "Physics"
        EstimatedTime = "6-11s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "How do vaccines work?"
        Category = "Medicine"
        EstimatedTime = "6-12s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "What is the Amazon rainforest?"
        Category = "Geography"
        EstimatedTime = "4-8s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "Who discovered America?"
        Category = "History"
        EstimatedTime = "3-6s (first), 0.001s (cached)"
        Complexity = "Low"
    },
    @{
        Prompt = "What is machine learning?"
        Category = "Technology"
        EstimatedTime = "6-12s (first), 0.001s (cached)"
        Complexity = "Medium"
    },
    @{
        Prompt = "How do you bake a cake?"
        Category = "Instructions"
        EstimatedTime = "5-10s (first), 0.001s (cached)"
        Complexity = "Medium"
    }
)

Write-Host "=== Local Chatbot Performance Test ===" -ForegroundColor Cyan
Write-Host "Model: $model"
Write-Host "Performance Mode: $performanceMode"
Write-Host "Cache: $useCache"
Write-Host "API: $apiUrl"
Write-Host ""

$results = @()

foreach ($test in $testPrompts) {
    $prompt = $test.Prompt
    $category = $test.Category
    $estimated = $test.EstimatedTime
    $complexity = $test.Complexity
    
    Write-Host "Testing: $prompt" -ForegroundColor Yellow
    Write-Host "Category: $category | Complexity: $complexity"
    Write-Host "Estimated: $estimated"
    
    $body = @{
        messages = @(@{role="user";content=$prompt})
        model = $model
        provider = "ollama"
        use_cache = $useCache
        performance_mode = $performanceMode
    } | ConvertTo-Json
    
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-WebRequest -Uri $apiUrl -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
        $sw.Stop()
        
        $response = $r.Content | ConvertFrom-Json
        $actualTime = $response.duration
        $cacheHit = $response.cache_hit
        $responseText = $response.response.Substring(0, [Math]::Min(100, $response.response.Length))
        
        Write-Host "Actual Time: $actualTime seconds" -ForegroundColor Green
        Write-Host "Cache Hit: $cacheHit"
        Write-Host "Response Preview: $responseText..."
        
        $results += @{
            Prompt = $prompt
            Category = $category
            Complexity = $complexity
            EstimatedTime = $estimated
            ActualTime = $actualTime
            CacheHit = $cacheHit
            ResponseLength = $response.response.Length
        }
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
        $results += @{
            Prompt = $prompt
            Category = $category
            Complexity = $complexity
            EstimatedTime = $estimated
            ActualTime = "ERROR"
            CacheHit = "ERROR"
            ResponseLength = 0
        }
    }
    
    Write-Host "---"
    Start-Sleep -Milliseconds 500
}

# Summary
Write-Host ""
Write-Host "=== Performance Test Summary ===" -ForegroundColor Cyan
Write-Host ""

$successCount = ($results | Where-Object { $_.ActualTime -ne "ERROR" }).Count
$errorCount = ($results | Where-Object { $_.ActualTime -eq "ERROR" }).Count
$cacheHitCount = ($results | Where-Object { $_.CacheHit -eq $true }).Count

Write-Host "Total Tests: $($results.Count)"
Write-Host "Successful: $successCount"
Write-Host "Errors: $errorCount"
Write-Host "Cache Hits: $cacheHitCount"
Write-Host ""

# Average times
$numericTimes = ($results | Where-Object { $_.ActualTime -is [double] }).ActualTime
if ($numericTimes.Count -gt 0) {
    $avgTime = ($numericTimes | Measure-Object -Average).Average
    $maxTime = ($numericTimes | Measure-Object -Maximum).Maximum
    $minTime = ($numericTimes | Measure-Object -Minimum).Minimum
    
    Write-Host "Average Response Time: $([math]::Round($avgTime, 2)) seconds"
    Write-Host "Fastest Response: $([math]::Round($minTime, 2)) seconds"
    Write-Host "Slowest Response: $([math]::Round($maxTime, 2)) seconds"
}

Write-Host ""
Write-Host "Detailed Results:" -ForegroundColor Yellow
$results | Format-Table -Property Prompt, Category, Complexity, EstimatedTime, ActualTime, CacheHit