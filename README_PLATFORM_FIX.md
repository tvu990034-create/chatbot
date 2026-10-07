# README Platform Separation Fix

## Issue

User accidentally ran macOS/Linux `wget` command in Windows PowerShell, causing an error. The README had both platform commands in the same code block, making it confusing which commands to use.

## Fix Applied

Separated platform-specific commands into clear sections:

### Before (Confusing)
```bash
# Download model (Windows PowerShell)
Invoke-WebRequest -Uri "..." -OutFile "models/..."

# Or on macOS/Linux
wget https://... -P models/
```

### After (Clear)
**Windows PowerShell:**
```powershell
# Create models directory (may already exist - ignore error)
New-Item -ItemType Directory -Path models -ErrorAction SilentlyContinue

# Download model
Invoke-WebRequest -Uri "https://..." -OutFile "models/..."
```

**macOS/Linux:**
```bash
# Create models directory
mkdir -p models

# Download model
wget https://... -P models/
```

## Changes Made

1. **Step 2 (Model Download)**: Separated into Windows and macOS/Linux sections
2. **Step 3 (Dependencies)**: Separated into Windows and macOS/Linux sections
3. **Step 4 (Environment)**: Separated into Windows and macOS/Linux sections
4. **Web Server**: Separated into Windows and macOS/Linux sections
5. **Added Warning**: Added note at top: "Use only the commands for your platform!"
6. **Fixed mkdir**: Changed to `New-Item -ErrorAction SilentlyContinue` to handle existing directories

## Verification

Tested all platform-specific commands in Windows PowerShell:

| Command | Status | Result |
|---------|--------|--------|
| New-Item (mkdir) | ✅ Works | Handles existing directories |
| Copy-Item (.env) | ✅ Works | Creates .env file |
| Virtual env activation | ✅ Works | Environment activated |
| CLI (basic) | ✅ Works | Starts and exits cleanly |
| CLI (optimized) | ✅ Works | Starts and exits cleanly |
| Web server | ✅ Works | Starts successfully |

## Commit

**Commit**: `be39545` - Fix README to separate Windows and macOS/Linux commands

**Pushed**: ✅ Yes

## Impact

Users will no longer accidentally run wrong platform commands. The README is now clearer and less error-prone.
