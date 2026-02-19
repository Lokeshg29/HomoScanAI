# Setup Guide

## Quick Setup (5 Minutes)

### 1. Install Python Packages
```bash
pip install flask flask-cors flask-sqlalchemy flask-bcrypt flask-jwt-extended pandas numpy scikit-learn pdfplumber pytesseract pdf2image pillow requests
```

### 2. Install Tesseract OCR

**Windows:**
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Install to: `C:\Program Files\Tesseract-OCR\`

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 3. Set API Key (Optional - for AI chat)
```bash
# Windows
set GROQAI_API_KEY=your_key_here

# Mac/Linux
export GROQAI_API_KEY=your_key_here
```

### 4. Start Backend
```bash
python application.py
```

You should see: `Running on http://127.0.0.1:5000`

### 5. Start Frontend (New Terminal)
```bash
cd hemoscan_frontend
npm install
npm run dev
```

You should see: `Local: http://localhost:5173`

### 6. Open Browser
Go to: `http://localhost:5173`

## First Time Use

1. Click "Register" and create an account
2. Login with your credentials
3. Upload a blood test PDF or enter values manually
4. Click "Analyze" to see results!

## Troubleshooting

**Problem: "Tesseract not found"**
- Make sure Tesseract is installed
- Check the path in `application.py` line 40

**Problem: "Cannot connect to backend"**
- Make sure backend is running on port 5000
- Check if another app is using port 5000

**Problem: "npm install fails"**
- Make sure Node.js 16+ is installed
- Try: `npm cache clean --force` then `npm install` again

## That's It!

You're ready to analyze blood tests! 🎉
