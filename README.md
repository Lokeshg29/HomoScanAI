# HemoScan - Anemia Detection System

AI-powered Complete Blood Count (CBC) analysis system for anemia detection with 99.5% accuracy.

## Features

- 🔐 User authentication with JWT
- 👥 Multiple patient profile management
- 📄 PDF CBC report upload with OCR
- 🤖 AI-powered anemia detection (Random Forest - 99.5% accuracy)
- 💬 AI chatbot for medical insights (powered by GROQ)
- 📊 Detailed analysis with WHO clinical guidelines
- 📈 Report history and tracking
- 🎯 Risk assessment and recommendations

## Tech Stack

**Backend:**
- Flask 3.0.0
- SQLAlchemy (SQLite)
- Scikit-learn 1.8.0
- Tesseract OCR
- JWT Authentication

**Frontend:**
- React 18.2.0
- Vite 7.3.1
- Axios
- React Router DOM

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Tesseract OCR

### Installation

1. **Install Python dependencies:**
```bash
cd HemoScan-Animea_Detection-main
pip install -r requirements.txt
```

2. **Install Frontend dependencies:**
```bash
cd hemoscan_frontend
npm install
cd ..
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your GROQ API key (optional, for chatbot)
```

4. **Install Tesseract OCR:**
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### Running the Application

**Terminal 1 - Backend:**
```bash
python application.py
```

**Terminal 2 - Frontend:**
```bash
cd hemoscan_frontend
npm run dev
```

**Access:** http://localhost:5173

## Usage

1. **Register/Login** - Create an account
2. **Create Profile** - Add patient information
3. **Upload CBC Report** - Upload PDF or enter data manually
4. **View Analysis** - See AI-powered anemia detection results
5. **Chat** - Ask questions about the results (requires GROQ API key)

## Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 99.50%
- **Features**: Gender, Age, Hb, RBC, PCV, MCV, MCH, MCHC
- **Classes**: Normal, Mild Anemia, Moderate Anemia, Severe Anemia
- **Safety**: WHO Clinical Override (Hb < 7.0 g/dL → Severe)

## Configuration

### Environment Variables (.env)

```env
# Required for chatbot
GROQAI_API_KEY=your_groq_api_key_here

# Security (change in production)
JWT_SECRET_KEY=your-secret-key

# Database
DATABASE_URI=sqlite:///hemoscan.db

# CORS
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174
```

### Get GROQ API Key (Free)

1. Visit: https://console.groq.com/keys
2. Sign up for free account
3. Create API key
4. Add to `.env` file

## Retraining the Model

If you have a new dataset:

```bash
python train_model.py
```

The script will:
- Load dataset from `C:\Users\keert\Downloads\dataset_with_severity1.csv`
- Train multiple models
- Select best performing model
- Save model, scaler, and metadata

## Project Structure

```
HemoScan-Animea_Detection-main/
├── application.py              # Flask backend
├── train_model.py             # Model training script
├── model.pkl                  # Trained ML model
├── scaler.pkl                 # Feature scaler
├── model_metadata.pkl         # Model information
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration
├── hemoscan_frontend/         # React frontend
│   ├── src/
│   │   ├── pages/            # Page components
│   │   ├── components/       # Reusable components
│   │   ├── api.jsx           # API client
│   │   └── styles.css        # Styling
│   └── package.json          # Node dependencies
└── README.md                 # This file
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/register` | POST | No | User registration |
| `/login` | POST | No | User authentication |
| `/create-profile` | POST | Yes | Create patient profile |
| `/profiles` | GET | Yes | List profiles |
| `/upload-pdf` | POST | Yes | Upload & analyze CBC PDF |
| `/predict` | POST | Yes | Manual CBC analysis |
| `/chat` | POST | Yes | AI chatbot |
| `/profile/<id>/reports` | GET | Yes | Get reports |
| `/report/<id>/chat` | GET | Yes | Get chat history |

## Troubleshooting

### 401 Unauthorized Errors
- **Solution**: Logout and login again (JWT token expired)

### Chatbot Not Working
- **Solution**: Add GROQ API key to `.env` and restart backend

### Tesseract Not Found
- **Solution**: Install Tesseract OCR and ensure it's in PATH

### Port Already in Use
- **Backend (5000)**: Kill process or change port
- **Frontend (5173)**: Vite will auto-increment to 5174

## Security Notes

⚠️ **For Development Only**
- Change `JWT_SECRET_KEY` in production
- Use PostgreSQL/MySQL instead of SQLite
- Enable HTTPS
- Set proper CORS origins
- Use environment variables for secrets

## License

See LICENSE file for details.

## Support

For issues or questions:
1. Check QUICK_START.md for detailed setup
2. Check SETUP_GUIDE.md for troubleshooting
3. Review .env.example for configuration options

---

**Version**: 2.0
**Model Accuracy**: 99.50%
**Last Updated**: 2026-02-19
