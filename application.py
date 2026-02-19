from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import requests
import os
import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from sklearn.impute import SimpleImputer
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from dotenv import load_dotenv
import platform

# Load environment variables
load_dotenv()



# ==============================
# APP CONFIG
# ==============================
app = Flask(__name__)

# CORS Configuration - use environment variable or default to localhost
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:5174").split(",")
CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True
)

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///hemoscan.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-this")

# Warn if using default JWT secret
if app.config["JWT_SECRET_KEY"] == "super-secret-key-change-this":
    print("\n⚠️  WARNING: Using default JWT secret key. Set JWT_SECRET_KEY environment variable for production!\n")

jwt = JWTManager(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Tesseract Configuration - cross-platform support
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if not TESSERACT_CMD:
    # Auto-detect based on platform
    system = platform.system()
    if system == "Windows":
        TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    elif system == "Darwin":  # macOS
        TESSERACT_CMD = "/usr/local/bin/tesseract"
    else:  # Linux
        TESSERACT_CMD = "/usr/bin/tesseract"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
print(f"✅ Tesseract path: {TESSERACT_CMD}")

# =========================
# USERS TABLE
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    profiles = db.relationship('Profile', backref='user', lazy=True)


# =========================
# PROFILES TABLE
# =========================
class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    profile_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    reports = db.relationship('Report', backref='profile', lazy=True)


# =========================
# REPORTS TABLE
# =========================
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey('profile.id'), nullable=False)

    patient_name = db.Column(db.String(120))   # ✅ ADD THIS

    severity = db.Column(db.String(50))
    risk_percentage = db.Column(db.Float)

    hb = db.Column(db.Float)
    rbc = db.Column(db.Float)
    pcv = db.Column(db.Float)
    mcv = db.Column(db.Float)
    mch = db.Column(db.Float)
    mchc = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

    chat_messages = db.relationship('ChatMessage', backref='report', lazy=True)



# =========================
# CHAT MESSAGES TABLE
# =========================
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    role = db.Column(db.String(10))  # user or ai
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ==============================
# LOAD MODEL + SCALER + METADATA
# ==============================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_metadata = pickle.load(open("model_metadata.pkl", "rb"))
imputer = SimpleImputer(strategy="mean")

# Check if model needs scaling
uses_scaling = model_metadata.get("uses_scaling", False)
print(f"✅ Model loaded: {model_metadata.get('model_name', 'Unknown')}")
print(f"   Uses scaling: {uses_scaling}")

# ==============================
# GROQ API KEY
# ==============================
API_KEY = os.getenv("GROQAI_API_KEY")

# Check if API key is set
if not API_KEY:
    print("\n" + "="*60)
    print("⚠️  WARNING: GROQ API KEY NOT SET!")
    print("="*60)
    print("The chat feature will not work without a valid API key.")
    print("\nYour environment variable GROQAI_API_KEY is not set.")
    print("="*60 + "\n")
else:
    print(f"\n✅ GROQ API Key loaded: {API_KEY[:10]}...{API_KEY[-4:]}\n")


# =====================================================
# CLINICAL OVERRIDE FUNCTION (WHO GUIDELINES + ENHANCED)
# =====================================================
def clinical_override(hb_value, mcv_value, mch_value, model_prediction):
    """
    Enhanced Clinical Override with WHO Guidelines
    
    WHO Hemoglobin Thresholds:
    - Severe Anemia: Hb < 7.0 g/dL
    - Moderate Anemia: Hb 7.0-9.9 g/dL
    - Mild Anemia: Hb 10.0-11.9 g/dL (women), 10.0-12.9 g/dL (men)
    - Normal: Hb >= 12.0 g/dL (women), >= 13.0 g/dL (men)
    
    This overrides ML model prediction for medical safety
    """
    # Critical threshold - Severe Anemia
    if hb_value < 7.0:
        return 3  # Force Severe classification
    
    # Moderate to Severe threshold
    elif hb_value < 8.0:
        # If model predicts mild or normal, upgrade to at least moderate
        if model_prediction < 2:
            return 2  # Force Moderate
        return max(model_prediction, 2)  # At least Moderate
    
    # Moderate Anemia threshold
    elif hb_value < 10.0:
        # If model predicts normal, upgrade to at least mild
        if model_prediction == 0:
            return 1  # Force Mild
        # If model predicts severe but Hb is not that low, downgrade to moderate
        if model_prediction == 3:
            return 2  # Force Moderate
        return model_prediction
    
    # Mild Anemia threshold
    elif hb_value < 12.0:
        # If model predicts severe or moderate, downgrade to mild
        if model_prediction >= 2:
            return 1  # Force Mild
        return max(model_prediction, 0)  # At least check if normal
    
    # Normal range
    else:
        # If Hb is normal but model predicts severe/moderate, trust Hb more
        if model_prediction >= 2 and hb_value >= 12.5:
            return 0  # Force Normal if Hb is clearly normal
        return model_prediction


# =====================================================
# COMMON FUNCTION → CLINICAL INTERPRETATION
# =====================================================
def process_cbc_data(data_dict):
    # Ensure all required features are present with defaults
    required_features = ["Gender", "Age", "Hb", "RBC", "PCV", "MCV", "MCH", "MCHC"]
    
    # Fill missing values with reasonable defaults
    defaults = {
        "Gender": 1,
        "Age": 30,
        "Hb": 13.0,
        "RBC": 4.5,
        "PCV": 40.0,
        "MCV": 85.0,
        "MCH": 28.0,
        "MCHC": 32.0
    }
    
    # Create complete data dict with defaults for missing values
    complete_data = {}
    for feature in required_features:
        if feature in data_dict and data_dict[feature] is not None:
            # Convert to native Python types to avoid JSON serialization issues
            value = data_dict[feature]
            if isinstance(value, (int, float)):
                complete_data[feature] = float(value) if feature not in ["Gender", "Age"] else int(value)
            else:
                complete_data[feature] = value
        else:
            complete_data[feature] = defaults[feature]
    
    features = pd.DataFrame([complete_data])

    # Scale features only if model requires it
    if uses_scaling:
        features_scaled = scaler.transform(features)
        ml_prediction = int(model.predict(features_scaled)[0])
        probability = model.predict_proba(features_scaled)[0]
    else:
        ml_prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0]

    # Apply Enhanced WHO Clinical Override
    hb_value = complete_data.get("Hb")
    mcv_value = complete_data.get("MCV")
    mch_value = complete_data.get("MCH")
    final_prediction = int(clinical_override(hb_value, mcv_value, mch_value, ml_prediction))
    
    # Track if override was triggered
    override_triggered = (final_prediction != ml_prediction)

    # Calculate anemia risk based on final prediction
    if final_prediction == 3:  # Severe
        anemia_risk = 100.0
    elif final_prediction == 2:  # Moderate
        anemia_risk = max(75.0, round((probability[2] + probability[3]) * 100, 2))
    elif final_prediction == 1:  # Mild
        anemia_risk = max(40.0, round((probability[1] + probability[2] + probability[3]) * 100, 2))
    else:  # Normal
        anemia_risk = round((probability[1] + probability[2] + probability[3]) * 100, 2)

    severity_map = {
        0: "Normal",
        1: "Mild Anemia",
        2: "Moderate Anemia",
        3: "Severe Anemia"
    }

    severity = severity_map.get(final_prediction, "Unknown")

    Hb = complete_data.get("Hb")
    MCV = complete_data.get("MCV")
    MCHC = complete_data.get("MCHC")
    PCV = complete_data.get("PCV")

    reasoning, deficiencies, suggestions = generate_clinical_reasoning(
        Hb, MCV, MCHC, PCV, override_triggered
    )

    return {
        "Severity": severity,
        "Risk_Percentage": float(anemia_risk),
        "Reasoning": reasoning,
        "Deficiencies": deficiencies,
        "Suggestions": suggestions,
        "Extracted": complete_data,
        "Clinical_Override": bool(override_triggered)
    }


def generate_clinical_reasoning(Hb, MCV, MCHC, PCV, override_triggered=False):
    reasoning = []
    deficiencies = []
    suggestions = []

    # Critical - Severe Anemia
    if Hb and Hb < 7.0:
        reasoning.append("⚠️ CRITICAL: Hemoglobin < 7.0 g/dL - WHO Severe Anemia Threshold")
        reasoning.append("Immediate medical attention required.")
        deficiencies.append("Severe Anemia (WHO Guidelines)")
        suggestions.append("URGENT: Consult hematologist immediately")
        suggestions.append("Blood transfusion may be required")
        suggestions.append("Hospitalization may be necessary")
        return reasoning, deficiencies, suggestions
    
    # Severe to Moderate Anemia
    elif Hb and Hb < 8.0:
        reasoning.append("⚠️ SEVERE: Hemoglobin < 8.0 g/dL - Critical anemia level")
        reasoning.append("Urgent medical evaluation needed.")
        deficiencies.append("Severe Anemia")
        suggestions.append("URGENT: See a doctor within 24 hours")
        suggestions.append("Iron supplementation likely required")
        suggestions.append("Further testing for underlying causes needed")
    
    # Moderate Anemia
    elif Hb and Hb < 10.0:
        reasoning.append("Hemoglobin significantly below normal (< 10.0 g/dL)")
        reasoning.append("Moderate anemia detected - medical attention recommended.")
        deficiencies.append("Moderate Anemia")
        suggestions.append("Schedule appointment with doctor soon")
        suggestions.append("Iron-rich diet and supplements recommended")
        suggestions.append("Monitor symptoms: fatigue, weakness, dizziness")
    
    # Mild Anemia
    elif Hb and Hb < 12.0:
        reasoning.append("Hemoglobin below normal range (< 12.0 g/dL)")
        deficiencies.append("Mild Anemia - Possible Iron Deficiency")
        suggestions.append("Increase iron-rich foods (red meat, spinach, lentils)")
        suggestions.append("Consider iron supplements after consulting doctor")
        suggestions.append("Check ferritin and B12 levels")
    
    # Normal but low-normal
    elif Hb and Hb < 13.0:
        reasoning.append("Hemoglobin in low-normal range")
        suggestions.append("Maintain balanced diet rich in iron and vitamins")
        suggestions.append("Monitor for any symptoms of fatigue")

    # Additional indicators
    if MCV and MCV < 80:
        reasoning.append("Low MCV indicates microcytic anemia (small red blood cells)")
        deficiencies.append("Likely Iron Deficiency or Thalassemia")
        suggestions.append("Iron studies recommended (serum iron, ferritin, TIBC)")
    elif MCV and MCV > 100:
        reasoning.append("High MCV indicates macrocytic anemia (large red blood cells)")
        deficiencies.append("Possible B12 or Folate Deficiency")
        suggestions.append("Check vitamin B12 and folate levels")

    if MCHC and MCHC < 32:
        reasoning.append("Low MCHC suggests hypochromic cells (pale red blood cells)")
        deficiencies.append("Low hemoglobin concentration in RBCs")
        suggestions.append("Iron supplementation may help")
    
    if PCV and PCV < 36:
        reasoning.append("Low PCV indicates reduced red cell volume")
        suggestions.append("Increase hydration and iron intake")

    # If everything is normal
    if not reasoning:
        reasoning.append("Blood parameters appear within normal range")
        suggestions.append("Maintain balanced nutrition and regular health checks")
        suggestions.append("Continue healthy lifestyle habits")

    return reasoning, deficiencies, suggestions

@app.route("/register", methods=["POST"])
def register():
    data = request.json

    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Email and password are required"}), 400

    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "User already exists"}), 400

    hashed = bcrypt.generate_password_hash(
        data["password"]
    ).decode("utf-8")

    new_user = User(
        email=data["email"],
        password=hashed
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201



@app.route("/login", methods=["POST"])
def login():
    data = request.json

    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=data["email"]).first()

    if not user:
        return jsonify({"error": "User not found"}), 404

    if not bcrypt.check_password_hash(
        user.password,
        data["password"]
    ):
        return jsonify({"error": "Invalid password"}), 401

    access_token = create_access_token(identity=str(user.id))

    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "user_id": user.id
    })



# =====================================================
# PDF UPLOAD ROUTE
# =====================================================

@app.route("/upload-pdf", methods=["POST"])
@jwt_required()
def upload_pdf():
    try:
        file = request.files["file"]

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        text = ""

        # =========================
        # Extract Text (PDF)
        # =========================
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"

        # =========================
        # OCR Fallback
        # =========================
        if not text.strip():
            file.seek(0)
            images = convert_from_bytes(file.read())
            for img in images:
                text += pytesseract.image_to_string(img)

        # =========================
        # REGEX EXTRACTION
        # =========================

        def extract_value(pattern):
            match = re.search(pattern, text, re.IGNORECASE)
            return float(match.group(1)) if match else None

        def extract_text_value(pattern):
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else None

        # CBC Values
        Hb = extract_value(r"(?:Hemoglobin|Hb).*?(\d+\.?\d*)")
        RBC = extract_value(r"(?:RBC|Red Blood Cell).*?(\d+\.?\d*)")
        PCV = extract_value(r"(?:PCV|Packed Cell Volume|Hematocrit).*?(\d+\.?\d*)")
        MCV = extract_value(r"(?:MCV|Mean Corpuscular Volume).*?(\d+\.?\d*)")
        MCH = extract_value(r"(?:MCH|Mean Corpuscular Hemoglobin)(?!\s*Concentration).*?(\d+\.?\d*)")
        MCHC = extract_value(r"(?:MCHC|Mean Corpuscular.*?Concentration).*?(\d+\.?\d*)")

        # =========================
        # Patient Name & Age Regex
        # =========================

        # Common hospital patterns
        patient_name = extract_text_value(
            r"(?:Patient Name|Name)\s*:?\s*([A-Za-z\s]+)"
        )

        age = extract_value(
            r"Age\s*:?\s*(\d+)"
        )

        # Default fallbacks
        if age is None:
            age = 30

        if patient_name is None:
            patient_name = "Unknown Patient"

        if Hb is None:
            return jsonify({"error": "Hemoglobin value not detected"}), 400

        # =========================
        # Prepare Data Dictionary
        # =========================

        data_dict = {
            "Gender": 1,
            "Age": age,
            "Hb": Hb,
            "RBC": RBC,
            "PCV": PCV,
            "MCV": MCV,
            "MCH": MCH,
            "MCHC": MCHC
        }

        # =========================
        # Process Using Shared Logic
        # =========================

        result = process_cbc_data(data_dict)

        # Add patient info
        result["Patient_Name"] = patient_name
        result["Age"] = age
        # Save to DB (profile_id must come from frontend)
        current_user = int(get_jwt_identity())
        profile_id = request.form.get("profile_id")

        if not profile_id:
            return jsonify({"error": "profile_id missing"}), 400

        profile_id = int(profile_id)

        profile = Profile.query.filter_by(id=profile_id, user_id=current_user).first()
        if not profile:
            return jsonify({"error": "Unauthorized profile"}), 403
        
        expires_at = datetime.utcnow() + timedelta(days=30)

        new_report = Report(
            profile_id=profile_id,
            patient_name=patient_name,
            severity=result["Severity"],
            risk_percentage=result["Risk_Percentage"],
            hb=data_dict["Hb"],
            rbc=data_dict["RBC"],
            pcv=data_dict["PCV"],
            mcv=data_dict["MCV"],
            mch=data_dict["MCH"],
            mchc=data_dict["MCHC"],
            expires_at=expires_at
        )
        print("CURRENT USER:", current_user)
        print("PROFILE ID RECEIVED:", profile_id)

        profile_check = Profile.query.get(profile_id)
        print("PROFILE USER_ID IN DB:", profile_check.user_id if profile_check else None)


        db.session.add(new_report)
        db.session.commit()

        result["report_id"] = new_report.id


        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =====================================================
# MANUAL PREDICT ROUTE
# =====================================================
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    try:
        data = request.json

        data_dict = {
            "Gender": float(data["Gender"]),
            "Age": float(data["Age"]),
            "Hb": float(data["Hb"]),
            "RBC": float(data["RBC"]),
            "PCV": float(data["PCV"]),
            "MCV": float(data["MCV"]),
            "MCH": float(data["MCH"]),
            "MCHC": float(data["MCHC"])
        }

        result = process_cbc_data(data_dict)

        profile_id = data.get("profile_id")

        if not profile_id:
            return jsonify({"error": "profile_id missing"}), 400
        
        current_user = int(get_jwt_identity())

        profile = Profile.query.filter_by(id=profile_id, user_id=current_user).first()

        if not profile:
            return jsonify({"error": "Unauthorized profile"}), 403


        expires_at = datetime.utcnow() + timedelta(days=30)

        new_report = Report(
            profile_id=int(profile_id),
            patient_name="Manual Entry",
            severity=result["Severity"],
            risk_percentage=result["Risk_Percentage"],
            hb=data_dict["Hb"],
            rbc=data_dict["RBC"],
            pcv=data_dict["PCV"],
            mcv=data_dict["MCV"],
            mch=data_dict["MCH"],
            mchc=data_dict["MCHC"],
            expires_at=expires_at
        )

        db.session.add(new_report)
        db.session.commit()

        result["report_id"] = new_report.id

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =====================================================
# CHAT ROUTE (GROQ)
# =====================================================
@app.route("/chat", methods=["POST"])
@jwt_required()
def chat():
    try:
        # Check if API key is configured
        if not API_KEY:
            return jsonify({
                "error": "Chat feature is not configured. GROQAI_API_KEY environment variable is not set."
            }), 503

        data = request.json
        user_message = data["message"]
        context = data["context"]

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Dr. HemoScan, an expert medical AI assistant specializing in Complete Blood Count (CBC) analysis and anemia diagnosis.\n\n"
                        
                        "YOUR ROLE:\n"
                        "- Provide clear, accurate, and easy-to-understand medical explanations\n"
                        "- Focus exclusively on CBC parameters and anemia-related conditions\n"
                        "- Use simple language that patients can understand, avoiding excessive medical jargon\n"
                        "- When using medical terms, always explain them in plain English\n\n"
                        
                        "ANALYSIS GUIDELINES:\n"
                        "1. Interpret CBC values in context (Hemoglobin, RBC, MCV, MCH, MCHC, PCV)\n"
                        "2. Explain what abnormal values mean for the patient's health\n"
                        "3. Identify potential types of anemia (iron deficiency, B12 deficiency, etc.)\n"
                        "4. Discuss possible causes and contributing factors\n"
                        "5. Suggest dietary recommendations and lifestyle changes\n"
                        "6. Explain the relationship between different blood parameters\n\n"
                        
                        "RESPONSE FORMAT:\n"
                        "- Use short paragraphs (2-3 sentences max)\n"
                        "- Use bullet points for lists\n"
                        "- Bold important terms using **term**\n"
                        "- Number steps when giving instructions\n"
                        "- Use line breaks for readability\n\n"
                        
                        "IMPORTANT RULES:\n"
                        "- NEVER provide a final diagnosis - only interpretations and insights\n"
                        "- ALWAYS recommend consulting a healthcare provider for diagnosis and treatment\n"
                        "- DO NOT answer questions unrelated to blood tests or anemia\n"
                        "- If asked about other medical conditions, politely redirect to CBC analysis\n"
                        "- Be empathetic and supportive in your tone\n"
                        "- Acknowledge patient concerns and validate their questions\n\n"
                        
                        "EXAMPLE RESPONSE STYLE:\n"
                        "Based on your CBC results, here's what I can tell you:\n\n"
                        "**Hemoglobin Level**: Your hemoglobin is [value], which is [below/above/within] the normal range.\n\n"
                        "This suggests:\n"
                        "• [Point 1]\n"
                        "• [Point 2]\n\n"
                        "**What you can do**:\n"
                        "1. [Action 1]\n"
                        "2. [Action 2]\n\n"
                        "⚠️ Important: Please consult your doctor for proper diagnosis and treatment plan."
                    )
                },
                {
                    "role": "user",
                    "content": f"CBC Report Context:\n{context}\n\nPatient Question: {user_message}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        result = response.json()

        if "choices" not in result:
            return jsonify(result), 400

        reply = result["choices"][0]["message"]["content"]
        report_id = data.get("report_id")

        if not report_id:
            return jsonify({"error": "report_id missing"}), 400
        
        current_user = int(get_jwt_identity())

        report = Report.query.get(report_id)

        if not report:
            return jsonify({"error": "Report not found"}), 404

        profile = Profile.query.get(report.profile_id)

        if not profile or profile.user_id != current_user:
            return jsonify({"error": "Unauthorized access"}), 403



        # Save user message
        user_msg = ChatMessage(
            report_id=report_id,
            role="user",
            message=user_message
        )
        db.session.add(user_msg)

        # Save AI reply
        ai_msg = ChatMessage(
            report_id=report_id,
            role="ai",
            message=reply
        )
        db.session.add(ai_msg)

        db.session.commit()

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



    
@app.route("/profile/<int:profile_id>/reports", methods=["GET"])
@jwt_required()
def get_reports(profile_id):
    current_user = int(get_jwt_identity())

    profile = Profile.query.filter_by(
        id=profile_id,
        user_id=current_user
    ).first()

    if not profile:
        return jsonify({"error": "Unauthorized access"}), 403

    reports = Report.query.filter(
        Report.profile_id == profile_id,
        Report.expires_at > datetime.utcnow()
    ).all()

    return jsonify([
        {
            "report_id": r.id,
            "patient_name": r.patient_name,
            "severity": r.severity,
            "risk": r.risk_percentage,
            "hb": r.hb,
            "rbc": r.rbc,
            "pcv": r.pcv,
            "mcv": r.mcv,
            "mch": r.mch,
            "mchc": r.mchc,
            "created_at": r.created_at
        }
        for r in reports
    ])

@app.route("/report/<int:report_id>/chat", methods=["GET"])
@jwt_required()
def get_chat(report_id):
    current_user = int(get_jwt_identity())

    report = Report.query.get(report_id)

    if not report:
        return jsonify({"error": "Report not found"}), 404

    profile = Profile.query.get(report.profile_id)

    if not profile or profile.user_id != current_user:
        return jsonify({"error": "Unauthorized access"}), 403

    messages = ChatMessage.query.filter_by(report_id=report_id).all()


    return jsonify([
        {
            "role": m.role,
            "message": m.message,
            "created_at": m.created_at
        }
        for m in messages
    ])


@app.route("/cleanup", methods=["POST"])
@jwt_required()
def cleanup():
    current_user = int(get_jwt_identity())

    profiles = Profile.query.filter_by(user_id=current_user).all()
    profile_ids = [p.id for p in profiles]

    expired = Report.query.filter(
        Report.profile_id.in_(profile_ids),
        Report.expires_at < datetime.utcnow()
    ).all()


    for r in expired:
        db.session.delete(r)

    db.session.commit()

    return jsonify({"message": "Expired reports deleted"})


@app.route("/create-profile", methods=["POST"])
@jwt_required()
def create_profile():
    current_user = int(get_jwt_identity())
    data = request.json

    profile = Profile(
        user_id=current_user,
        profile_name=data.get("profile_name", "Default Profile"),
        age=data.get("age", 30),
        gender=data.get("gender", "Male")
    )

    db.session.add(profile)
    db.session.commit()

    return jsonify({
        "message": "Profile created",
        "profile_id": profile.id
    })


@app.route("/profiles", methods=["GET"])
@jwt_required()
def get_profiles():
    current_user = int(get_jwt_identity())
    profiles = Profile.query.filter_by(user_id=current_user).all()

    return jsonify([
        {
            "profile_id": p.id,
            "profile_name": p.profile_name,
            "age": p.age,
            "gender": p.gender,
            "created_at": p.created_at
        }
        for p in profiles
    ])

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
