# 🍔 Food Calorie Estimation AI

An AI-powered Food Calorie Estimation Web Application built using **Flask, PyTorch, EfficientNet, Computer Vision, and Deep Learning**.

The system detects food items from uploaded images, estimates portion size, predicts calories, tracks daily calorie intake, and provides analytics dashboards with interactive charts.

---

# 🚀 Features

## ✅ AI Food Detection
- Detects multiple food items in a single image
- Deep learning-based food classification
- EfficientNet + Food101 dataset

## ✅ Calorie Estimation
- Estimates:
  - Food weight
  - Portion ratio
  - Calories per 100g
  - Total calories

## ✅ Bounding Box Detection
- Highlights detected food regions
- Displays processed image with annotations

## ✅ User Authentication
- User Registration
- Login / Logout
- Secure password hashing

## ✅ Daily Calorie Tracking
- Track daily calorie intake
- Set calorie goals
- Monitor eating habits

## ✅ Analytics Dashboard
- Calorie intake charts
- Food category distribution
- Weight trend visualization
- Interactive Plotly graphs

## ✅ History Management
- Stores previous analyses
- View detection history
- Delete old analyses

## ✅ Database Integration
- SQLite database support
- User management
- Food analysis records

---

# 🛠 Tech Stack

## Backend
- Python
- Flask

## AI / Machine Learning
- PyTorch
- EfficientNet
- Food101 Dataset
- Computer Vision

## Frontend
- HTML
- CSS
- JavaScript

## Database
- SQLite
- SQLAlchemy

## Authentication
- Flask-Login
- Werkzeug Security

## Visualization
- Plotly
- Pandas
- NumPy

---

# 📂 Project Structure

```bash
food-calorie-estimation-ai/
│
├── static/
│   ├── uploads/
│   └── processed/
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── dashboard.html
│
├── model_utils.py
├── efficientnet_food101.pth
├── app.py
├── requirements.txt
├── food_calorie.db
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Girivardhan-Reddy/food-calorie-estimation-ai.git
```

## 2️⃣ Move into Project Directory

```bash
cd food-calorie-estimation-ai
```

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

## 4️⃣ Activate Environment

### Windows
```bash
venv\Scripts\activate
```

### Linux / Mac
```bash
source venv/bin/activate
```

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python app.py
```

Open browser:

```bash
http://localhost:5000
```

---

# 🧠 AI Model

The project uses:
- EfficientNet
- Food101 dataset
- PyTorch deep learning framework

Supports:
- Multi-food detection
- Portion estimation
- Calorie prediction

---

# 📸 Supported File Types

- PNG
- JPG
- JPEG
- GIF
- BMP

---

# 📊 Dashboard Features

## Calorie Charts
- Daily calorie tracking
- Weekly calorie analytics

## Food Distribution
- Category-wise calorie analysis

## Weight Trends
- Food weight monitoring

---

# 🔐 Authentication System

Secure authentication using:
- Flask-Login
- Password hashing
- Session management

---

# 🗄 Database Models

## User
Stores:
- Username
- Email
- Password

## FoodAnalysis
Stores:
- Food name
- Calories
- Confidence score
- Weight estimation
- Bounding boxes

## DailyCalorieLog
Stores:
- Daily calories
- Calorie goals
- Progress tracking

---

# 🔥 Future Improvements

- Real-time webcam food detection
- Mobile application
- Nutrition recommendation engine
- Meal planning AI
- Barcode scanner integration
- Fitness tracking integration

---

# 👨‍💻 Author

## Girivardhan Reddy

AI & Full Stack Developer  
Machine Learning & Computer Vision Enthusiast

GitHub:
https://github.com/Girivardhan-Reddy

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the MIT License.
