# 🎓 Classroom Student Attention Monitoring (OpenCV + Dlib + Streamlit)

This project monitors students in a classroom using **computer vision**.  
It detects whether students are **attentive** or **inattentive** based on:

- 👀 Eye Aspect Ratio (EAR) → closed eyes / sleep detection  
- 🧑‍🤝‍🧑 Head Pose Estimation → looking forward vs. sideways  
- 📊 Logs and Dashboard → attention percentage per student  

---

## 🚀 Features
- Real-time video monitoring
- Logs attention status in CSV
- Dashboard to visualize attentiveness over time
- Extendable with gaze detection / engagement datasets

---

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/classroom-attention-monitoring.git
   cd classroom-attention-monitoring
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the **shape predictor model**:  
   👉 [Download here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)  
   Extract and place `shape_predictor_68_face_landmarks.dat` in the project folder.

---

## ▶️ Usage

### Real-time monitoring
```bash
python main.py
```

- Press `q` to quit.
- Logs saved in `logs/attention_log.csv`.

### Dashboard (Streamlit)
```bash
streamlit run dashboard.py
```

- Opens a browser window with charts and stats.

---

## 📊 Example Output

- **Video Feed**:  
  🟢 Green box → attentive student  
  🔴 Red box → inattentive student  

- **Dashboard**:  
  - Overall attentiveness %  
  - Per-student engagement  
  - Timeline chart of attention  

---

## 📂 Dataset (Optional)

If you want to train ML models:
- [DAiSEE Dataset](https://people.iith.ac.in/vineethnb/publication/daisee/) (student engagement levels)
- [MAVS Dataset](https://zenodo.org/record/6502452)

---

## 🛠 Extensions
- Use YOLOv8 or MediaPipe for faster detection
- Add alerts when attention < 50%
- Multi-camera classroom support
- Cloud-based analytics

---

## 👨‍💻 Author
Built with ❤️ using OpenCV, dlib, and Streamlit.
