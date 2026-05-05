🏢 HVAC AI Smart Building Dashboard

An AI-powered smart building system that uses machine learning to analyze HVAC sensor data, detect anomalies, predict ambient temperature, and simulate real-world building scenarios through an interactive web dashboard.

🚀 Project Overview

This project focuses on applying artificial intelligence to building automation systems. Using real HVAC sensor data, the system learns normal behavior patterns and identifies abnormal conditions that could indicate inefficiencies, faults, or environmental issues.

The solution includes:

Data analysis and preprocessing in Jupyter Notebook
Machine learning models for prediction and anomaly detection
A fully interactive web dashboard built with Streamlit
🧠 Key Features
🔍 Anomaly Detection
Uses Isolation Forest (scikit-learn)
Identifies unusual HVAC behavior patterns
Highlights abnormal temperature spikes and system irregularities
🌡️ Temperature Prediction
Uses Linear Regression
Predicts ambient temperature based on:
humidity
power usage
CO2 levels
🎛️ Scenario Simulator
Interactive sliders to simulate:
humidity
power
CO2 levels
AI predicts system response in real time
Provides system status:
Normal
Anomaly
📊 HVAC Signal Visualization
Displays:
damper behavior
system on/off states
pressure signals
temperature trends
Helps visualize how systems behave over time
🛠️ Technologies Used
Python
Pandas / NumPy
Matplotlib
Scikit-learn
Isolation Forest
Linear Regression
Streamlit (web app dashboard)
Jupyter Notebook
📁 Project Structure
├── app.py                     # Streamlit web application
├── smartbuilding_ai.ipynb     # Data analysis & model training
├── hvac_data_cleaned.parquet  # Dataset
├── README.md                  # Project documentation
▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/Jefferyhood/HVAC-AI-SmartBuilding.git
cd HVAC-AI-SmartBuilding
2. Install dependencies
pip install pandas numpy matplotlib scikit-learn streamlit pyarrow
3. Run the dashboard
streamlit run app.py
4. Open in browser
http://localhost:8501
📊 Example Use Case

A facility manager can input:

High humidity
Increased power usage
Elevated CO2 levels

The AI system will:

Predict resulting temperature
Determine if the system is operating normally
Flag potential inefficiencies or ventilation issues
⚠️ Challenges & Learnings

During this project, several challenges were encountered:

Understanding how to properly structure machine learning pipelines
Learning how anomaly detection differs from prediction models
Handling feature consistency between training and prediction
Integrating models into a real-time web application
Debugging authentication and deployment issues (Git & Streamlit)

These challenges helped deepen understanding of:

Model behavior
Data preprocessing
Real-world AI system integration
🔮 Future Improvements
Add real-time IoT sensor integration
Improve model accuracy with more features
Deploy the app to cloud (Streamlit Cloud / AWS)
Add alert notifications for anomalies
Enhance UI/UX design
👨‍💻 Author

Jeffery Hood
Master’s in Software Engineering
