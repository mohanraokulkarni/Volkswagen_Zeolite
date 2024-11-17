import tkinter as tk
from tkinter import messagebox
import sqlite3
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load trained models
with open('clf_model.pkl', 'rb') as clf_file:
    clf_model = pickle.load(clf_file)

with open('reg_model.pkl', 'rb') as reg_file:
    reg_model = pickle.load(reg_file)

# Initialize scaler (assuming scaler was the same during training)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define class names
class_names = ["Normal Operation", "Thermal Failure", "Electrical Failure", "Mechanical Failure", "Environmental Failure"]

# Function to convert minutes to human-readable format
def convert_minutes(total_minutes):
    if total_minutes < 60:
        return f"{int(total_minutes)} minute(s)"
    elif total_minutes < 1440:
        hours = total_minutes // 60
        return f"{int(hours)} hour(s)"
    else:
        days = total_minutes // 1440
        hours = (total_minutes % 1440) // 60
        return f"{int(days)} day(s) and {int(hours)} hour(s)"

# SQLite setup
def setup_database():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    # Drop existing table (optional, for debugging)
    cursor.execute("DROP TABLE IF EXISTS predictions")
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_message TEXT
        )
    """)
    conn.commit()
    conn.close()

# Save to database
def save_to_database(full_message):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (full_message) VALUES (?)", (full_message,))
    conn.commit()
    conn.close()


# Generate Real-Time Data
def generate_data():
    global real_time_data
    real_time_data = np.array([[360, 85, 90, 3.5, 45, 80]])  # Example real-time data
    messagebox.showinfo("Data Generated", "Real-time data has been generated successfully!")

# Predict Failure
def predict_failure():
    try:
        real_time_data_scaled = scaler.transform(real_time_data)
        predicted_class = clf_model.predict(real_time_data_scaled)[0]
        predicted_time = reg_model.predict(real_time_data_scaled)[0]
        readable_time = convert_minutes(predicted_time)

        if predicted_class == 0:
            result_message = f"🚨 Predicted: {class_names[predicted_class]} (No failure detected). 😊"
            messagebox.showinfo("Prediction Result", result_message)
        else:
            fault_device = ""
            if class_names[predicted_class] == "Thermal Failure":
                fault_device = "Battery Pack"
            elif class_names[predicted_class] == "Electrical Failure":
                fault_device = "Battery Management System (BMS)"
            elif class_names[predicted_class] == "Mechanical Failure":
                fault_device = "Electric Motor Bearings"
            elif class_names[predicted_class] == "Environmental Failure":
                fault_device = "Connectors and Cables"

            result_message = f"🚨 Attention! We've detected a potential {fault_device} failure.\nEstimated time to failure: {readable_time} .\nPlease check the system promptly to avoid disruptions. 😊"
            save_to_database(result_message)  # Save only the alert message
            messagebox.showinfo("Prediction Result", result_message)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI Setup
root = tk.Tk()
root.title("Predictive Maintenance System")
root.geometry("500x300")

# Buttons
generate_button = tk.Button(root, text="Generate Real-Time Data", command=generate_data, width=30, height=2, bg="lightblue")
generate_button.pack(pady=20)

predict_button = tk.Button(root, text="Predict Failure", command=predict_failure, width=30, height=2, bg="lightgreen")
predict_button.pack(pady=20)

# Database Setup
setup_database()

# Run the GUI
root.mainloop()
