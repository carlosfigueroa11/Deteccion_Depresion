import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import gradio as gr
from openai import OpenAI
import os
# Cliente OpenAI
client = OpenAI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Cargar el dataset
df = pd.read_csv("student_depression_dataset.csv")

# Diagnostico
print("Sleep Duration:", df["Sleep Duration"].unique())
print("Dietary Habits:", df["Dietary Habits"].unique())
print("Degree:", df["Degree"].unique())
print("Suicidal Thoughts:", df["Have you ever had suicidal thoughts ?"].unique())
print("Family History:", df["Family History of Mental Illness"].unique())

#  Limpiar comillas simples en los valores categoricos
df["Sleep Duration"] = df["Sleep Duration"].str.strip("'")
df["Degree"] = df["Degree"].str.strip("'")

#  Preprocesamiento
data = df.copy()
data = data.drop(columns=["id", "City", "Profession"])


# Normalizacion de valores para que coincidan con la interfaz 
data["Sleep Duration"] = data["Sleep Duration"].replace({
    "5-6 hrs": "5-6 hours",
    "7-8 hrs": "7-8 hours",
    "<5 hrs": "Less than 5 hours",
    "Less than 5 hrs": "Less than 5 hours"
})

# Codificar variables categóricas
label_cols = data.select_dtypes(include=["object"]).columns
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le

# Variables predictoras y objetivo
X = data.drop(columns=["Depression"])
y = data["Depression"]

# Division en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Funcion de prediccion
def predecir_depresion(gender, age, academic_pressure, work_pressure, cgpa,
                       study_satisfaction, job_satisfaction, sleep_duration,
                       dietary_habits, degree, suicidal_thoughts,
                       work_study_hours, financial_stress, family_history):

    try:
        input_dict = {
            "Gender": [le_dict["Gender"].transform([gender])[0]],
            "Age": [age],
            "Academic Pressure": [academic_pressure],
            "Work Pressure": [work_pressure],
            "CGPA": [cgpa],
            "Study Satisfaction": [study_satisfaction],
            "Job Satisfaction": [job_satisfaction],
            "Sleep Duration": [le_dict["Sleep Duration"].transform([sleep_duration])[0]],
            "Dietary Habits": [le_dict["Dietary Habits"].transform([dietary_habits])[0]],
            "Degree": [le_dict["Degree"].transform([degree])[0]],
            "Have you ever had suicidal thoughts ?": [le_dict["Have you ever had suicidal thoughts ?"].transform([suicidal_thoughts])[0]],
            "Work/Study Hours": [work_study_hours],
            "Financial Stress": [float(financial_stress)],
            "Family History of Mental Illness": [le_dict["Family History of Mental Illness"].transform([family_history])[0]],
        }

        input_df = pd.DataFrame(input_dict)
        pred = model.predict(input_df)[0]

        return "🟡 Posible Depresión" if pred == 1 else "🟢 Sin signos de depresión"

    except Exception as e:
        return f"❌ Error en la predicción: {str(e)}"

# Interfaz de prediccion con Gradio
interface = gr.Interface(
    fn=predecir_depresion,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Género"),
        gr.Slider(18, 40, step=1, label="Edad"),
        gr.Slider(1, 5, step=1, label="Presión Académica"),
        gr.Slider(0, 5, step=1, label="Presión Laboral"),
        gr.Slider(0.0, 10.0, step=0.1, label="CGPA"),
        gr.Slider(1, 5, step=1, label="Satisfacción con el Estudio"),
        gr.Slider(0, 5, step=1, label="Satisfacción Laboral"),
        gr.Dropdown(["5-6 hours", "7-8 hours", "Less than 5 hours"], label="Duración del Sueño"),
        gr.Dropdown(["Healthy", "Moderate", "Unhealthy"], label="Hábitos Alimenticios"),
        gr.Dropdown(["B.Pharm", "BSc", "BA", "BCA", "M.Tech"], label="Grado Académico"),
        gr.Dropdown(["Yes", "No"], label="Pensamientos Suicidas"),
        gr.Slider(0, 12, step=1, label="Horas de Trabajo/Estudio"),
        gr.Slider(1.0, 5.0, step=1.0, label="Estrés Financiero"),
        gr.Dropdown(["Yes", "No"], label="Historial Familiar de Enfermedad Mental"),
    ],
    outputs="text",
    title="Detección de Depresión en Estudiantes",
    description="Ingresa los datos del estudiante y el modelo predecirá si hay signos de depresión."
)

# Chat GPT
chat_historial = []

def chat_gpt(usuario_input):
    chat_historial.append({"role": "user", "content": usuario_input})

    messages = [
        {"role": "system", "content": "Eres un asistente empático que da consejos sobre salud mental y bienestar para estudiantes."},
        *chat_historial
    ]

    try:
        respuesta = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        contenido = respuesta.choices[0].message.content.strip()
        chat_historial.append({"role": "assistant", "content": contenido})
        return contenido
    except Exception as e:
        return f"Ocurrió un error al conectar con GPT: {e}"

chat_interface = gr.Interface(
    fn=chat_gpt,
    inputs=gr.Textbox(lines=3, placeholder="Escribe tu pregunta..."),
    outputs="text",
    title="Asistente de Bienestar Emocional",
    description="Haz una pregunta sobre estrés, sueño, depresion o cómo sentirte mejor. Un asistente IA te responderá con empatía y consejos útiles."
)

# Lanzar ambas interfaces en pestañas
gr.TabbedInterface(
    [interface, chat_interface],
    ["Prediccion de Depresion", "Chat con IA (GPT)"]
).launch(server_name="0.0.0.0", server_port=7860)
