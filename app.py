import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open('student_rf_pipeline.pkl','rb') as file:
    model = pickle.load(file)

all_columns = ['gender', 'age', 'address', 'famsize', 'Pstatus', 'M_Edu', 'F_Edu',
       'M_Job', 'F_Job', 'relationship', 'smoker', 'tuition_fee',
       'time_friends', 'ssc_result']

def predict_gpa(gender, age, address, famsize, Pstatus, m_edu, f_edu, m_job, f_job, rls, smoker,
                tuition_fee, time_friends, ssc_r):

    input_df = pd.DataFrame([[
        gender, age, address, famsize, Pstatus, m_edu, f_edu, m_job, f_job, rls, smoker,
        tuition_fee, time_friends, ssc_r
    ]],
    columns=all_columns
    )

    prediction =model.predict(input_df)[0]

    return f"Predicted HSC result {np.clip(prediction, 0,5):.2f}"

inputs = [
    gr.Radio(["M", "F"], label='Gender'),
    gr.Number(label="Age", value=18),
    gr.Radio(['Rural', 'Urban'], label='Address'),
    gr.Radio(['GT3', "LE3"], label='Famsize'),
    gr.Radio(['Together', 'Apart'], label='Pstatus'),
    gr.Slider(0, 4, step=1, label='Mother Edu'),
    gr.Slider(0, 4, step=1, label='Fother Edu'),
    gr.Dropdown(['At_home', 'Health', 'Other', 'Services', "Teacher"], label='Mother Job'),
    gr.Dropdown(['Business', 'Health', 'Farmer', 'Services', "Teacher"], label='Father Job'),
    gr.Radio(["Yes", 'NO'], label='Relationship'),
    gr.Radio(["Yes", 'NO'], label='Smoker'),
    gr.Number(label='tution_fee'),
    gr.Slider(0,5, step=1, label='Time With friends'),
    gr.Number(label='SSC Result (GPA)')

]

app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs='text',
    title='HSC Result Predictior'
)

app.launch(share=True)