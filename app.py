from flask import Flask, jsonify, request, send_from_directory
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__, static_folder='static', static_url_path='')

MODEL_PATH = Path(__file__).parent / 'model' / 'Random_Forest_Regressor_Model.pkl'
model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)

NUMERIC_FEATURES = ['Age', 'Years of Experience']
GENDER_OPTIONS = ['Female', 'Male', 'Other']
EDUCATION_OPTIONS = ["Bachelor's", "Bachelor's Degree", 'High School', "Master's", "Master's Degree", 'PhD', 'phD']
JOB_TITLE_OPTIONS = ['Account Manager', 'Accountant', 'Administrative Assistant', 'Back end Developer', 'Business Analyst', 'Business Development Manager', 'Business Intelligence Analyst', 'CEO', 'Chief Data Officer', 'Chief Technology Officer', 'Content Marketing Manager', 'Copywriter', 'Creative Director', 'Customer Service Manager', 'Customer Service Rep', 'Customer Service Representative', 'Customer Success Manager', 'Customer Success Rep', 'Data Analyst', 'Data Entry Clerk', 'Data Scientist', 'Delivery Driver', 'Developer', 'Digital Content Producer', 'Digital Marketing Manager', 'Digital Marketing Specialist', 'Director', 'Director of Business Development', 'Director of Data Science', 'Director of Engineering', 'Director of Finance', 'Director of HR', 'Director of Human Capital', 'Director of Human Resources', 'Director of Marketing', 'Director of Operations', 'Director of Product Management', 'Director of Sales', 'Director of Sales and Marketing', 'Event Coordinator', 'Financial Advisor', 'Financial Analyst', 'Financial Manager', 'Front End Developer', 'Front end Developer', 'Full Stack Engineer', 'Graphic Designer', 'HR Generalist', 'HR Manager', 'Help Desk Analyst', 'Human Resources Coordinator', 'Human Resources Director', 'Human Resources Manager', 'IT Manager', 'IT Support', 'IT Support Specialist', 'Junior Account Manager', 'Junior Accountant', 'Junior Advertising Coordinator', 'Junior Business Analyst', 'Junior Business Development Associate', 'Junior Business Operations Analyst', 'Junior Copywriter', 'Junior Customer Support Specialist', 'Junior Data Analyst', 'Junior Data Scientist', 'Junior Designer', 'Junior Developer', 'Junior Financial Advisor', 'Junior Financial Analyst', 'Junior HR Coordinator', 'Junior HR Generalist', 'Junior Marketing Analyst', 'Junior Marketing Coordinator', 'Junior Marketing Manager', 'Junior Marketing Specialist', 'Junior Operations Analyst', 'Junior Operations Coordinator', 'Junior Operations Manager', 'Junior Product Manager', 'Junior Project Manager', 'Junior Recruiter', 'Junior Research Scientist', 'Junior Sales Associate', 'Junior Sales Representative', 'Junior Social Media Manager', 'Junior Social Media Specialist', 'Junior Software Developer', 'Junior Software Engineer', 'Junior UX Designer', 'Junior Web Designer', 'Junior Web Developer', 'Juniour HR Coordinator', 'Juniour HR Generalist', 'Marketing Analyst', 'Marketing Coordinator', 'Marketing Director', 'Marketing Manager', 'Marketing Specialist', 'Network Engineer', 'Office Manager', 'Operations Analyst', 'Operations Director', 'Operations Manager', 'Principal Engineer', 'Principal Scientist', 'Product Designer', 'Product Manager', 'Product Marketing Manager', 'Project Engineer', 'Project Manager', 'Public Relations Manager', 'Receptionist', 'Recruiter', 'Research Director', 'Research Scientist', 'Sales Associate', 'Sales Director', 'Sales Executive', 'Sales Manager', 'Sales Operations Manager', 'Sales Representative', 'Senior Account Executive', 'Senior Account Manager', 'Senior Accountant', 'Senior Business Analyst', 'Senior Business Development Manager', 'Senior Consultant', 'Senior Data Analyst', 'Senior Data Engineer', 'Senior Data Scientist', 'Senior Engineer', 'Senior Financial Advisor', 'Senior Financial Analyst', 'Senior Financial Manager', 'Senior Graphic Designer', 'Senior HR Generalist', 'Senior HR Manager', 'Senior HR Specialist', 'Senior Human Resources Coordinator', 'Senior Human Resources Manager', 'Senior Human Resources Specialist', 'Senior IT Consultant', 'Senior IT Project Manager', 'Senior IT Support Specialist', 'Senior Manager', 'Senior Marketing Analyst', 'Senior Marketing Coordinator', 'Senior Marketing Director', 'Senior Marketing Manager', 'Senior Marketing Specialist', 'Senior Operations Analyst', 'Senior Operations Coordinator', 'Senior Operations Manager', 'Senior Product Designer', 'Senior Product Development Manager', 'Senior Product Manager', 'Senior Product Marketing Manager', 'Senior Project Coordinator', 'Senior Project Engineer', 'Senior Project Manager', 'Senior Quality Assurance Analyst', 'Senior Research Scientist', 'Senior Researcher', 'Senior Sales Manager', 'Senior Sales Representative', 'Senior Scientist', 'Senior Software Architect', 'Senior Software Developer', 'Senior Software Engineer', 'Senior Training Specialist', 'Senior UX Designer', 'Social M', 'Social Media Man', 'Social Media Manager', 'Social Media Specialist', 'Software Developer', 'Software Engineer', 'Software Engineer Manager', 'Software Manager', 'Software Project Manager', 'Strategy Consultant', 'Supply Chain Analyst', 'Supply Chain Manager', 'Technical Recruiter', 'Technical Support Specialist', 'Technical Writer', 'Training Specialist', 'UX Designer', 'UX Researcher', 'VP of Finance', 'VP of Operations', 'Web Developer']


def build_feature_row(age, years_of_experience, gender, education_level, job_title):
    row = {name: 0 for name in feature_names}

    if 'Age' in row:
        row['Age'] = float(age)
    if 'Years of Experience' in row:
        row['Years of Experience'] = float(years_of_experience)

    gender_col = f'Gender_{gender}'
    education_col = f'Education Level_{education_level}'
    job_col = f'Job Title_{job_title}'

    for col, value, label in [
        (gender_col, gender, 'gender'),
        (education_col, education_level, 'education level'),
        (job_col, job_title, 'job title'),
    ]:
        if col not in row:
            raise ValueError(f'Unsupported {label}: {value}')
        row[col] = 1

    return row


@app.get('/api/options')
def options():
    return jsonify({
        'model_class': type(model).__name__,
        'feature_count': len(feature_names),
        'inputs': {
            'numeric': NUMERIC_FEATURES,
            'gender': GENDER_OPTIONS,
            'education_level': EDUCATION_OPTIONS,
            'job_title': JOB_TITLE_OPTIONS,
        }
    })


@app.post('/api/predict')
def predict():
    data = request.get_json(silent=True) or {}
    required = ['age', 'years_of_experience', 'gender', 'education_level', 'job_title']
    missing = [field for field in required if field not in data or data[field] in [None, '']]
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

    try:
        row = build_feature_row(
            data['age'],
            data['years_of_experience'],
            data['gender'],
            data['education_level'],
            data['job_title'],
        )
        df = pd.DataFrame([row], columns=feature_names)
        prediction = float(model.predict(df)[0])
        return jsonify({
            'predicted_value': round(prediction, 2),
            'currency_note': 'The model output is returned as stored by the trained regressor. Rename the label if your target is not salary.'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
