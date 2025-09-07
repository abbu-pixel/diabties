Pima Indians Diabetes Predictor
This is a web application built with Python Flask and Scikit-learn that predicts whether a person has diabetes based on several health metrics. The front end is a modern, responsive single-page application built with HTML, CSS (Tailwind CSS), and vanilla JavaScript.

The machine learning model is trained on the Pima Indians Diabetes Dataset, which is fetched directly from a URL.

Features
Dynamic Prediction: The application uses JavaScript's fetch API to get predictions from the Flask backend without a page reload.

Improved Model: The machine learning model uses a RandomForestClassifier with GridSearchCV to find the best hyperparameters, resulting in a more accurate model.

Clean UI: The user interface is designed with a responsive layout using Tailwind CSS for a better user experience on both desktop and mobile devices.

Dependencies: All necessary Python libraries are listed in requirements.txt for easy installation.

Automated Workflow: A GitHub Actions workflow is included to automate testing and ensure the application remains in a working state.

Files
app.py: The core Flask application that handles web requests and interacts with the machine learning model.

model.py: A script to train and save the machine learning model. It fetches the dataset, preprocesses the data, and saves the trained model to model.pkl.

index.html: The front-end of the application, containing the user form, CSS styling, and JavaScript for dynamic predictions.

requirements.txt: A list of all required Python packages.

.github/workflows/ci.yml: The GitHub Actions workflow file for continuous integration.

How to Run the Application
Clone the Repository:

git clone [https://github.com/](https://github.com/)[your-username]/[your-repository-name].git
cd [your-repository-name]

Install Dependencies:

pip install -r requirements.txt

Train the Model:
Run the model.py script to train the model and generate the model.pkl file. This is a one-time step.

python model.py

Run the Flask App:

python app.py

Open in Browser:
Navigate to http://127.0.0.1:5000 in your web browser.

Dataset
The model is trained on the Pima Indians Diabetes Dataset. It includes the following features:

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

The model predicts the presence of diabetes (1) or the absence of diabetes (0).

Feel free to open an issue or submit a pull request if you have any suggestions or improvements.
