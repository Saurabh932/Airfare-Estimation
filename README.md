# Airfare Prediction

Predict flight prices based on various parameters using machine learning.


---

# Webapp Link
Link: http://3.85.79.169:7000

---

## Workflow

- setup
- experiments
- components
- pipeline
- Docker
- app
- AWS (ECS)


---

## Overview

This project utilizes machine learning to predict flight prices based on factors such as departure time, arrival time, airline, and more. The application is built using Flask for the backend server, providing a web interface where users can input their flight details and get an estimated price prediction.

---


## Features

- Predict flight prices based on user-provided details.
- Interactive web interface built with Flask and Bootstrap.
- Easy-to-use form for inputting flight details.
- Responsive design for seamless user experience on various devices.

---

## How to Use This Repository

1. Clone the repository to your local machine.
2. Install the required dependencies specified in the `requirements.txt` file.
3. Explore the project structure to understand the organization of code, data, and resources.
4. Follow the instructions in the README.md file to run the prediction pipeline and deploy the models to production environments.



### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/airfare-prediction.git
   cd airfare-prediction


2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```


3. ***Run the Application:***
    To run the Flask web application:
    ```bash
    python app.py
    ```
* Access the application in your web browser at http://localhost:7000.


5. **Run with Docker:**
* If you prefer to use Docker, you can easily run the project with Docker Compose. Simply navigate to the project directory and run:
```bash
docker-compose up --build
```
* And the run:
```bash
docker-compose up
```
<br>


### Techstack Used:

- Python
- Git
- Flask
- Docker


## Web Application Demo

**1. Click the button to procedded:**

![](https://github.com/Saurabh932/Airfare-Estimation/blob/main/images/image-1.png)


**2. Enter the values.**

![](https://github.com/Saurabh932/Airfare-Estimation/blob/main/images/image-2.png)


**3. Click on submit to get final Result**

![](https://github.com/Saurabh932/Airfare-Estimation/blob/main/images/image-3.png)
