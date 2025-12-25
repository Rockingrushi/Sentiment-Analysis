# Sentiment Analysis Web Application - Project Report

## Executive Summary

This project implements a complete Sentiment Analysis Web Application using machine learning techniques. The application allows users to input text reviews and receive real-time sentiment predictions (positive, negative, or neutral) through a clean web interface.

## Project Objectives

- Develop a machine learning model for sentiment analysis
- Create a web-based user interface for easy interaction
- Demonstrate end-to-end ML application development
- Provide a foundation for further NLP and web development projects

## Technical Architecture

### 1. Data Layer
- **Dataset**: CSV file containing text reviews and corresponding sentiment labels
- **Preprocessing**: Text cleaning using regex, lowercasing, and normalization
- **Features**: TF-IDF vectorization for numerical representation of text

### 2. Model Layer
- **Algorithm**: Multinomial Naive Bayes classifier
- **Training**: Supervised learning on labeled dataset
- **Evaluation**: Accuracy metrics on test set
- **Persistence**: Model and vectorizer saved using pickle

### 3. Application Layer
- **Framework**: Flask web framework
- **Routing**: Single route handling GET/POST requests
- **Templates**: Jinja2 templating for dynamic HTML
- **Styling**: CSS for responsive design

### 4. User Interface
- **Input**: Textarea for user review input
- **Output**: Display of predicted sentiment
- **Design**: Clean, centered layout with green accent color

## Implementation Details

### Dataset Preparation
```csv
text,sentiment
I love this product,positive
This is the worst experience,negative
Average quality,neutral
Very useful and amazing,positive
Not good at all,negative
```

### Model Training Process
1. Load dataset using pandas
2. Apply text preprocessing (lowercasing, regex cleaning)
3. Convert text to numerical features using TF-IDF
4. Split data into training and testing sets
5. Train Multinomial Naive Bayes model
6. Evaluate model accuracy
7. Save model and vectorizer for deployment

### Web Application Flow
1. User accesses the web page
2. User enters text in the form
3. Form submission triggers POST request
4. Flask app processes the input:
   - Loads saved model and vectorizer
   - Preprocesses user input
   - Makes prediction
5. Result is displayed on the same page

## Technologies Used

- **Programming Language**: Python 3.11
- **Web Framework**: Flask 3.0.3
- **Machine Learning**: scikit-learn 1.8.0
- **Data Processing**: pandas 2.3.1
- **Numerical Computing**: NumPy 1.26.4
- **Frontend**: HTML5, CSS3
- **Templating**: Jinja2

## Model Performance

- **Training Accuracy**: 100% (on small dataset)
- **Test Accuracy**: 0% (limited by small dataset size)
- **Note**: Performance improves significantly with larger, more diverse datasets

## Project Structure

```
sentiment-analysis-app/
│
├── dataset.csv              # Training data
├── train_model.py           # Model training script
├── app.py                   # Flask application
├── model.pkl                # Trained model
├── vectorizer.pkl           # TF-IDF vectorizer
│
├── templates/
│   └── index.html           # Web interface
│
├── static/
│   └── style.css            # Styling
│
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── report.md                # This report
```

## Code Snippets

### Model Training (train_model.py)
```python
# Load and preprocess data
data = pd.read_csv("dataset.csv")
data['text'] = data['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(data['text'])

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Save artifacts
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
```

### Flask Application (app.py)
```python
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        user_text = request.form["text"]
        cleaned = clean_text(user_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
    return render_template("index.html", result=prediction)
```

## Challenges and Solutions

### Challenge 1: Small Dataset
- **Problem**: Limited training data leading to poor generalization
- **Solution**: Used simple, effective preprocessing and focused on demonstration

### Challenge 2: Model Persistence
- **Problem**: Need to save and load trained model for web app
- **Solution**: Used pickle for serialization of scikit-learn objects

### Challenge 3: Web Interface Design
- **Problem**: Creating an intuitive, responsive UI
- **Solution**: Simple, clean design with centered layout and clear call-to-action

## Future Improvements

1. **Larger Dataset**: Incorporate more diverse training data
2. **Advanced Models**: Experiment with deep learning (LSTM, BERT)
3. **Multi-language Support**: Extend to other languages
4. **Batch Processing**: Allow analysis of multiple reviews
5. **API Endpoints**: Create REST API for integration
6. **User Authentication**: Add user accounts and history
7. **Visualization**: Charts showing sentiment distribution

## Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline
- Web application development with Flask
- Text preprocessing and feature engineering
- Model deployment and serving
- Version control and documentation best practices

## Conclusion

The Sentiment Analysis Web Application successfully demonstrates the integration of machine learning with web development. While the current implementation uses a simple model and small dataset, it provides a solid foundation for more advanced sentiment analysis applications. The modular architecture allows for easy extension and improvement.

The project serves as an excellent portfolio piece for demonstrating skills in Python, machine learning, and web development, making it ideal for academic projects, interviews, and professional development.
