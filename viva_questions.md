# Sentiment Analysis Web Application - Viva Questions & Answers

## Basic Questions

### 1. What is sentiment analysis?
**Answer:** Sentiment analysis is a natural language processing technique used to determine the emotional tone or attitude expressed in a piece of text. It classifies text as positive, negative, or neutral.

### 2. What technologies did you use in this project?
**Answer:** I used Python as the programming language, Flask for the web framework, scikit-learn for machine learning, pandas for data manipulation, and HTML/CSS for the frontend.

### 3. Explain the project architecture.
**Answer:** The project follows a three-tier architecture:
- Data layer: CSV dataset and preprocessing
- Model layer: Machine learning model training and prediction
- Presentation layer: Flask web application with HTML templates

## Machine Learning Questions

### 4. Why did you choose Naive Bayes for sentiment analysis?
**Answer:** Naive Bayes is effective for text classification tasks like sentiment analysis because:
- It works well with high-dimensional data (like TF-IDF vectors)
- It's computationally efficient and fast to train
- It performs well on text data despite the "naive" assumption
- It's less prone to overfitting compared to more complex models

### 5. What is TF-IDF and why is it used?
**Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. It's used because:
- It converts text into numerical vectors
- It gives higher weight to rare but important words
- It reduces the impact of common words (like "the", "is")

### 6. How does text preprocessing work in your project?
**Answer:** Text preprocessing involves:
- Converting text to lowercase
- Removing special characters and punctuation using regex
- Tokenization (handled by TF-IDF vectorizer)
- The goal is to normalize text and reduce noise

## Web Development Questions

### 7. Why did you choose Flask over other frameworks?
**Answer:** Flask was chosen because:
- It's lightweight and flexible
- Easy to set up and deploy
- Perfect for small to medium-sized applications
- Provides just the essentials without unnecessary complexity

### 8. How does the Flask routing work in your application?
**Answer:** The application uses a single route "/" that handles both GET and POST requests:
- GET: Displays the form
- POST: Processes the form data, makes prediction, and displays result
- The same template is used for both cases with conditional rendering

### 9. Explain the template rendering process.
**Answer:** Flask uses Jinja2 templating:
- Templates are stored in the `templates/` directory
- Variables are passed from the Python code to the template
- Conditional logic (like `{% if result %}`) controls what is displayed
- This allows dynamic content generation

## Technical Implementation Questions

### 10. How do you handle model persistence?
**Answer:** I use Python's pickle module to serialize and save the trained model and vectorizer:
- `pickle.dump(model, open("model.pkl", "wb"))`
- `pickle.load(open("model.pkl", "rb"))`
- This allows the web app to load the pre-trained model without retraining

### 11. What happens when a user submits a review?
**Answer:** The process is:
1. User enters text and clicks submit
2. Flask receives POST request with form data
3. Text is preprocessed (cleaned)
4. Text is vectorized using the loaded TF-IDF vectorizer
5. Model makes prediction
6. Result is passed to template and displayed

### 12. How would you improve the model's accuracy?
**Answer:** Several ways to improve accuracy:
- Increase dataset size and diversity
- Use more advanced algorithms (SVM, Random Forest, LSTM)
- Implement feature engineering (n-grams, word embeddings)
- Add domain-specific preprocessing
- Use ensemble methods

## Project Management Questions

### 13. What challenges did you face during development?
**Answer:** Key challenges included:
- Handling small dataset limitations
- Ensuring proper model serialization
- Creating an intuitive user interface
- Managing dependencies and environment setup

### 14. How would you deploy this application to production?
**Answer:** For production deployment:
- Use a WSGI server like Gunicorn
- Set up a reverse proxy with Nginx
- Containerize with Docker
- Deploy to cloud platforms (Heroku, AWS, etc.)
- Implement proper logging and monitoring

### 15. What are the limitations of your current implementation?
**Answer:** Current limitations:
- Small training dataset
- Simple model (Naive Bayes)
- No user authentication or history
- Single language support
- No batch processing capabilities

## Advanced Questions

### 16. How would you handle imbalanced datasets?
**Answer:** For imbalanced sentiment data:
- Use class weighting in the model
- Implement oversampling (SMOTE) or undersampling
- Collect more data for minority classes
- Use evaluation metrics like F1-score instead of accuracy

### 17. Explain the difference between stemming and lemmatization.
**Answer:** 
- **Stemming**: Removes suffixes to reduce words to their root form (e.g., "running" → "run"). It's fast but can be inaccurate.
- **Lemmatization**: Uses linguistic knowledge to reduce words to their base or dictionary form (e.g., "better" → "good"). It's more accurate but slower.

### 18. How would you implement real-time sentiment analysis?
**Answer:** For real-time analysis:
- Use streaming data sources (Twitter API, etc.)
- Implement asynchronous processing with Celery
- Use WebSockets for live updates
- Optimize model inference speed
- Implement caching for frequent queries

## Code Quality Questions

### 19. How do you ensure code quality in this project?
**Answer:** Code quality measures:
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings and comments
- Modular code structure
- Error handling and validation

### 20. What testing strategies would you implement?
**Answer:** Testing strategies:
- Unit tests for preprocessing functions
- Integration tests for Flask routes
- Model validation tests
- UI testing with Selenium
- Performance testing for model inference

## Future Enhancements

### 21. What features would you add next?
**Answer:** Future enhancements:
- Multi-language support
- User accounts and review history
- Batch processing for multiple reviews
- Sentiment visualization (charts, graphs)
- API endpoints for third-party integration
- Advanced NLP features (emotion detection, sarcasm detection)

### 22. How would you scale this application?
**Answer:** Scaling strategies:
- Implement caching (Redis)
- Use load balancers
- Database integration for large datasets
- Microservices architecture
- Cloud-native deployment with Kubernetes
- Model versioning and A/B testing

## Domain Knowledge Questions

### 23. What are some real-world applications of sentiment analysis?
**Answer:** Real-world applications:
- Social media monitoring
- Customer feedback analysis
- Brand reputation management
- Market research
- Political sentiment tracking
- Product review analysis
- Chatbot response optimization

### 24. How does sentiment analysis differ from emotion detection?
**Answer:** 
- **Sentiment Analysis**: Classifies text as positive/negative/neutral
- **Emotion Detection**: Identifies specific emotions (joy, anger, sadness, fear, etc.)
- Sentiment is broader; emotion detection is more granular

### 25. What ethical considerations are there in sentiment analysis?
**Answer:** Ethical considerations:
- Privacy concerns with user data
- Bias in training data leading to unfair predictions
- Potential misuse for manipulation or surveillance
- Transparency in AI decision-making
- Handling of sensitive or controversial content
