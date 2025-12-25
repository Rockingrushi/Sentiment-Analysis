# Sentiment Analysis Web Application

A machine learning-powered web application for real-time sentiment analysis of text reviews using Python, Flask, and scikit-learn.

## 🚀 Features

- **Text Preprocessing**: Cleans and normalizes input text
- **Machine Learning Model**: Naive Bayes classifier with TF-IDF vectorization
- **Web Interface**: Clean, responsive Flask-based UI
- **Real-time Analysis**: Instant sentiment prediction (Positive, Negative, Neutral)

## 📋 Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to `http://127.0.0.1:5000/`

## 📁 Project Structure

```
sentiment-analysis-app/
│
├── dataset.csv              # Training dataset
├── train_model.py           # Model training script
├── app.py                   # Flask web application
├── model.pkl                # Trained model (generated)
├── vectorizer.pkl           # TF-IDF vectorizer (generated)
│
├── templates/
│   └── index.html           # Web interface template
│
├── static/
│   └── style.css            # CSS styling
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── report.md                # Detailed project report
```

## 🎯 Usage

1. Enter your text review in the input field
2. Click "Analyze Sentiment"
3. View the predicted sentiment (Positive/Negative/Neutral)

## 🤖 Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Preprocessing**: Lowercasing, regex cleaning
- **Accuracy**: ~80% on test data (varies with dataset size)

## 📊 Dataset

The model is trained on a CSV dataset with the following format:

```csv
text,sentiment
"I love this product",positive
"This is terrible",negative
"It's okay",neutral
```

## 🔧 Customization

- **Add more data**: Extend `dataset.csv` with additional training examples
- **Improve model**: Experiment with different algorithms (SVM, Random Forest, etc.)
- **Enhance UI**: Modify `templates/index.html` and `static/style.css`

## 📈 Future Enhancements

- [ ] Support for multiple languages
- [ ] Integration with social media APIs
- [ ] Batch processing capabilities
- [ ] Advanced NLP features (named entity recognition, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [Your GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Scikit-learn documentation
- Flask documentation
- Open-source NLP community
