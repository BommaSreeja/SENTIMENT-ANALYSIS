# SENTIMENT-ANALYSIS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: BOMMA SREEJA

*INTERN ID*: CT04DL352

*DOMAIN*: DATA ANALYSIS

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

This Python script performs sentiment analysis on airline-related tweets using Natural Language Processing (NLP) and machine learning techniques. The goal of the project is to classify each tweet as positive, neutral, or negative, providing insights into public sentiment towards various airlines. The code covers the full NLP pipeline — from data loading and text preprocessing to model training and evaluation — using powerful libraries like NLTK, Pandas, Scikit-learn, Matplotlib, and Seaborn.
The process begins by importing essential libraries. Pandas is used for data manipulation, while NLTK (Natural Language Toolkit) is used for processing and cleaning raw textual data. The machine learning workflow is handled by Scikit-learn, and the results are visualized with Matplotlib and Seaborn.
The dataset is provided in a compressed ZIP file format, which the script reads directly using Python’s zipfile module. Inside the ZIP, a CSV file is extracted and loaded into a Pandas DataFrame. The script filters the dataset to keep only the relevant columns: airline_sentiment and text, which represent the sentiment label and tweet content, respectively. For simplicity and clarity, the column names are renamed to sentiment and text.
Next, the text data undergoes preprocessing using a custom preprocess_text function. This function converts all characters to lowercase, removes URLs, Twitter mentions (e.g., @username), special characters, and punctuation using regular expressions. Then it uses word_tokenize to split the text into individual words. Common English stopwords (like "the", "is", "in") are removed to focus only on meaningful words. The cleaned tokens are then rejoined into a string. This processed version of each tweet is stored in a new column called processed_text.
The dataset is then split into training and testing sets using an 80-20 ratio via train_test_split. To convert the text data into numerical format suitable for machine learning models, TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is applied using TfidfVectorizer. This technique transforms the textual data into a matrix of numerical values, giving more weight to words that are unique and informative.
The model used for sentiment classification is Multinomial Naive Bayes, a commonly used algorithm for text classification tasks due to its simplicity and efficiency. The model is trained on the TF-IDF-transformed training data and then used to predict sentiments for the test data.
To evaluate model performance, the script prints the accuracy score, classification report (including precision, recall, and F1-score for each class), and a confusion matrix. The confusion matrix is visualized using Seaborn’s heatmap, making it easy to identify how many tweets were correctly or incorrectly classified into each sentiment category.
This sentiment analysis project is applicable in areas such as brand reputation monitoring, customer feedback analysis, and social media analytics. By analyzing user-generated content like tweets, businesses can gain valuable insights into customer opinions and service quality. The project can be deployed as a standalone script, integrated into a larger analytics pipeline, or expanded into a real-time web application for monitoring public sentiment.

*OUTPUT*:

