"""
Book Review Scraper & Sentiment Tester
Tests Amazon Food Review sentiment model on book reviews
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests
from time import sleep
import random

# For text preprocessing (matching your original pipeline)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

class BookReviewScraper:
    """Scraper for Goodreads book reviews"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def scrape_goodreads_book(self, book_url, max_reviews=20):
        """
        Scrape reviews from a Goodreads book page
        
        Parameters:
        -----------
        book_url : str
            URL of the Goodreads book page
        max_reviews : int
            Maximum number of reviews to scrape
            
        Returns:
        --------
        pd.DataFrame with columns: Review, Rating
        """
        try:
            response = requests.get(book_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            reviews = []
            ratings = []
            
            # Find review sections (this selector may need updating based on Goodreads' current structure)
            review_divs = soup.find_all('div', class_='ReviewText', limit=max_reviews)
            rating_divs = soup.find_all('span', class_='staticStars', limit=max_reviews)
            
            for review_div, rating_div in zip(review_divs, rating_divs):
                # Extract review text
                review_text = review_div.get_text(strip=True)
                
                # Extract rating (e.g., "it was amazing" = 5 stars)
                rating_text = rating_div.get('title', '')
                rating = self._parse_rating(rating_text)
                
                reviews.append(review_text)
                ratings.append(rating)
                
            return pd.DataFrame({
                'Review': reviews,
                'Rating': ratings
            })
            
        except Exception as e:
            print(f"Error scraping: {e}")
            return pd.DataFrame()
    
    def _parse_rating(self, rating_text):
        """Convert Goodreads rating text to numeric score"""
        rating_map = {
            'it was amazing': 5,
            'really liked it': 4,
            'liked it': 3,
            'it was ok': 2,
            'did not like it': 1
        }
        return rating_map.get(rating_text.lower(), 3)
    
    def scrape_amazon_books(self, asin, max_reviews=20):
        """
        Scrape reviews from Amazon book page using ASIN
        
        Parameters:
        -----------
        asin : str
            Amazon Standard Identification Number
        max_reviews : int
            Maximum number of reviews to scrape
        """
        base_url = f"https://www.amazon.com/product-reviews/{asin}"
        
        try:
            response = requests.get(base_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            reviews = []
            ratings = []
            
            # Find review containers
            review_divs = soup.find_all('div', {'data-hook': 'review'}, limit=max_reviews)
            
            for review_div in review_divs:
                # Extract review text
                review_body = review_div.find('span', {'data-hook': 'review-body'})
                if review_body:
                    review_text = review_body.get_text(strip=True)
                    reviews.append(review_text)
                
                # Extract rating
                rating_span = review_div.find('i', {'data-hook': 'review-star-rating'})
                if rating_span:
                    rating_text = rating_span.get_text(strip=True)
                    rating = float(rating_text.split()[0])
                    ratings.append(rating)
            
            return pd.DataFrame({
                'Review': reviews,
                'Rating': ratings
            })
            
        except Exception as e:
            print(f"Error scraping Amazon: {e}")
            return pd.DataFrame()


class TextPreprocessor:
    """
    Text preprocessing pipeline matching your Amazon Food Review project
    """
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_html(self, text):
        """Remove HTML tags"""
        if pd.isna(text):
            return ""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', str(text))
        return text
    
    def clean_text(self, text):
        """Clean and preprocess text - matching your pipeline"""
        # Remove HTML
        text = self.clean_html(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word, pos='v')  # Lemmatize as verb
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def preprocess_dataframe(self, df):
        """Apply preprocessing to entire dataframe"""
        df['Cleaned_Review'] = df['Review'].apply(self.clean_text)
        return df


class SentimentModelTester:
    """
    Test your trained sentiment models on new book review data
    """
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize with saved model and vectorizer
        
        Parameters:
        -----------
        model_path : str
            Path to saved model (.pkl file)
        vectorizer_path : str
            Path to saved TF-IDF vectorizer (.pkl file)
        """
        self.model = None
        self.vectorizer = None
        
        if model_path:
            import joblib
            self.model = joblib.load(model_path)
            print(f"✓ Loaded model from {model_path}")
            
        if vectorizer_path:
            import joblib
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"✓ Loaded vectorizer from {vectorizer_path}")
    
    def predict_sentiment(self, reviews_df):
        """
        Predict sentiment for reviews
        
        Parameters:
        -----------
        reviews_df : pd.DataFrame
            DataFrame with 'Cleaned_Review' column
            
        Returns:
        --------
        pd.DataFrame with predictions
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first!")
        
        # Vectorize the cleaned reviews
        X = self.vectorizer.transform(reviews_df['Cleaned_Review'])
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Add to dataframe
        reviews_df['Predicted_Sentiment'] = predictions
        reviews_df['Predicted_Sentiment_Label'] = reviews_df['Predicted_Sentiment'].map({
            1: 'Positive',
            0: 'Negative'
        })
        reviews_df['Confidence_Positive'] = probabilities[:, 1]
        reviews_df['Confidence_Negative'] = probabilities[:, 0]
        
        return reviews_df
    
    def evaluate_performance(self, reviews_df):
        """
        Evaluate model performance if ground truth is available
        
        Parameters:
        -----------
        reviews_df : pd.DataFrame
            Must have 'Rating' and 'Predicted_Sentiment' columns
        """
        # Convert ratings to binary (matching your original: 4-5 = positive, 1-3 = negative)
        reviews_df['Actual_Sentiment'] = (reviews_df['Rating'] >= 4).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(reviews_df['Actual_Sentiment'], 
                                 reviews_df['Predicted_Sentiment'])
        
        print("=" * 60)
        print("MODEL PERFORMANCE ON BOOK REVIEWS")
        print("=" * 60)
        print(f"\n📊 Overall Accuracy: {accuracy:.2%}")
        print("\n📈 Classification Report:")
        print(classification_report(reviews_df['Actual_Sentiment'], 
                                   reviews_df['Predicted_Sentiment'],
                                   target_names=['Negative', 'Positive']))
        
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(reviews_df['Actual_Sentiment'], 
                            reviews_df['Predicted_Sentiment'])
        print(cm)
        
        # Domain transfer analysis
        print("\n" + "=" * 60)
        print("DOMAIN TRANSFER ANALYSIS (Food Reviews → Book Reviews)")
        print("=" * 60)
        
        # Calculate misclassification patterns
        misclassified = reviews_df[reviews_df['Actual_Sentiment'] != reviews_df['Predicted_Sentiment']]
        
        print(f"\n⚠️  Misclassification Rate: {len(misclassified)/len(reviews_df):.2%}")
        print(f"   - False Positives (predicted positive, actually negative): {len(misclassified[misclassified['Predicted_Sentiment']==1])}")
        print(f"   - False Negatives (predicted negative, actually positive): {len(misclassified[misclassified['Predicted_Sentiment']==0])}")
        
        return accuracy


def create_sample_book_reviews():
    """
    Create sample book reviews for testing (since web scraping may be blocked)
    """
    sample_reviews = [
        # Positive reviews
        ("This book was absolutely amazing! The plot kept me engaged from start to finish. Highly recommend!", 5),
        ("A masterpiece of modern literature. The author's writing style is captivating and beautiful.", 5),
        ("Couldn't put it down! The characters were so well-developed and the story was compelling.", 5),
        ("One of the best books I've read this year. The ending was perfect and satisfying.", 4),
        ("Really enjoyed this read. Great character development and interesting plot twists.", 4),
        ("A good book overall. The pacing was excellent and kept me interested throughout.", 4),
        
        # Negative reviews
        ("Disappointing. The plot was predictable and the characters felt flat and uninteresting.", 2),
        ("I couldn't finish this book. The writing was poor and the story didn't engage me at all.", 1),
        ("Not what I expected. The book dragged on and I found myself bored halfway through.", 2),
        ("Weak storyline and underdeveloped characters. Would not recommend to others.", 2),
        ("The author tried too hard to be clever. The result was confusing and frustrating to read.", 1),
        ("Terrible book. Complete waste of time and money. Save yourself the trouble.", 1),
        
        # Mixed reviews
        ("It was okay. Some parts were good but overall just average. Nothing special.", 3),
        ("Decent read but not memorable. The story was fine but didn't leave much impact.", 3),
        ("Had potential but fell short. Some interesting ideas but poor execution.", 3),
    ]
    
    df = pd.DataFrame(sample_reviews, columns=['Review', 'Rating'])
    return df


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("BOOK REVIEW SENTIMENT ANALYSIS TESTER")
    print("Testing Amazon Food Review model on Book Reviews")
    print("=" * 60)
    
    # Create sample data (or use scraper if web access is available)
    print("\n📚 Loading sample book reviews...")
    book_reviews = create_sample_book_reviews()
    print(f"   Loaded {len(book_reviews)} reviews")
    
    # Preprocess
    print("\n🔧 Preprocessing reviews...")
    preprocessor = TextPreprocessor()
    book_reviews = preprocessor.preprocess_dataframe(book_reviews)
    
    # Display sample
    print("\n📝 Sample preprocessed review:")
    print(f"   Original: {book_reviews.iloc[0]['Review'][:100]}...")
    print(f"   Cleaned:  {book_reviews.iloc[0]['Cleaned_Review'][:100]}...")
    
    print("\n✅ Ready for model testing!")
    print("\nTo test your model, run:")
    print("   tester = SentimentModelTester('amazon_sentiment_final_lr_tfidf.pkl', 'tfidf_vectorizer.pkl')")
    print("   results = tester.predict_sentiment(book_reviews)")
    print("   tester.evaluate_performance(results)")
