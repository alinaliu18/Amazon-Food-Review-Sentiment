"""
CRITICAL FIX: Save Your TF-IDF Vectorizer
===========================================

Your original notebook saved the model but NOT the vectorizer.
To test on new data, you MUST save the vectorizer too!

Add this cell to your original AmazonFoodReviewProject.ipynb:
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# After you create and fit your vectorizer, add this:
# (Look for where you did: vectorizer.fit_transform(X_train))

"""
Example from your notebook - find the cell where you created the vectorizer:

# This is what you probably did:
vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train['Cleaned_Review'])
X_test_tfidf = vectorizer.transform(X_test['Cleaned_Review'])

# ADD THIS LINE after fitting:
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✅ Vectorizer saved!")

"""

# Quick check script to verify your saved files
def verify_saved_files():
    """Verify that all necessary files are saved"""
    import os
    
    required_files = {
        'amazon_sentiment_final_lr_tfidf.pkl': 'Logistic Regression Model (TF-IDF)',
        'tfidf_vectorizer.pkl': 'TF-IDF Vectorizer (CRITICAL!)',
    }
    
    print("=" * 60)
    print("CHECKING SAVED MODEL FILES")
    print("=" * 60)
    
    all_good = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"✅ {filename}")
            print(f"   {description}")
            print(f"   Size: {size:.2f} MB\n")
        else:
            print(f"❌ MISSING: {filename}")
            print(f"   {description}")
            print(f"   ⚠️  You need to save this from your original notebook!\n")
            all_good = False
    
    if all_good:
        print("🎉 All required files found! Ready to test on book reviews!")
    else:
        print("⚠️  Missing files! Go back to your original notebook and save them.")
    
    return all_good


if __name__ == "__main__":
    verify_saved_files()
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS TO FIX YOUR ORIGINAL NOTEBOOK")
    print("=" * 60)
    print("""
1. Open: AmazonFoodReviewProject.ipynb

2. Find the cell where you created the TF-IDF vectorizer
   (Search for: TfidfVectorizer)

3. After the line where you fit it (vectorizer.fit_transform(...))
   Add this line:
   
   joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

4. Run that cell

5. Also find where you created lr_final_tfidf model
   Make sure it's saved with:
   
   joblib.dump(lr_final_tfidf, 'amazon_sentiment_final_lr_tfidf.pkl')

6. Verify files exist by running this script again

7. Then you're ready to test on book reviews! 🚀
""")
