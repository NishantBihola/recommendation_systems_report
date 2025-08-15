# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Ignore all warnings for cleaner output
warnings.filterwarnings('ignore')

# --- DATASET LOADING & CLEANUP ---
def load_and_clean_data():
    """
    Attempts to load the books.csv dataset and handles potential parsing errors.
    If the file is not found, it uses a mock dataset for demonstration.
    """
    data_string = """
bookID,title,authors,average_rating,ratings_count,text_reviews_count,language_code
1,The Hunger Games (The Hunger Games, #1),Suzanne Collins,4.34,4780653,155254,eng
2,Harry Potter and the Sorcerer's Stone (Harry Potter, #1),J.K. Rowling,4.44,4602479,75867,eng
3,Twilight (Twilight, #1),Stephenie Meyer,3.58,4355799,94265,eng
4,To Kill a Mockingbird,Harper Lee,4.25,2631557,59138,eng
5,The Great Gatsby,F. Scott Fitzgerald,3.89,2683103,40182,eng
6,The Book Thief,Markus Zusak,4.37,1316086,10484,eng
7,1984,George Orwell,4.14,1956877,33027,eng
8,The Lord of the Rings,J.R.R. Tolkien,4.50,1780516,14986,eng
9,The Hobbit,J.R.R. Tolkien,4.27,2466037,34438,eng
10,Pride and Prejudice,Jane Austen,4.25,2341492,44813,eng
11,The Hitchhiker's Guide to the Galaxy (Hitchhiker's Guide to the Galaxy, #1),Douglas Adams,4.34,1266014,24204,eng
12,The Da Vinci Code (Robert Langdon, #1),Dan Brown,3.78,1600880,48828,eng
13,Angels & Demons (Robert Langdon, #1),Dan Brown,3.89,1434310,30263,eng
14,A Wrinkle in Time (A Wrinkle in Time Quintet, #1),Madeleine L'Engle,4.00,1034379,15016,eng
15,The Catcher in the Rye,J.D. Salinger,3.80,2403987,41968,eng
"""
    from io import StringIO
    try:
        # The 'on_bad_lines='skip'' parameter tells pandas to skip rows
        # that have too many fields, which is the cause of your ParserError.
        print("Attempting to load 'books.csv'...")
        df = pd.read_csv('books.csv', on_bad_lines='skip')
        print("Dataset 'books.csv' loaded successfully.")
    except FileNotFoundError:
        print("Dataset 'books.csv' not found. Using mock data for demonstration.")
        df = pd.read_csv(StringIO(data_string))

    # Remove duplicate entries based on book title to avoid skewed results
    df.drop_duplicates(subset=['title'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Define a function for the Popularity-Based Recommender
def popularity_recommender(df, num_recommendations=5):
    """
    Recommends books based on a weighted popularity score.
    A book must have at least 'm' ratings to be considered.
    """
    print("\n--- Popularity-Based Recommendations ---")
    
    # Calculate the mean rating of all books
    C = df['average_rating'].mean()
    
    # Define 'm', the minimum number of ratings required to be considered.
    # We use the 80th percentile to ensure only widely-rated books are included.
    m = df['ratings_count'].quantile(0.80)
    
    # Filter the DataFrame to include only books that meet the ratings threshold
    popular_books = df[df['ratings_count'] >= m].copy()
    
    # Define the weighted rating formula
    def weighted_rating(row):
        v = row['ratings_count']
        R = row['average_rating']
        return (v/(v+m) * R) + (m/(v+m) * C)
    
    # Apply the formula to create a new 'score' column
    popular_books['score'] = popular_books.apply(weighted_rating, axis=1)
    
    # Sort the books by their weighted score in descending order
    popular_books = popular_books.sort_values(by='score', ascending=False)
    
    # Return the top N recommendations
    return popular_books[['title', 'authors', 'average_rating', 'ratings_count']].head(num_recommendations)

# Define a function for the Content-Based Recommender
def content_based_recommender(df, title, num_recommendations=5):
    """
    Recommends books based on text similarity of their title and authors.
    """
    print(f"\n--- Content-Based Recommendations for '{title}' ---")
    
    # Check if the book title exists in the DataFrame
    # Using .isin() is a robust way to check for a value in a Series
    if not df['title'].isin([title]).any():
        print(f"'{title}' not found in the dataset. Please check the spelling.")
        return pd.DataFrame() # Return an empty DataFrame

    # Create a "soup" of features to be used for the recommender
    # This combines the title and authors into a single string
    # We'll also strip leading/trailing whitespace to be safe
    df['soup'] = df['title'].str.strip() + ' ' + df['authors'].str.strip().str.replace(' ', '')
    
    # Create the TF-IDF Vectorizer to convert the text to a numerical matrix
    # TfidfVectorizer gives more weight to words that are rare and important
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    
    # Compute the cosine similarity matrix. This is the core of the recommender.
    # It calculates a score between 0 and 1 for every pair of books,
    # where a score of 1 means they are identical in content.
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the book that matches the title
    book_index = df[df['title'] == title].index[0]
    
    # Get the similarity scores for all other books with that book
    sim_scores = list(enumerate(cosine_sim[book_index]))
    
    # Sort the list of tuples based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top N most similar books
    # We slice from 1 to avoid recommending the same book
    sim_scores = sim_scores[1:num_recommendations + 1]
    
    # Get the book indices
    book_indices = [i[0] for i in sim_scores]
    
    # Return the top N recommendations as a DataFrame
    return df.iloc[book_indices][['title', 'authors', 'average_rating', 'ratings_count']]

# --- EXECUTION ---
if __name__ == "__main__":
    data = load_and_clean_data()
    
    if not data.empty:
        # Print a list of titles to help the user find the correct string
        print("\nHere are the first 20 book titles in your dataset to help you with the search:")
        for i, title in enumerate(data['title'].head(20)):
            print(f"{i+1}. {title}")
        
        # Get popularity-based recommendations
        top_popular_books = popularity_recommender(data, num_recommendations=5)
        print(top_popular_books.to_string())

        # Get content-based recommendations for a specific book
        # You'll need to update this string with the exact title from the list above.
        # For example, if "The Hunger Games" is formatted as "The Hunger Games", use that.
        # Based on your previous output, a string like "Harry Potter and the Half-Blood Prince (Harry Potter #6)"
        # is a good example of the kind of string you'll need to use.
        recommended_books = content_based_recommender(data, title="The Hunger Games (The Hunger Games, #1)", num_recommendations=5)
        print(recommended_books.to_string())

        # Get content-based recommendations for another book
        # The search for this one seems to be working for you, so it's a good example to keep.
        recommended_books_2 = content_based_recommender(data, title="The Lord of the Rings", num_recommendations=5)
        print(recommended_books_2.to_string())
    else:
        print("Could not load data. Please check the dataset.")

