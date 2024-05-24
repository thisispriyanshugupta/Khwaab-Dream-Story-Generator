import praw
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Reddit API credentials
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

# Specify the subreddit to scrape
subreddit_name = 'Dreams'
subreddit = reddit.subreddit(subreddit_name)

# Define the number of posts to scrape
num_posts_to_scrape = 100  # Adjust this number as needed

# Create an empty list to store dream descriptions and metadata
dream_data = []

# Initialize dream ID counter
dream_id_counter = 1

# Scrape posts from the subreddit
for submission in subreddit.new(limit=num_posts_to_scrape):
    # Extract the title and selftext (post content) of each submission
    title = submission.title
    selftext = submission.selftext

    # Combine the title and selftext to get the dream description
    dream_description = title + '\n' + selftext

    # Tokenize the dream description and remove stopwords
    tokens = word_tokenize(dream_description)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

    # Extract keywords based on frequency
    keyword_counter = Counter(filtered_tokens)
    top_keywords = keyword_counter.most_common(5)  # Get top 5 keywords

    # Remove the word "dream" from dreamer's comments if present
    dreamer_comments = ', '.join([keyword for keyword, _ in top_keywords if keyword.lower() != 'dream'])

    # Append dream data to the list with incremented dream ID
    dream_data.append({'Dream ID': dream_id_counter, 'Dream Description': dream_description, 'Dreamer Comments': dreamer_comments})

    # Increment dream ID counter
    dream_id_counter += 1

# Save dream data to a CSV file
csv_file_path = 'dream_data.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Dream ID', 'Dream Description', 'Dreamer Comments']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for dream in dream_data:
        writer.writerow(dream)

print("Dream data saved to:", csv_file_path)
