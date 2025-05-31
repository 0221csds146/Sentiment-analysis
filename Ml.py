import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
import re
from dotenv import load_dotenv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords


# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
# AIzaSyCQ-pqpmyRFNkDA826BvsPynJWYxIckQGQ
# Load environment variables
# https://www.youtube.com/watch?v=d558tMKjvgc&t=851s
load_dotenv()

class YouTubeMLSentimentAnalyzer:
    def __init__(self, api_key=None, model_type="transformer"):
        """
        Initialize the YouTube API client and ML sentiment analysis model
        :param api_key: Your YouTube API key
        :param model_type: Type of model to use ('transformer' or 'traditional')
        """
        if api_key is None:
            api_key = os.getenv('YOUTUBE_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided. Set YOUTUBE_API_KEY environment variable or pass key directly.")
        
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.comments_data = []
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        
        # Initialize sentiment model
        if model_type == "transformer":
            print("Loading pretrained transformer model...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print(f"Model loaded and using {self.device}")
        else:
            self.vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def extract_video_id(self, video_url):
        """
        Extract the video ID from a YouTube URL
        :param video_url: Full YouTube URL
        :return: YouTube video ID
        """
        video_id = None
        if 'youtube.com/watch' in video_url:
            video_id = video_url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1].split('?')[0]
        
        return video_id
    
    def get_comments(self, video_url, max_comments=100):
        """
        Get comments from a YouTube video
        :param video_url: YouTube video URL or ID
        :param max_comments: Maximum number of comments to retrieve
        :return: DataFrame with comments
        """
        # Check if input is a URL or a video ID
        if 'youtube.com' in video_url or 'youtu.be' in video_url:
            video_id = self.extract_video_id(video_url)
        else:
            video_id = video_url
            
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
        
        self.comments_data = []
        next_page_token = None
        
        print(f"Fetching comments for video ID: {video_id}")
        
        # Keep fetching comments until we have enough or no more are available
        while len(self.comments_data) < max_comments:
            try:
                response = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(self.comments_data)),
                    pageToken=next_page_token,
                    textFormat='plainText'
                ).execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    self.comments_data.append({
                        'author': comment['authorDisplayName'],
                        'text': comment['textDisplay'],
                        'likeCount': comment['likeCount'],
                        'publishedAt': comment['publishedAt']
                    })
                
                # Check if there are more pages
                next_page_token = response.get('nextPageToken')
                if not next_page_token or len(self.comments_data) >= max_comments:
                    break
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                break
        
        print(f"Retrieved {len(self.comments_data)} comments")
        return pd.DataFrame(self.comments_data)

    def clean_text(self, text):
        """
        Clean the text by removing HTML tags, URLs, etc.
        :param text: Text to clean
        :return: Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def predict_sentiment_transformer(self, texts):
        """
        Predict sentiment using a transformer model (DistilBERT fine-tuned on SST-2)
        :param texts: List of text strings
        :return: Dictionary with sentiment scores
        """
        results = {'polarity': [], 'sentiment': []}
        
        # Process in batches to avoid memory issues
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process outputs
            probs = softmax(outputs.logits, dim=1)
            probs = probs.cpu().numpy()
            
            # Get polarity scores (-1 to 1 scale, where -1 is negative and 1 is positive)
            polarities = (probs[:, 1] - probs[:, 0]) * 2 - 1
            results['polarity'].extend(polarities)
            
            # Get sentiment labels
            sentiments = ['negative' if p <= -0.1 else ('positive' if p >= 0.1 else 'neutral') for p in polarities]
            results['sentiment'].extend(sentiments)
            
        return results

    def analyze_sentiment(self, df=None):
        """
        Perform sentiment analysis on the comments using ML model
        :param df: DataFrame with comments. If None, use the stored comments
        :return: DataFrame with sentiment analysis
        """
        if df is None:
            if not self.comments_data:
                raise ValueError("No comments data available. Call get_comments first.")
            df = pd.DataFrame(self.comments_data)
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Skip empty texts
        df = df[df['cleaned_text'].str.strip() != '']
        
        if self.model_type == "transformer":
            # Predict sentiment with transformer model
            print("Predicting sentiment with transformer model...")
            results = self.predict_sentiment_transformer(df['cleaned_text'].tolist())
            df['polarity'] = results['polarity']
            df['sentiment'] = results['sentiment']
        else:
            # Traditional ML approach would be implemented here
            # For simplicity, we'll use a placeholder implementation
            print("Traditional ML approach not implemented in this example")
            # You would typically:
            # 1. Train a model on a sentiment dataset
            # 2. Use the trained model to predict sentiment on new data
        
        # Convert published date string to datetime objects
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        
        return df
    
    def get_sentiment_summary(self, df=None):
        """
        Get a summary of the sentiment analysis
        :param df: DataFrame with sentiment analysis
        :return: Dictionary with sentiment summary
        """
        if df is None:
            if not hasattr(self, 'sentiment_df'):
                raise ValueError("No sentiment analysis available")
            df = self.sentiment_df
            
        summary = {
            'total_comments': len(df),
            'avg_polarity': df['polarity'].mean(),
            'sentiment_counts': df['sentiment'].value_counts().to_dict(),
            'sentiment_percentage': (df['sentiment'].value_counts(normalize=True) * 100).to_dict(),
            'top_positive': df[df['sentiment'] == 'positive'].nlargest(5, 'polarity')[['text', 'polarity']].to_dict('records'),
            'top_negative': df[df['sentiment'] == 'negative'].nsmallest(5, 'polarity')[['text', 'polarity']].to_dict('records'),
        }
        
        return summary
    
    def visualize_sentiment(self, df=None, save_path=None):
        """
        Visualize the sentiment analysis results
        :param df: DataFrame with sentiment analysis
        :param save_path: Path to save the visualization
        """
        if df is None:
            if not hasattr(self, 'sentiment_df'):
                raise ValueError("No sentiment analysis available")
            df = self.sentiment_df
        
        # Set up the figure
        plt.figure(figsize=(18, 14))
        plt.style.use('ggplot')
        
        # Plot 1: Sentiment Distribution
        plt.subplot(2, 3, 1)
        sentiment_counts = df['sentiment'].value_counts()
        colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        ax = sentiment_counts.plot(kind='bar', color=[colors[x] for x in sentiment_counts.index])
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Comments', fontsize=12)
        plt.xticks(rotation=0, fontsize=10)
        
        # Add percentage labels on the bars
        total = len(df)
        for i, v in enumerate(sentiment_counts):
            ax.text(i, v + 0.5, f"{v/total:.1%}", ha='center', fontsize=10)
        
        # Plot 2: Polarity Distribution
        plt.subplot(2, 3, 2)
        sns.histplot(df['polarity'], bins=20, kde=True, color='#3498db')
        plt.title('Polarity Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Polarity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Plot 3: Word Cloud for Positive Comments
        plt.subplot(2, 3, 3)
        positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_text'])
        if positive_text.strip():
            stop_words = set(stopwords.words('english'))
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                  stopwords=stop_words, max_words=100, colormap='YlGn').generate(positive_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Positive Comments Word Cloud', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No positive comments', ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Plot 4: Word Cloud for Negative Comments
        plt.subplot(2, 3, 4)
        negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_text'])
        if negative_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                  stopwords=stop_words, max_words=100, colormap='Reds').generate(negative_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Negative Comments Word Cloud', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No negative comments', ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Plot 5: Sentiment Over Time (if applicable)
        plt.subplot(2, 3, 5)
        if 'publishedAt' in df.columns:
            # Group by date and calculate average polarity
            df['date'] = df['publishedAt'].dt.date
            time_data = df.groupby('date')['polarity'].mean().reset_index()
            
            if len(time_data) > 1:  # Only if we have multiple dates
                plt.plot(time_data['date'], time_data['polarity'], marker='o', linestyle='-', color='#9b59b6')
                plt.title('Sentiment Trend Over Time', fontsize=14, fontweight='bold')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Average Polarity', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'Insufficient time data', ha='center', va='center', fontsize=12)
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'No time data available', ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Plot 6: Likes vs. Sentiment
        plt.subplot(2, 3, 6)
        if 'likeCount' in df.columns:
            # Create a scatter plot of like count vs. polarity
            plt.scatter(df['polarity'], np.log1p(df['likeCount']), alpha=0.6, c=df['polarity'], 
                       cmap='coolwarm', edgecolors='w', linewidth=0.5)
            plt.title('Sentiment vs. Log(Likes)', fontsize=14, fontweight='bold')
            plt.xlabel('Polarity', fontsize=12)
            plt.ylabel('Log(Like Count + 1)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.colorbar(label='Polarity')
        else:
            plt.text(0.5, 0.5, 'No like count data', ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        
        
    
    def extract_topics(self, df=None, num_topics=5, num_words=10):
        """
        Extract main topics from comments using NMF topic modeling
        :param df: DataFrame with comments
        :param num_topics: Number of topics to extract
        :param num_words: Number of words per topic
        :return: List of topics
        """
        if df is None:
            if not hasattr(self, 'sentiment_df'):
                raise ValueError("No sentiment analysis available")
            df = self.sentiment_df
        
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import NMF
            
            # Initialize the count vectorizer
            count_vectorizer = CountVectorizer(
                max_df=0.95, min_df=2, max_features=1000, stop_words='english'
            )
            
            # Fit and transform the comments
            comment_counts = count_vectorizer.fit_transform(df['cleaned_text'])
            
            # Extract the vocabulary
            vocab = count_vectorizer.get_feature_names_out()
            
            # Use NMF for topic modeling
            nmf_model = NMF(n_components=num_topics, random_state=42)
            nmf_topics = nmf_model.fit_transform(comment_counts)
            
            # Get the top words for each topic
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[:-num_words-1:-1]
                top_words = [vocab[i] for i in top_words_idx]
                topics.append({
                    'id': topic_idx,
                    'words': top_words,
                    'weight': float(nmf_topics[:, topic_idx].sum() / nmf_topics.sum())
                })
            
            return topics
            
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
    
    def analyze_video(self, video_url, max_comments=100):
        """
        Complete analysis pipeline for a YouTube video
        :param video_url: YouTube video URL or ID
        :param max_comments: Maximum number of comments to retrieve
        :return: DataFrame with sentiment analysis
        """
        # Get comments
        comments_df = self.get_comments(video_url, max_comments)
        
        if len(comments_df) == 0:
            print("No comments retrieved. Check the video URL and API key.")
            return None
        
        # Analyze sentiment
        self.sentiment_df = self.analyze_sentiment(comments_df)
        
        # Get summary
        self.summary = self.get_sentiment_summary(self.sentiment_df)
        
        # Extract topics
        try:
            self.topics = self.extract_topics(self.sentiment_df)
        except Exception as e:
            print(f"Warning: Topic extraction failed: {e}")
            self.topics = []
        
        # Print summary
        print("\n" + "="*50)
        print("YOUTUBE COMMENT SENTIMENT ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total Comments Analyzed: {self.summary['total_comments']}")
        print(f"Average Sentiment (Polarity): {self.summary['avg_polarity']:.2f} (-1 very negative, 1 very positive)")
        print("\nSentiment Distribution:")
        for sentiment, count in self.summary['sentiment_counts'].items():
            percentage = self.summary['sentiment_percentage'][sentiment]
            print(f"  {sentiment.upper()}: {count} comments ({percentage:.1f}%)")
        
        if self.topics:
            print("\nMain Topics Discussed:")
            for topic in self.topics:
                print(f"  Topic {topic['id']+1} ({topic['weight']:.1%}): {', '.join(topic['words'][:5])}")
        
        print("\nTop Positive Comments:")
        for i, comment in enumerate(self.summary['top_positive'][:3], 1):
            print(f"  {i}. \"{comment['text'][:100]}...\" (polarity: {comment['polarity']:.2f})")
        
        print("\nTop Negative Comments:")
        for i, comment in enumerate(self.summary['top_negative'][:3], 1):
            print(f"  {i}. \"{comment['text'][:100]}...\" (polarity: {comment['polarity']:.2f})")
        
        print("="*50)
        
        return self.sentiment_df
    
    def export_results(self, output_folder="youtube_analysis"):
        """
        Export analysis results to files
        :param output_folder: Folder to save results
        """
        if not hasattr(self, 'sentiment_df'):
            raise ValueError("No analysis results to export. Run analyze_video first.")
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export comments data to CSV
        csv_path = os.path.join(output_folder, f"youtube_comments_{timestamp}.csv")
        self.sentiment_df.to_csv(csv_path,encoding='utf-8', index=False)
        print(f"Comments data exported to {csv_path}")
        
        # Export summary to text file
        summary_path = os.path.join(output_folder, f"analysis_summary_{timestamp}.txt")
        with open(summary_path, 'w',encoding='utf-8') as f:
            f.write("YOUTUBE COMMENT SENTIMENT ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Total Comments Analyzed: {self.summary['total_comments']}\n")
            f.write(f"Average Sentiment (Polarity): {self.summary['avg_polarity']:.2f}\n\n")
            
            f.write("Sentiment Distribution:\n")
            for sentiment, count in self.summary['sentiment_counts'].items():
                percentage = self.summary['sentiment_percentage'][sentiment]
                f.write(f"  {sentiment.upper()}: {count} comments ({percentage:.1f}%)\n")
            
            if hasattr(self, 'topics') and self.topics:
                f.write("\nMain Topics Discussed:\n")
                for topic in self.topics:
                    f.write(f"  Topic {topic['id']+1}: {', '.join(topic['words'])}\n")
            
            f.write("\nTop Positive Comments:\n")
            for i, comment in enumerate(self.summary['top_positive'], 1):
                f.write(f"  {i}. \"{comment['text'][:100]}...\"\n")
            
            f.write("\nTop Negative Comments:\n")
            for i, comment in enumerate(self.summary['top_negative'], 1):
                f.write(f"  {i}. \"{comment['text'][:100]}...\"\n")
        
        print(f"Analysis summary exported to {summary_path}")
        
        # Create and save visualizations
        viz_path = os.path.join(output_folder, f"sentiment_visualization_{timestamp}.png")
        self.visualize_sentiment(save_path=viz_path)
        
        return {
            'csv_path': csv_path,
            'summary_path': summary_path,
            'visualization_path': viz_path
        }


def main():
    """
    Main function to run the YouTube sentiment analysis
    """
    print("="*50)
    print("YouTube Comment Sentiment Analyzer")
    print("="*50)
    
    # Get API key
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        api_key = input("Enter your YouTube API Key: ")
        if not api_key:
            print("API key is required. Exiting.")
            return
    
    # Choose model type
    model_type = "transformer"  # Default to transformer model
    print(f"\nUsing {model_type} model for sentiment analysis.")
    
    # Initialize analyzer
    try:
        analyzer = YouTubeMLSentimentAnalyzer(api_key=api_key, model_type=model_type)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        return
    
    # Get video URL
    video_url = input("\nEnter YouTube video URL: ")
    if not video_url:
        print("Video URL is required. Exiting.")
        return
    
    # Get number of comments to analyze
    try:
        max_comments = int(input("Enter number of comments to analyze (default: 100): ") or "100")
    except ValueError:
        print("Invalid number. Using default of 100 comments.")
        max_comments = 100
    
    # Analyze the video
    try:
        results = analyzer.analyze_video(video_url, max_comments)
        if results is None:
            print("Analysis failed. Exiting.")
            return
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return
    
    # Export results
    try:
        export = input("\nExport results to files? (y/n): ")
        if export.lower() in ['y', 'yes']:
            output_folder = input("Enter output folder (default: youtube_analysis): ") or "youtube_analysis"
            analyzer.export_results(output_folder)
    except Exception as e:
        print(f"Error exporting results: {e}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

    