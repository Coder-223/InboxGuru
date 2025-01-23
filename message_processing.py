import base64
from email_cleaning import clean_email_content  # Import your cleaning functions
import re
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

def get_header_value(message, header_name):
  """Retrieve the value of a specific header from the message."""
  for header in message["payload"]["headers"]:
      if header["name"] == header_name:
          return header["value"]
  return None  # Return None if the header is not found


def get_sender_email(message):
  """Extract the sender's email from the message object."""
  from_value = get_header_value(message, "From")
  if from_value:
      match = re.search(r"<(.+)>", from_value)
      if match:
          return match.group(1)
      else:
          return from_value
  return None


def get_message_body(message):
  """Extract and clean the message body from a message object."""
  body = ""
  if "parts" xf message["payload"]:
      for part in message["payload"]["parts"]:
          if part['mimeType'] == 'text/plain':
              body += part['body']['data']
          elif part['mimeType'] == 'text/html':
              body += part['body']['data']
  else:
      # For single-part messages
      body += message["payload"]["body"]["data"]




  # Decode the body using base64
  body = base64.urlsafe_b64decode(body).decode('utf-8')




  # Clean the email body
  cleaned_body = clean_email_content(body)
  return cleaned_body


import time


HIGH_PRIORITY_SENDER_SCORE = 5
URGENT_KEYWORD_SCORE = 3
UNREAD_SCORE = 2
RECENT_MESSAGE_SCORE = 2
RECENT_TIME_FRAME = 86400


def calculate_priority_score(message, high_priority_senders, urgency_keywords):
  priority_score = 0
  sender_email = get_sender_email(message)
  email_body = get_message_body(message)
  timestamp = int(message["internalDate"]) / 1000


  current_time = time.time()


  found_keywords = []


  is_high_priority_sender = sender_email in high_priority_senders
  if is_high_priority_sender:
      priority_score += HIGH_PRIORITY_SENDER_SCORE


     # Clean the email body for better keyword matching
  # Remove punctuation and make lowercase
  email_body_cleaned = re.sub(r'[^\w\s]', '', email_body.lower())  # Remove punctuation, convert to lowercase


  words_in_email = set(email_body_cleaned.split())  # Split on spaces


  for keyword in urgency_keywords:
      if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', email_body_cleaned):
          priority_score += URGENT_KEYWORD_SCORE
          found_keywords.append(keyword)
  is_unread = "UNREAD" in message.get("labelIds", [])
  if is_unread:
      priority_score += UNREAD_SCORE




  is_recent = current_time - timestamp < RECENT_TIME_FRAME
  if is_recent:
      priority_score += RECENT_MESSAGE_SCORE  # Fixed this line


  #ADD debugging output here in case we have issues with priority scores


  return priority_score


def add_priority_sender(high_priority_senders):
  new_sender = input("If you want to add a high-priority sender, then please enter their email address here: ")
  if new_sender and "@" in new_sender:
      high_priority_senders.add(new_sender)
      print(f"{new_sender} added to high_priority senders.")


def sort_messages_by_priority(service, messages, high_priority_senders, urgency_keywords):
  prioritized_messages = []
  for message in messages:
      msg = service.users().messages().get(userId="me", id=message["id"]).execute()
      priority_score = calculate_priority_score(msg, high_priority_senders, urgency_keywords)  # Use msg, not message
      prioritized_messages.append((msg, priority_score))
    
  prioritized_messages.sort(key=lambda x: x[1], reverse=True)


  return [msg for msg, priority_score in prioritized_messages]


def extract_email_features(message, high_priority_senders, urgency_keywords):
   sender_email = get_sender_email(message)
   email_body = get_message_body(message)
   timestamp = int(message["internalDate"]) / 1000  # Convert to seconds

   current_time = time.time()

   # 1. Feature 1: Is sender high priority?
   is_high_priority_sender = 1 if sender_email in high_priority_senders else 0


   # 2. Feature 2: Contains urgency keyword?
   email_body_cleaned = re.sub(r'[^\w\s]', '', email_body.lower())
   contains_urgency_keywords = any(
       re.search(r'\b' + re.escape(keyword.lower()) + r'\b', email_body_cleaned) for keyword in urgency_keywords
   )


   # 3. Feature 3: Sentiment polarity
   sentiment = TextBlob(email_body).sentiment.polarity  # Sentiment score between -1 and 1


   # 4. Feature 4: Email length
   email_length = len(email_body.split())


   # 5. Feature 5: Time since received
   time_since_received = current_time - timestamp


   # Return features as a dictionary
   return {
       'is_high_priority_sender': is_high_priority_sender,
       'contains_urgency_keywords': contains_urgency_keywords,
       'sentiment': sentiment,
       'email_length': email_length,
       'time_since_received': time_since_received,
   }


# To train the model, you'd collect the labels manually or from previous data:
def collect_training_data(messages, high_priority_senders, urgency_keywords):
   features = []


   for message in messages:
       features_dict = extract_email_features(message, high_priority_senders, urgency_keywords)
       features.append([features_dict['is_high_priority_sender'],
                        features_dict['contains_urgency_keywords'],
                        features_dict['sentiment'],
                        features_dict['email_length'],
                        features_dict['time_since_received']])


       # Define the label manually for now (e.g., "high" or "low" priority).
       # You can replace this with a more sophisticated method or use labeled data
       # to train the model.


   return pd.DataFrame(features, columns=['is_high_priority_sender', 'contains_urgency_keywords', 'sentiment', 'email_length', 'time_since_received'])


def extract_keywords_tfidf(messages, top_n=10):
    """Extract keywords from email messages using TF-IDF."""
    email_texts = [get_message_body(msg) for msg in messages]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(email_texts)
    feature_names = vectorizer.get_feature_names_out()

    keywords_per_email = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = row.argsort()[-top_n:][::-1]
        keywords = [feature_names[idx] for idx in top_indices]
        keywords_per_email.append(keywords)

    return keywords_per_email


def extract_keywords_rake(message_body, top_n=10):
    """Extract keywords using RAKE."""
    r = Rake()
    r.extract_keywords_from_text(message_body)
    return r.get_ranked_phrases()[:top_n]

def extract_named_entities(message_body):
    """Extract named entities using SpaCy."""
    doc = nlp(message_body)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def topic_modeling(messages, num_topics=5, num_words=10, model_type='lda'):
    # Assuming 'messages' is a list of email bodies
    email_bodies = [get_message_body(msg) for msg in messages]  # Extract email bodies
    
    # Convert the email bodies into a TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(email_bodies)
    
    # Choose the model based on the model_type argument ('lda' or 'nmf')
    if model_type == 'lda':
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    elif model_type == 'nmf':
        model = NMF(n_components=num_topics, random_state=42)
    else:
        raise ValueError("Invalid model_type. Use 'lda' or 'nmf'.")
    
    # Fit the model to the TF-IDF matrix
    model.fit(X)
    
    # Get the topics (words associated with each topic)
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(", ".join(top_words))  # Remove topic number from here
    
    return topics

def analyze_message_content(message):
    """Extract and analyze content from a message."""
    email_body = get_message_body(message)
    keywords_tfidf = extract_keywords_tfidf([message])
    keywords_rake = extract_keywords_rake(email_body)
    named_entities = extract_named_entities(email_body)

    return {
        "keywords_tfidf": keywords_tfidf,
        "keywords_rake": keywords_rake,
        "named_entities": named_entities,
    }
