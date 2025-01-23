from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import pandas as pd
import pickle
from message_processing import get_sender_email, get_message_body, extract_email_features, collect_training_data, calculate_priority_score, extract_keywords_tfidf, topic_modeling, analyze_message_content
from google.auth.transport.requests import Request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error

# Define the scopes for the Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_email():
  """Authenticate the user and return a service object."""
  creds = None
  # Load credentials from token.pickle if it exists
  if os.path.exists('token.pickle'):
      with open('token.pickle', 'rb') as token:
          creds = pickle.load(token)
  # If no valid credentials are available, let the user log in
  if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
          creds.refresh(Request())
      else:
          flow = InstalledAppFlow.from_client_secrets_file('/Users/michaelpancras/Desktop/Smart_Email_Message_Prioritizer/credentials.json', SCOPES)
          creds = flow.run_local_server(port=0)
      # Save the credentials for the next run
      with open('token.pickle', 'wb') as token:
          pickle.dump(creds, token)

  # Build the service object for the Gmail API
  service = build('gmail', 'v1', credentials=creds)
  return service

def fetch_emails(service, max_results):
  """Fetch unread emails from the user's inbox."""
  # Fetch the metadata of unread messages
  results = service.users().messages().list(userId='me', labelIds=['INBOX'], q='is:unread', maxResults=max_results).execute()
  messages = results.get('messages', [])


  # Fetch the full content of each email using the message ID
  full_messages = []
  for message in messages:
      msg = service.users().messages().get(userId="me", id=message["id"]).execute()
      full_messages.append(msg)


  return full_messages


def display_prioritized_messages(prioritized_messages, high_priority_senders, urgency_keywords, model_choice):
   for idx, (msg, prediction) in enumerate(prioritized_messages, start=1):
       # print(f"Message {idx} structure: {msg}")  # Inspecting the structure of the message
          
       # Access the sender email from the headers
       sender_email = get_sender_email(msg)
      
       # Access and decode the body (if necessary)
       email_body = get_message_body(msg)
      
       if model_choice == "classification":
           priority_label = "High Priority" if prediction == 1 else "Low Priority"
       else:
           priority_label = prediction


       print(f"\n--- Message {idx} ---")
       print(f"From: {sender_email}")
       print(f"ML Priority: {priority_label}")
       print(f"Body:\n{email_body[:500]}")  # Display first 500 characters of the body
       print("-" * 50)


def train_random_forest_classifier(messages, high_priority_senders, urgency_keywords):
   """Train a Random Forest Classifier."""
   # Collect training data
   features_df = collect_training_data(messages, high_priority_senders, urgency_keywords)


   labels = []
   for message in messages:
       priority_score = calculate_priority_score(message, high_priority_senders, urgency_keywords)
       labels.append(1 if priority_score > 5 else 0)  # 1 -> High priority, 0 -> Low priority


   # Split the data into training and test sets
   X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)


   # Create and train the Random Forest model
   clf = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_depth=5)
   clf.fit(X_train, y_train)

   # Save the trained model to a file for later use
   with open("email_priority_classification.pkl", "wb") as f:
       pickle.dump(clf, f)


   # Evaluate the model on the test set
   accuracy = clf.score(X_test, y_test)
   print(f"Model accuracy: {accuracy * 100:.2f}%")
   return clf


def predict_priority_classification(message, model, high_priority_senders, urgency_keywords):
   # Extract features from the email
   features_dict = extract_email_features(message, high_priority_senders, urgency_keywords)
   # Create a DataFrame with the correct column names (same as during training)
   # Use the model to predict the priority
   features = pd.DataFrame([
       [
           features_dict['is_high_priority_sender'],
           features_dict['contains_urgency_keywords'],
           features_dict['sentiment'],
           features_dict['email_length'],
           features_dict['time_since_received']
       ]
   ], columns=['is_high_priority_sender', 'contains_urgency_keywords', 'sentiment', 'email_length', 'time_since_received'])
  
   prediction = model.predict(features)[0]
   return prediction  # 1 -> High priority, 0 -> Low priority


def evaluate_classification(model, test_messages, high_priority_senders, urgency_keywords):
   true_labels = []  # This will contain the actual labels (high priority or low priority)
   predicted_labels = []  # This will contain the predicted labels (high priority or low priority)


   for message in test_messages:
       # Determine the true label based on the priority score
       priority_label = predict_priority_classification(message, model, high_priority_senders, urgency_keywords)
       predicted_labels.append(1 if priority_label == 1 else 0)  # Assign label 1 for high priority, 0 for low priority


       # Predict the priority using the model's predicted score
       true_labels.append(1 if calculate_priority_score(message, high_priority_senders, urgency_keywords) >= 5 else 0)  # Model's prediction based on the same threshold


   print(true_labels)
   # Calculate accuracy
   accuracy = accuracy_score(true_labels, predicted_labels)
   print(f"Model Accuracy: {accuracy * 100:.2f}%")


def train_regressor_model(messages, high_priority_senders, urgency_keywords):
   features_df = collect_training_data(messages, high_priority_senders, urgency_keywords)
  
   labels = []
   for message in messages:
       priority_score = calculate_priority_score(message, high_priority_senders, urgency_keywords)
       labels.append(priority_score)
  
   x_train, x_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=43)
   regressor = RandomForestRegressor(n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_depth=10)
   regressor.fit(x_train, y_train)
  
   with open("email_regressor_model.pkl", "wb") as f:
       pickle.dump(regressor, f)


   accuracy = regressor.score(x_test, y_test)
   print(f"Model accuracy: {accuracy * 100:.2f}%")
   return regressor


def predict_priority_regressor(message, model, high_priority_senders, urgency_keywords):
   features_dict = extract_email_features(message, high_priority_senders, urgency_keywords)
   features = pd.DataFrame([[
      features_dict['is_high_priority_sender'],
      features_dict['contains_urgency_keywords'],
      features_dict['sentiment'],
      features_dict['email_length'],
      features_dict['time_since_received']
  ]], columns=['is_high_priority_sender', 'contains_urgency_keywords', 'sentiment', 'email_length', 'time_since_received'])
   prediction = model.predict(features)[0]
   return prediction


def evaluate_regressor(model, test_messages, high_priority_senders, urgency_keywords):
   predicted_labels = []
   true_labels = []


   for message in test_messages:
       true_labels.append(calculate_priority_score(message, high_priority_senders, urgency_keywords))
       predicted_label = predict_priority_regressor(message, model, high_priority_senders, urgency_keywords)
       predicted_labels.append(predicted_label)


   mse = mean_squared_error(true_labels, predicted_labels)
   mae = mean_absolute_error(true_labels, predicted_labels)
   r2 = r2_score(true_labels, predicted_labels)


   print(f"Mean Squared Error (MSE): {mse:.4f}")
   print(f"Mean Absolute Error (MAE): {mae:.4f} ")
   print(f"R^2 Score: {r2:.4f}")


def process_messages(service, messages, high_priority_senders, urgency_keywords):
    """Process and analyze messages."""
    for message in messages:
        msg = service.users().messages().get(userId="me", id=message["id"]).execute()

        # Analyze content
        content_analysis = analyze_message_content(msg)

        # Print the extracted keywords
        print(f"Message ID: {message['id']}")
        print("TF-IDF Keywords:", content_analysis["keywords_tfidf"])
        print("RAKE Keywords:", content_analysis["keywords_rake"])
        print("Named Entities:", content_analysis["named_entities"])
        print("\n")

def analyze_all_messages(service, messages):
    """Perform topic modeling on all messages."""
    topics = topic_modeling(messages, num_topics=5, num_words=10, model_type='lda')
    print("Discovered Topics:")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx + 1}: {topic}")


def main():
   # Authenticate and fetch unread messages
  service = authenticate_email()

    # Retrieve messages using the new function
  messages = fetch_emails(service, 150)  # Fetch messages with this new function

  if not messages:
        print("No unread messages found.")
        return

    # Analyze all the fetched messages (this could include processing for keywords, etc.)
  analyze_all_messages(service, messages)  # Analyzing all messages

   # Define high-priority senders and urgency keywords
  high_priority_senders = set([
       "learn@itr.mail.codecademy.com", "collegedata@email.collegedata.com", "emails@emails.cinemark.com", "updates@academia-mail.com", "news@smfc.edx.org", "premium@academia-mail.com", "toysrus@emails.toysrus.ca", "no-reply@yesglasses.com", "noreply@email.apple.com", "nytimes@e.newyorktimes.com", "events@seatgeek.com", "emails@emails.rakuten.com"
   ])
  urgency_keywords = [
       "urgent", "important", "asap", "immediately", "priority", "time-sensitive", "attention required",
       "deadline", "emergency", "critical", "last chance", "final notice", "respond now",
       "action needed", "act now", "response required", "limited time", "do not miss", "offer expires",
       "final opportunity", "immediate attention", "expiring soon", "follow up", "reminder", "second notice",
       "payment due", "account update", "past due", "overdue", "required action", "mandatory", "confirm now",
       "important update", "last warning", "security alert", "verification needed"
   ]


   # Split the messages into training and testing data
  train_messages, test_messages = train_test_split(messages, test_size=0.2, random_state=42)


  model_choice = input("Choose the model to use (classification/regression): ").strip().lower()


  if model_choice == "classification":
       model_file = "email_priority_classification.pkl"
  elif model_choice == "regression":
       model_file = "email_regressor_model.pkl"
  else:
       print("Invalid model choice. Exiting.")
       return


   # Check if the model already exists
  if os.path.exists(model_file):
       print("Loading the pre-trained model...")
       with open(model_file, "rb") as f:
           model = pickle.load(f)
  else:
       print("Model not found. Training a new model...")
       # Train the model with the training messages
       if model_choice == "classification":
           model = train_random_forest_classifier(train_messages, high_priority_senders, urgency_keywords)
       else:
           model = train_regressor_model(train_messages, high_priority_senders, urgency_keywords)
      
       # Save the trained model to disk
       with open(model_file, "wb") as f:
           pickle.dump(model, f)


   # If test messages exist, predict and prioritize them
  if test_messages:
       prioritized_messages = []
       for message in test_messages:
           if model_choice == "classification":
               prediction = predict_priority_classification(message, model, high_priority_senders, urgency_keywords)
           else:
               prediction = predict_priority_regressor(message, model, high_priority_senders, urgency_keywords)
          
           prioritized_messages.append((message, prediction))


       # Sort the messages based on predicted priority (higher priority first)
       prioritized_messages.sort(key=lambda x: x[1], reverse=True)


       # Display prioritized messages
       display_prioritized_messages(prioritized_messages, high_priority_senders, urgency_keywords, model_choice)


       # Evaluate the model accuracy
       if model_choice == "classification":
           evaluate_classification(model, test_messages, high_priority_senders, urgency_keywords)
       else:
           evaluate_regressor(model, test_messages, high_priority_senders, urgency_keywords)
  else:
       print("No unread messages found.")

if __name__ == '__main__':
  main()