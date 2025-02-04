# InboxGuru 📬🧠  

**InboxGuru** is an AI-powered email prioritization and analysis tool that helps you stay on top of your inbox effortlessly. By analyzing email content, sender importance, and urgency indicators, InboxGuru ensures that you focus on what matters most while keeping your inbox organized.

**Created and Coded by Michael Pancras

---

## Features  

- **AI-Driven Email Prioritization**  
   - Classifies emails into high or low priority based on sender importance, urgency keywords, and content analysis.  
   - Option to score email priority using regression models for more nuanced rankings.  

- **Content Analysis**  
   - Extracts keywords, sentiment, and urgency cues from emails.  
   - Performs topic modeling to uncover recurring themes across your inbox.  

- **Gmail API Integration**  
   - Securely fetches unread emails and prioritizes them.  
   - Compatible with Google OAuth2 for seamless authentication.  

- **Customizable Configuration**  
   - Define your high-priority senders and urgency keywords for a personalized experience.  
   - Adjustable training and test datasets for model optimization.  

- **Machine Learning Support**  
   - Trains and saves Random Forest models for classification or regression of email priorities.  
   - Supports evaluation of model performance using metrics like accuracy, R² score, and error rates.  

---

## Example Output  

```plaintext
--- Message 1 ---
From: john.doe@example.com
ML Priority: High Priority
Body: This is an important update regarding your account...

--- Message 2 ---
From: support@shopping.com
ML Priority: Low Priority
Body: Your order has been shipped. Thank you for shopping...

Model Accuracy: 92.45%
