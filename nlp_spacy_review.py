import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

reviews = [
    "I love my new Apple iPhone! The camera quality is amazing.",
    "The Samsung TV was disappointing, poor picture quality."
]

for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    
    # Simple rule-based sentiment
    if any(word in review.lower() for word in ["love", "great", "amazing", "excellent"]):
        sentiment = "Positive"
    elif any(word in review.lower() for word in ["bad", "poor", "disappointing", "terrible"]):
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print("Sentiment:", sentiment)
