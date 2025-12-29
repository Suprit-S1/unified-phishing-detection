import re


def clean_text(text):
    """
    Standardizes text but keeps 'phishing signals' like !, $, %.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Replace URLs with 'url_token'
    text = re.sub(r'http\S+|www\S+|https\S+', 'url_token', text)

    # 3. Replace Emails with 'email_token'
    text = re.sub(r'\S+@\S+', 'email_token', text)

    # 4. Remove generic special chars, BUT KEEP: ! ? $ % &
    # These are high-value signals for spam/phishing
    text = re.sub(r'[^a-zA-Z0-9\s\!\?\$\%\&]', '', text)

    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text