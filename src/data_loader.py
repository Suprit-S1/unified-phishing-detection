import pandas as pd
import os
import sys
import io

# Check for Google Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def clean_labels(df, label_col):
    """
    Standardizes labels to 0 (Safe) and 1 (Phishing).
    """
    s = df[label_col].astype(str).str.lower().str.strip()

    text_map = {
        'ham': 0, 'legitimate': 0, 'safe': 0, 'good': 0, '0': 0, '0.0': 0,
        'spam': 1, 'phishing': 1, 'malicious': 1, 'bad': 1, '1': 1, '1.0': 1,
        'safe email': 0, 'phishing email': 1
    }

    num_map = {'-1': 1, '-1.0': 1, '1': 0, '1.0': 0}

    final_labels = s.map(text_map)
    if final_labels.isnull().any():
        final_labels = final_labels.fillna(s.map(num_map))

    return final_labels  # Returns NaNs for unknown labels


def is_text_data(series):
    sample = series.dropna().head(5).astype(str).tolist()
    if not sample: return False
    looks_numeric = all(s.replace('.', '').replace('-', '').isdigit() for s in sample)
    return not looks_numeric


def upload_and_validate(dataset_name, validation_func):
    """
    - If Canceled/Empty: Returns None (Skips this step).
    - If Invalid File: Loops and asks for retry.
    - If Valid: Returns DataFrame.
    """
    while True:
        print(f"\n REQUEST: Please upload your {dataset_name.upper()} dataset.")

        if IN_COLAB:
            uploaded = files.upload()
            if not uploaded:
                print(f"   ️ No file uploaded. Skipping {dataset_name}...")
                return None

            filename = list(uploaded.keys())[0]
            file_content = io.BytesIO(uploaded[filename])
        else:
            path = input(f"   (Local) Enter path for {dataset_name} ...: ").strip().replace('"', '')
            if not path:
                print(f"    Skipping {dataset_name}...")
                return None
            filename = path
            file_content = path

        try:
            is_valid, result = validation_func(file_content, filename)

            if is_valid:
                print(f"    Accepted! {dataset_name} loaded successfully.")
                return result
            else:
                print(f"   ️ REJECTED: {result}")
                print(f"    Please upload the CORRECT {dataset_name} file.")

        except Exception as e:
            print(f"   ️ ERROR: {e}")
            print(f"   Please try again.")


# --- VALIDATION LOGIC (STRICT EXACT MATCHING) ---

def check_url_file(file_content, filename):
    try:
        print("    Reading URL file...")
        # Read full file as string to avoid type errors
        df = pd.read_csv(file_content, on_bad_lines='skip', dtype=str)
        df.columns = df.columns.str.lower().str.strip()

        # EXACT MATCH KEYS
        url_keys = ['url', 'link', 'website', 'domain', 'uri', 'phishing_url']
        label_keys = ['label', 'type', 'class', 'result', 'target', 'phishing', 'label_int']

        # Find column ONLY if it matches the string EXACTLY in the list
        url_col = next((c for c in df.columns if c in url_keys), None)
        label_col = next((c for c in df.columns if c in label_keys), None)

        if not url_col or not label_col:
            return False, f"Missing columns. Found: {list(df.columns)}. Needed exact match for: {url_keys}"

        if not is_text_data(df[url_col]):
            return False, f"Column '{url_col}' contains numbers, expected text links."

        df = df.rename(columns={url_col: 'text'})

        # === CRITICAL FIX FOR URLs ===
        # We strip 'http', 'www', and punctuation so the tokenizer sees words (e.g. "paypal login")
        # instead of just "url_token".
        df['text'] = df['text'].astype(str).str.replace(r'^https?://', '', regex=True)
        df['text'] = df['text'].str.replace(r'^www\.', '', regex=True)
        df['text'] = df['text'].str.replace(r'[/\-\.]', ' ', regex=True)

        # Clean Labels
        df['label'] = clean_labels(df, label_col)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        # Shuffle and Limit to 100k
        if len(df) > 100000:
            df = df.sample(n=100000, random_state=42)

        df['source'] = 'url'
        return True, df[['text', 'label', 'source']]
    except Exception as e:
        return False, str(e)


def check_sms_file(file_content, filename):
    try:
        df = pd.read_csv(file_content, encoding='latin-1', on_bad_lines='skip', dtype=str)
        df.columns = df.columns.str.lower().str.strip()

        # EXACT MATCH KEYS
        text_keys = ['message', 'text', 'sms', 'v2', 'content', 'msg', 'body', 'data', 'sms_text']
        label_keys = ['label', 'type', 'v1', 'class', 'category', 'target', 'ham_spam']

        text_col = next((c for c in df.columns if c in text_keys), None)
        label_col = next((c for c in df.columns if c in label_keys), None)

        # Fallback for Headerless files (Standard UCI Dataset)
        if not text_col and not label_col and len(df.columns) >= 2:
            first_val = str(df.iloc[0, 0]).lower()
            if 'ham' in first_val or 'spam' in first_val:
                df.columns.values[0] = 'label_temp'
                df.columns.values[1] = 'text_temp'
                label_col = 'label_temp'
                text_col = 'text_temp'

        if not text_col or not label_col:
            return False, f"Missing columns. Found: {list(df.columns)}. Needed exact match for: {text_keys}"

        df = df.rename(columns={text_col: 'text'})

        df['label'] = clean_labels(df, label_col)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        df['source'] = 'sms'
        return True, df[['text', 'label', 'source']]
    except Exception as e:
        return False, str(e)


def check_mail_file(file_content, filename):
    try:
        df = pd.read_csv(file_content, on_bad_lines='skip', dtype=str)
        df.columns = df.columns.str.lower().str.strip()

        # EXACT MATCH KEYS
        sub_keys = ['subject', 'sub', 'title', 'text_formatsubject', 'email_subject']
        body_keys = ['body', 'content', 'message', 'text', 'email', 'data', 'text_formatbody', 'email_text']
        label_keys = ['label', 'type', 'class', 'category', 'target', 'spam', 'check', 'checklabelsort']

        sub_col = next((c for c in df.columns if c in sub_keys), None)
        body_col = next((c for c in df.columns if c in body_keys), None)
        label_col = next((c for c in df.columns if c in label_keys), None)

        if not label_col:
            return False, f"Missing LABEL column. Found: {list(df.columns)}"

        if sub_col and body_col:
            df['text'] = df[sub_col].fillna('').astype(str) + " " + df[body_col].fillna('').astype(str)
        elif body_col:
            df['text'] = df[body_col].fillna('').astype(str)
        elif sub_col:
            df['text'] = df[sub_col].fillna('').astype(str)
        else:
            return False, f"Missing Subject/Body. Found: {list(df.columns)}"

        df['label'] = clean_labels(df, label_col)

        # Remove unknown labels
        initial_len = len(df)
        df = df.dropna(subset=['label'])
        final_len = len(df)

        if initial_len != final_len:
            print(f"    Ignored {initial_len - final_len} rows with invalid labels.")

        df['label'] = df['label'].astype(int)
        df['source'] = 'email'

        return True, df[['text', 'label', 'source']]
    except Exception as e:
        return False, str(e)


def load_and_unify_data():
    print("\n==========================================")
    print("      MULTI-CHANNEL DATA LOADER          ")
    print("==========================================")

    # 1. Load Data
    df_url = upload_and_validate("URL (Phishing URLs)", check_url_file)
    df_sms = upload_and_validate("SMS (Spam Collection)", check_sms_file)
    df_mail = upload_and_validate("EMAIL (Phishing Emails)", check_mail_file)

    # 2. Filter Valid Datasets
    valid_dfs = [df for df in [df_url, df_sms, df_mail] if df is not None]

    if not valid_dfs:
        raise FileNotFoundError(" No data uploaded! You skipped all files. Cannot train.")

    # 3. Merge
    print("\n Merging uploaded datasets...")
    unified_df = pd.concat(valid_dfs, ignore_index=True)
    unified_df.dropna(subset=['text'], inplace=True)

    # 4. Final Balance Check
    counts = unified_df['label'].value_counts()
    print(f"    Final Distribution: Safe(0): {counts.get(0, 0)}, Phishing(1): {counts.get(1, 0)}")

    if len(counts) < 2:
        print("\n CRITICAL WARNING: Your combined dataset has ONLY ONE CLASS.")
        print("   Please upload at least one file containing the missing class.")

    print(f"\n COMPLETE: Ready to train on {len(unified_df)} total samples.")
    return unified_df