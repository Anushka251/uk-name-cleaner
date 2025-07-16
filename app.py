import streamlit as st
import pandas as pd
import re, unidecode, spacy
from fuzzywuzzy import fuzz
from metaphone import doublemetaphone
from collections import Counter
import base64
import os

# -------------------- Setup SpaCy --------------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["puthon", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
# -------------------- Mappings --------------------
title_mapping = {
    "mr": "Mr.", "mister": "Mr.", "ms": "Ms.", "miss": "Ms.",
    "mrs": "Mrs.", "dr": "Dr.", "doc": "Dr.", "doctor": "Dr.",
    "prof": "Prof.", "professor": "Prof.", "mx": "Mx.",
    "sir": "Sir", "madam": "Madam", "madame": "Madam"
}

accented_name_map = {
    "jose": "José", "rene": "René", "francois": "François", "soren": "Søren",
    "muller": "Müller", "chloe": "Chloé", "andre": "André", "zoe": "Zoë", "noel": "Noël"
}

prefix_whitelist = {
    "obrien", "oconnor", "omalley", "odonnell", "ohara", "mccarthy", "mcdonald", "mcgregor"
}

logic_rules = [
    "Remove accents", "Remove special characters and preserve commas",
    "Deduplication with Phonetics and Fuzzy Matching",
    "Normalising Title", "Accent Correction", "Normalising Address"
]

# -------------------- Cleaning Functions --------------------
def standardize_text(text):
    text = str(text).strip().lower()
    return re.sub(r'[^a-z0-9,\s]', '', unidecode.unidecode(text))

def normalize_token(token):
    token = token.strip()
    if re.fullmatch(r'([A-Za-z]{1,2}\d{1,2}[A-Za-z]?)\s?(\d[A-Za-z]{2})', token.upper()):
        return token.upper()
    if re.search(r'[A-Za-z]', token) and re.search(r'\d', token):
        return token.upper()
    if token.isdigit():
        return token
    return token.capitalize()

def normalize_address_field(field):
    if not field or not isinstance(field, str) or field.strip() == "":
        return None
    tokens = field.strip().split()
    return ' '.join(normalize_token(token) for token in tokens)

def normalize_full_uk_address(address_string):
    if not isinstance(address_string, str) or address_string.strip() == "":
        return 'Nan'
    parts = [p.strip() for p in address_string.split(',')]
    normalized_parts = [normalize_address_field(part) for part in parts if normalize_address_field(part)]
    return ', '.join(normalized_parts)


def parse_improved_address(text):
    result = {"City": 'None', "Postcode": 'None', "Address": 'None', "Phone": 'None', "Fax": 'None'}

    if pd.isna(text) or not isinstance(text, str):
        return result

    text = text.strip()

    # Find postcode
    postcode_match = re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", text, re.IGNORECASE)

    # Find phone/fax numbers
    numbers = re.findall(r"\b(?:\+44\s?\d{4}|\(?0\d{2,4}\)?)\s?\d{3,4}\s?\d{3,4}\b", text)

    result["Postcode"] = postcode_match.group().upper() if postcode_match else 'None'

    if numbers:
        result["Phone"] = numbers[0]
        if len(numbers) > 1:
            result["Fax"] = numbers[1]

    for number in numbers:
        text = text.replace(number, '')

    # Split around postcode
    if result["Postcode"] and result["Postcode"] in text:
        parts = text.split(result["Postcode"])
        before = parts[0].strip(", ")
        after = parts[1].strip(", ") if len(parts) > 1 else ""
        tokens_before = [t.strip() for t in before.split(",") if t.strip()]
        tokens_after = [t.strip() for t in after.split(",") if t.strip()]

        if tokens_before:
            city_candidate = tokens_before[-1]
            if not re.search(r'\d', city_candidate):
                result["City"] = city_candidate.title()
                address_tokens = tokens_before[:-1] + tokens_after
            else:
                address_tokens = tokens_before + tokens_after
        else:
            address_tokens = tokens_after

        if address_tokens:
            result["Address"] = ", ".join(address_tokens)

    # Final check: if comma in 'Address', split first token into City
    if result["Address"] != 'None' and ',' in result["Address"]:
        parts = result["Address"].split(',', 1)
        if result["City"] == 'None' and parts[0].strip():
            result["City"] = parts[0].strip().title()
        result["Address"] = parts[1].strip()

    return result

def fix_surname_prefix(word):
    w = word.lower()
    if w in prefix_whitelist:
        if w.startswith("mc"): return "Mc" + w[2:].capitalize()
        if w.startswith("o") and "'" not in w: return "O’" + w[1:].capitalize()
    return word.capitalize()

def normalize_title(full_name):
    if not isinstance(full_name, str): return "", full_name
    parts = full_name.strip().split()
    if not parts: return "", full_name
    norm_title = title_mapping.get(parts[0].lower().replace(".", ""), "")
    return norm_title, " ".join(parts[1:]) if norm_title else full_name

def fix_name(name):
    if pd.isna(name): return name
    parts = name.strip().split()
    return " ".join([accented_name_map.get(w.lower(), fix_surname_prefix(w)) for w in parts])

import pandas as pd

# --- Whitelist of UK surnames ---
uk_surnames = {
    'Abbott', 'Adams', 'Ahmed', 'Akhtar', 'Alexander', 'Ali', 'Allan', 'Allen', 'Anderson', 'Andrews', 'Archer',
    'Arnold', 'Ashton', 'Atkinson', 'Bailey', 'Baker', 'Ball', 'Barber', 'Barnes', 'Barnett', 'Barrett', 'Barry',
    'Bartlett', 'Barton', 'Bates', 'Baxter', 'Begum', 'Bell', 'Bennett', 'Bentley', 'Berry', 'Bevan', 'Bibi',
    'Bird', 'Black', 'Blackburn', 'Blake', 'Bolton', 'Booth', 'Boyle', 'Brady', 'Brennan', 'Brooks', 'Brown',
    'Bruce', 'Buckley', 'Bull', 'Burgess', 'Burns', 'Burrows', 'Burton', 'Butler', 'Byrne', 'Cameron', 'Campbell',
    'Carr', 'Carroll', 'Carter', 'Cartwright', 'Chadwick', 'Chambers', 'Chandler', 'Chapman', 'Charlton', 'Clark',
    'Clarke', 'Clayton', 'Coates', 'Cole', 'Coleman', 'Collins', 'Connor', 'Cook', 'Cooke', 'Cooper', 'Cox',
    'Crawford', 'Cross', 'Cunningham', 'Curtis', 'Dale', 'Davidson', 'Davies', 'Davis', 'Dawson', 'Day', 'Dean',
    'Dennis', 'Dixon', 'Dobson', 'Doherty', 'Douglas', 'Duncan', 'Dyer', 'Edwards', 'Elliott', 'Ellis', 'Evans',
    'Faulkner', 'Ferguson', 'Field', 'Finch', 'Fisher', 'Fitzgerald', 'Fleming', 'Fletcher', 'Flynn', 'Foster',
    'Fowler', 'Fox', 'Francis', 'Fraser', 'French', 'Frost', 'Fuller', 'Gardiner', 'Gardner', 'Garner', 'Gibson',
    'Gilbert', 'Goddard', 'Godfrey', 'Goodwin', 'Gordon', 'Gough', 'Gould', 'Graham', 'Grant', 'Gray', 'Green',
    'Gregory', 'Griffin', 'Griffiths', 'Hall', 'Hammond', 'Hanson', 'Harding', 'Harper', 'Harris', 'Harrison',
    'Hart', 'Hartley', 'Harvey', 'Hawkins', 'Hayes', 'Hayward', 'Henderson', 'Herbert', 'Hicks', 'Hill', 'Hobbs',
    'Hodgson', 'Holden', 'Holland', 'Holmes', 'Holt', 'Hooper', 'Hope', 'Hopkins', 'Horton', 'Howard', 'Howells',
    'Hudson', 'Hughes', 'Humphries', 'Hunt', 'Hunter', 'Hussain', 'Hutchinson', 'Hyde', 'Jackson', 'James',
    'Jarvis', 'Jennings', 'Johnson', 'Jones', 'Jordan', 'Kaur', 'Kay', 'Kelly', 'Kemp', 'Kent', 'Khan', 'King',
    'Kirby', 'Kirk', 'Knight', 'Knowles', 'Lamb', 'Lambert', 'Lane', 'Law', 'Lawrence', 'Leach', 'Lee', 'Lewis',
    'Little', 'Lloyd', 'Long', 'Lord', 'Lowe', 'Lucas', 'Macdonald', 'Mann', 'Manning', 'Marsden', 'Marsh',
    'Marshall', 'Martin', 'Mason', 'Matthews', 'May', 'Mccarthy', 'Metcalfe', 'Miah', 'Middleton', 'Miles',
    'Miller', 'Mills', 'Mistry', 'Mitchell', 'Moore', 'Moran', 'Morgan', 'Morley', 'Morris', 'Moss', 'Murphy',
    'Murray', 'Myers', 'Nash', 'NelsonSmith', 'Newman', 'Nicholls', 'Nicholson', 'Norman', "O'Brien", "O'Connor",
    "O'Sullivan", 'Oliver', 'Osborne', 'Owen', 'Owens', 'Palmer', 'Parker', 'Parkin', 'Parkinson', 'Parry',
    'Parsons', 'Patel', 'Patterson', 'Payne', 'Pearce', 'Pearson', 'Perkins', 'Perry', 'Peters', 'Phillips',
    'Pickering', 'Pollard', 'Poole', 'Pope', 'Potter', 'Potts', 'Powell', 'Pratt', 'Preston', 'Price', 'Pritchard',
    'Randall', 'Read', 'Reed', 'Reeves', 'Reid', 'Reynolds', 'Rhodes', 'Richards', 'Richardson', 'Riley',
    'Roberts', 'Robertson', 'Robinson', 'Rogers', 'Rowe', 'Russell', 'Ryan', 'Saunders', 'Schofield', 'Scott',
    'Shah', 'Sharpe', 'Shaw', 'Shepherd', 'Sheppard', 'Short', 'Simpson', 'Singh', 'Slater', 'Smart', 'Smith',
    'Spencer', 'Stephens', 'Stevens', 'Stevenson', 'Stewart', 'Stokes', 'Storey', 'Summers', 'Sutton', 'Swift',
    'Taylor', 'Thomas', 'Thompson', 'Thomson', 'Townsend', 'Tucker', 'Turner', 'Tyler', 'Vaughan', 'Wade',
    'Walker', 'Wall', 'Wallace', 'Walsh', 'Walters', 'Walton', 'Ward', 'Warner', 'Warren', 'Waters', 'Watkins',
    'Watson', 'Watts', 'Webb', 'Welch', 'Wells', 'West', 'Weston', 'White', 'Whitehouse', 'Whittaker', 'Wilkins',
    'Wilkinson', 'Williams', 'Williamson', 'Willis', 'Wilson', 'Wong', 'Wood', 'Woods', 'Woodward', 'Wright',
    'Wyatt', 'Yates', 'Young'
}

# --- Title patterns ---
titles = ['Mr.', 'Ms.', 'Mrs.', 'Dr.', 'Prof.', 'Mx.', 'Sir', 'Madam']

# --- Main name parsing function ---
def parse_name_parts(df):
    def split(name):
        # Check for missing/invalid names
        if pd.isna(name) or str(name).strip().lower() in ['nan', 'none', 'null', '']:
            return pd.Series(['None', 'None', 'None'])

        parts = str(name).strip().split()
        title = parts[0] if parts[0] in titles else 'None'
        rest = parts[1:] if title != 'None' else parts

        if len(rest) == 0:
            return pd.Series([title, 'None', 'None'])
        elif len(rest) == 1:
            if rest[0].title() in uk_surnames:
                return pd.Series([title, 'None', rest[0]])
            else:
                return pd.Series([title, rest[0], 'None'])
        else:
            last = rest[-1]
            if last.title() in uk_surnames:
                return pd.Series([title, ' '.join(rest[:-1]), last])
            else:
                return pd.Series([title, rest[0], ' '.join(rest[1:])])

    parsed = df['name'].apply(split)
    parsed.columns = ['Title', 'First Name', 'Last Name']
    return df.join(parsed)

import re
from collections import Counter
import pandas as pd

# --- Helpers ---
def check_invalid_chars(text):
    return bool(re.search(r"[^\w\s,.\-\/()]", str(text))) if pd.notnull(text) else False

def check_capitalisation(text):
    return text != text.title() if isinstance(text, str) else False

def check_missing_parts(address):
    parts = [part.strip() for part in str(address).split(',')] if pd.notnull(address) else []
    expected_parts = 5
    return expected_parts - len(parts)

# --- Name and Address Profiling ---
def profile_name(name):
    issues = []
    if pd.isnull(name) or name.strip().lower() in ['nan', 'null', 'none', '']:
        issues.append("Name Missing")
    if check_invalid_chars(name):
        issues.append("Invalid Characters Present (Name)")
    if check_capitalisation(name):
        issues.append("Capitalisation Error (Name)")
    return ", ".join(issues) if issues else "No Error"

def profile_address(address):
    issues = []
    if pd.isnull(address) or address.strip().lower() in ['nan', 'null', 'none', '']:
        issues.append("Address Missing")
    if check_invalid_chars(address):
        issues.append("Invalid Characters Present (Address)")
    if check_capitalisation(address):
        issues.append("Capitalisation Error (Address)")
    missing = check_missing_parts(address)
    if missing > 0:
        issues.append(f"Missing {missing} Address Parts")
    return ", ".join(issues) if issues else "No Error"

# --- Step 1: Add Issue Columns ---
def profile_all(df, name_col='name', address_col='address'):
    df['issues_name'] = df[name_col].apply(profile_name)
    df['issues_address'] = df[address_col].apply(profile_address)
    df['all_issues'] = df['issues_name'] + ", " + df['issues_address']
    return df

# --- Step 2: Summarize Issue Counts ---
def summarize_issues(df):
    total_rows = len(df)
    exploded = df['all_issues'].str.split(', ').explode()
    counts = Counter(exploded)

    summary_df = pd.DataFrame({
        "Issue Type": list(counts.keys()),
        "Count": list(counts.values())
    })
    summary_df["% of Total Rows"] = (summary_df["Count"] / total_rows * 100).round(2)
    return summary_df

# -------------------- Gradio Backend --------------------
def run_pipeline(file):
    # Step 1: Load and keep a copy of raw data
    df = pd.read_csv(file.name)
    df_raw = df.copy()
    df_raw = profile_all(df_raw)

    # Step 2: Run issue summary on raw data
    summary_df = summarize_issues(df_raw)

    # ✅ Step 3: Start with raw data
    df_orig = df.copy()
    df_orig['id'] = df_orig.index  # ✅ Add ID to track original rows

    # ✅ Step 4: Use a copy for cleaning
    df = df_orig.copy()  # ✅ Start working on df so original stays safe


    # Step 3: Continue with cleaning and processing
    df['name'] = df['name'].apply(standardize_text)
    df['address'] = df['address'].apply(standardize_text)
    df = df.drop_duplicates()

    df[['title', 'name']] = df['name'].apply(lambda x: pd.Series(normalize_title(x)))
    df['name'] = df['title'] + ' ' + df['name']
    df.drop(columns=['title'], inplace=True)

    df['name'] = df['name'].apply(fix_name)
    df['address'] = df['address'].apply(normalize_full_uk_address)

    # ✅ Join back to original if needed
    df_merged = df_orig.merge(df, on='id', how='left', suffixes=('_orig', '_cleaned'))
    df_merged = df_merged[['id', 'name_orig', 'address_orig', 'name_cleaned', 'address_cleaned']]


    logic_df = pd.DataFrame({"Logic Rules": logic_rules})

    # Step 4: Always parse name and address
    df_parsed = df.copy()

    # Run name parsing
    df_parsed = parse_name_parts(df_parsed)

    # Run address parsing
    address_parsed = df_parsed['address'].apply(parse_improved_address)
    df_parsed = pd.concat([df_parsed, pd.DataFrame(address_parsed.tolist())], axis=1)

    # ✅ Add ID and reorder parsed name columns
    df_parsed['id'] = df['id'].copy()
    df_parsed = df_parsed[['id','name','Title','First Name','Last Name','address','City','Postcode','Address','Phone','Fax']]

    # 1. Merge df_merged with df_parsed on 'id' to get Postcode
    df_final = df_merged.merge(df_parsed[['id', 'Postcode']], on='id', how='left')

    # 2. Update status
    def update_status(row):
        if str(row['Postcode']).strip().lower() == 'none' or pd.isna(row['Postcode']):
            return 'Cannot be corrected'
        elif pd.notna(row['name_cleaned']):
            return 'Corrected'
        else:
            return 'Dropped'

    df_final['status'] = df_final.apply(update_status, axis=1)

    # 3. Finalize output
    df_final = df_final[['id', 'name_orig', 'address_orig', 'name_cleaned', 'address_cleaned', 'status']]

    # Step 5: Save outputs
    summary_df.to_csv("/tmp/summary.csv", index=False)
    logic_df.to_csv("/tmp/logic.csv", index=False)
    df_final.to_csv("/tmp/final_output.csv", index=False)  # ✅ Save df_final
    df_parsed.to_csv("/tmp/cleaned.csv", index=False)


    # ✅ Return all key dataframes and file paths
    return summary_df, logic_df, df_final, df_parsed, "/tmp/summary.csv", "/tmp/logic.csv", "/tmp/final_output.csv", "/tmp/cleaned.csv"


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="UK Name & Address Cleaner", layout="wide")

st.title("🧹 UK Name & Address Cleaner")
st.markdown("Upload a CSV with `name` and `address` columns to begin.")

uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Running pipeline..."):
        summary_df, logic_df, df_final, df_parsed, f1, f2, f3, f4 = run_pipeline(uploaded_file)

    st.subheader("🧾 Issue Summary")
    st.dataframe(summary_df)

    def get_download_link(df, filename, label):
        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇ {label}</a>'
        return href

    st.markdown(get_download_link(summary_df, "issue_summary.csv", "Download Issue Summary"), unsafe_allow_html=True)

    # Proceed toggle
    proceed = st.checkbox("✅ Do you want to continue?", value=False)

    if proceed:
        st.subheader("📐 Logic Table")
        st.dataframe(logic_df)
        st.markdown(get_download_link(logic_df, "logic_rules.csv", "Download Logic Rules"), unsafe_allow_html=True)

        st.subheader("✅ Final Output")
        st.dataframe(df_final)
        st.markdown(get_download_link(df_final, "final_output.csv", "Download Final Output"), unsafe_allow_html=True)

        st.subheader("🧼 Parsed Name + Address Data")
        st.dataframe(df_parsed)
        st.markdown(get_download_link(df_parsed, "parsed_data.csv", "Download Parsed Data"), unsafe_allow_html=True)
