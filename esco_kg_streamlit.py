def render_future_profile_section(profile, ist_details, udemy_courses_df, skill_lookup):
    """Displays rankings and investment recommendations for a future target profile"""
    profile_name = profile.get('name', 'SOLL-Profil')
    required_skills = profile.get('skills', [])
    normalized_map = {
        normalize_skill_name(skill): skill
        for skill in required_skills if skill
    }
    
    if not normalized_map:
        st.info(f"No skills were found for {profile_name}.")
        return
    
    required_norm_set = set(normalized_map.keys())
    total_required = len(required_norm_set)
    
    ranking = []
    for detail in ist_details:
        norm_set = detail.get('normalized_skills_set', set())
        matched_norm = required_norm_set & norm_set
        missing_norm = required_norm_set - norm_set
        ranking.append({
            'ist_name': detail['name'],
            'match_count': len(matched_norm),
            'missing_count': len(missing_norm),
            'match_percent': round(len(matched_norm) / total_required * 100, 1) if total_required else 0,
            'matched_skills': sorted(normalized_map[n] for n in matched_norm),
            'missing_skills': sorted(normalized_map[n] for n in missing_norm)
        })
    
    ranking.sort(key=lambda x: (-x['match_count'], x['missing_count'], x['ist_name']))
    top_results = ranking[:5]
    
    st.markdown(f"### {profile_name} – {total_required} Skills")
    render_skill_list(required_skills)
    st.markdown("---")
    
    selected_course_request = None
    
    for idx, result in enumerate(top_results):
        title = f"Platz {idx + 1}: {result['ist_name']} ({result['match_count']}/{total_required} Treffer, {result['match_percent']}%)"
        with st.expander(title, expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Gemeinsame Fähigkeiten:**")
                if result['matched_skills']:
                    for skill in result['matched_skills']:
                        st.write(f"  ✓ {skill}")
                else:
                    st.info("Keine Übereinstimmungen gefunden.")
            with col_b:
                st.write("**Fehlende Fähigkeiten:**")
                if result['missing_skills']:
                    for skill in result['missing_skills']:
                        st.write(f"  ✗ {skill}")
                else:
                    st.success("Alle benötigten Fähigkeiten vorhanden!")
            
            if result['missing_skills']:
                button_key = f"courses_{profile_name}_{idx}_{result['ist_name']}"
                if st.button(f"▤ Kursempfehlungen für {result['ist_name']}", key=button_key):
                    selected_course_request = {
                        'ist_name': result['ist_name'],
                        'missing_skills': result['missing_skills']
                    }
    st.markdown("---")
    
    if selected_course_request:
        st.markdown(
            f"#### Kursempfehlungen für {selected_course_request['ist_name']} "
            f"(fehlende Skills: {len(selected_course_request['missing_skills'])})"
        )
        render_course_recommendations_for_profile(
            selected_course_request['missing_skills'],
            udemy_courses_df,
            skill_lookup
        )

import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import warnings
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict
import xml.etree.ElementTree as ET
import tempfile
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def data_path(*relative_parts):
    """Creates absolute paths to the data folder, regardless of the working directory."""
    return os.path.join(DATA_DIR, *relative_parts)

# Page config
st.set_page_config(
    page_title="Kompetenzabgleich & Weiterbildungsempfehlungen",
    page_icon="",
    layout="wide"
)

def save_employees_to_csv(employees_data, filename=None):
    """Saves employee data to a CSV file"""
    try:
        target_file = filename or data_path('employees_data.csv')
        employees_data.to_csv(target_file, index=False)
        return True
    except Exception as e:
        st.error(f"Fehler beim Speichern der Mitarbeiterdaten: {str(e)}")
        return False

def load_employees_from_csv(filename=None):
    """Imports employee data from a CSV file"""
    try:
        target_file = filename or data_path('employees_data.csv')
        if os.path.exists(target_file):
            df = pd.read_csv(target_file)
            # Make sure all the required columns are included
            required_columns = [
                'Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role',
                'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label',
                'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'
            ]
            # Add missing columns
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            # Make sure the columns are in the correct order
            df = df[required_columns]
            # Handle NaN values
            for col in ['Manual_Skills', 'ESCO_Role', 'KldB_5_digit', 'Name', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']:
                df[col] = df[col].fillna('')
            return df
        else:
            return pd.DataFrame(columns=[
                'Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role',
                'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label',
                'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'
            ])
    except Exception as e:
        st.error(f"Fehler beim Laden der Mitarbeiterdaten: {str(e)}")
        return pd.DataFrame(columns=[
            'Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role',
            'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label',
            'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'
        ])

def manual_csv_parser(file_path, skip_rows=0):
    """Manueller CSV-Parser für problematische Dateien"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip the first few lines
        lines = lines[skip_rows:]
        
        # Find the longest row to determine the number of columns
        max_columns = 0
        for line in lines:
            if line.strip():
                columns = line.count(',') + 1
                max_columns = max(max_columns, columns)
        
        # Create column names
        columns = [f'col_{i}' for i in range(max_columns)]
        
        # Parse the data
        data = []
        for line in lines:
            if line.strip():
                # Split the row and make sure it has the correct number of columns
                parts = line.strip().split(',')
                if len(parts) < max_columns:
                    parts.extend([''] * (max_columns - len(parts)))
                elif len(parts) > max_columns:
                    parts = parts[:max_columns]
                data.append(parts)
        
        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        st.error(f"Manueller Parser fehlgeschlagen: {str(e)}")
        return pd.DataFrame()

def normalize_esco_code(esco_code):
    """Standardizes ESCO codes for better comparison"""
    if pd.isna(esco_code):
        return ""
    
    esco_code = str(esco_code).strip()
    
    # If it is a complete URI, extract the code
    if esco_code.startswith('http://data.europa.eu/esco/occupation/'):
        # Extract the code from the URI
        parts = esco_code.split('/')
        if len(parts) > 0:
            return parts[-1]  # Take the last part
    
    # If it's a UUID, leave it as is
    if len(esco_code) == 36 and '-' in esco_code:
        return esco_code
    
    # If it's a short code (e.g., C0110), leave it as is
    return esco_code

def normalize_job_label(label):
    """Standardizes job titles for reliable comparisons"""
    if pd.isna(label):
        return ""
    
    text = str(label).strip()
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('–', '-')
    text = text.replace('/in', ' ')
    text = text.replace('(in)', ' ')
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = text.replace('-', ' ')
    text = text.replace(',', ' ')
    text = text.replace('.', ' ')
    
    replacements = {
        'ä': 'ae', 'Ä': 'ae',
        'ö': 'oe', 'Ö': 'oe',
        'ü': 'ue', 'Ü': 'ue',
        'ß': 'ss'
    }
    for src, target in replacements.items():
        text = text.replace(src, target)
    
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_display_label(label):
    """Standardized labels for accurate comparisons (e.g., en dash vs. hyphen)"""
    if pd.isna(label):
        return ""
    return str(label).replace('–', '-').strip()

def expand_job_aliases(label):
    """Adds individual synonyms to job titles"""
    if pd.isna(label):
        return []
    
    label = str(label).strip()
    if not label:
        return []
    
    candidates = [label]
    separators = ['\n', '|', ';']
    
    for sep in separators:
        next_candidates = []
        for item in candidates:
            parts = [p.strip() for p in re.split(rf'\s*{re.escape(sep)}\s*', item) if p.strip()]
            if len(parts) > 1:
                next_candidates.extend(parts)
            else:
                next_candidates.append(item)
        candidates = next_candidates
    
    def split_by_comma_outside_parentheses(text):
        parts = []
        current = []
        depth = 0
        for ch in text:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(depth - 1, 0)
            if ch == ',' and depth == 0:
                part = ''.join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(ch)
        last_part = ''.join(current).strip()
        if last_part:
            parts.append(last_part)
        return parts
    
    expanded = []
    for item in candidates:
        comma_parts = split_by_comma_outside_parentheses(item)
        if len(comma_parts) > 1:
            expanded.extend(comma_parts)
        else:
            expanded.append(item)
    
    unique = []
    seen = set()
    for item in expanded:
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

def normalize_skill_name(skill_name):
    """Standardizes skill names for robust comparisons"""
    if pd.isna(skill_name):
        return ""
    text = unicodedata.normalize('NFKC', str(skill_name)).lower().strip()
    text = text.replace('–', '').replace('-', '')
    text = text.replace('/', ' ')
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss'
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

_FORCED_HYPHEN_PATTERN = re.compile(r'([A-Za-zÄÖÜäöüß])\-([a-zäöüß])')

def clean_skill_label(label):
    """Removes forced hyphenation and unnecessary spaces from skill names."""
    if label is None:
        return ""
    text = str(label).replace('\u00ad', '').strip()
    if not text:
        return ""
    previous = None
    while previous != text:
        previous = text
        text = _FORCED_HYPHEN_PATTERN.sub(r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def is_valid_kldb_label(label):
    """Filters out incorrect KldB designations such as 'in - ...' or '-frau ...'."""
    if label is None:
        return False
    text = str(label).strip()
    if len(text) < 4:
        return False
    lower = text.lower()
    if lower.startswith('in -'):
        return False
    if not re.match(r'^[A-ZÄÖÜ]', text):
        return False
    return True

def is_informative_kldb_label(label):
    """Determines whether a KldB designation is sufficiently descriptive."""
    if not is_valid_kldb_label(label):
        return False
    text = str(label)
    if len(text) >= 12:
        return True
    if re.search(r'[\-()/]', text):
        return True
    if ' ' in text:
        return True
    return False

def select_preferred_alias(alias_list):
    """Select the most descriptive alias."""
    if not alias_list:
        return None
    informative = [alias for alias in alias_list if is_informative_kldb_label(alias)]
    candidates = informative if informative else alias_list
    candidates = sorted(candidates, key=lambda a: (-len(a), a))
    return candidates[0]

def render_skill_list(skills, columns=3):
    """Displays a simple list of skills in multiple columns"""
    if not skills:
        st.info("Keine Skills vorhanden.")
        return
    cols = st.columns(columns)
    for idx, skill in enumerate(skills):
        cols[idx % columns].write(f"• {skill}")

@st.cache_data
def build_skill_lookup(skills_df):
    """Creates a mapping of normalized skill names to their URIs"""
    lookup = {}
    if skills_df is None or skills_df.empty:
        return lookup
    for _, row in skills_df.iterrows():
        label = row.get('preferredLabel', '')
        if pd.isna(label):
            continue
        lookup[normalize_skill_name(label)] = row.get('conceptUri', '')
    return lookup

def prepare_missing_skill_entries(skill_names, skill_lookup):
    """Prepare missing skills for the course recommendation feature"""
    entries = []
    for skill in skill_names:
        norm = normalize_skill_name(skill)
        entries.append({
            'skill_label': skill,
            'skill_uri': skill_lookup.get(norm, ''),
            'is_essential': False
        })
    return entries

def render_course_recommendations_for_profile(skill_names, udemy_courses_df, skill_lookup):
    """Displays course recommendations for a set of missing skills"""
    if not skill_names:
        st.success("Alle benötigten Fähigkeiten sind bereits vorhanden – keine Kurse nötig.")
        return
    
    skill_entries = prepare_missing_skill_entries(skill_names, skill_lookup)
    with st.spinner("Suche passende Kurse für fehlende Fähigkeiten..."):
        recommendations = find_udemy_courses_for_skills(skill_entries, udemy_courses_df, top_k=3)
    
    if not recommendations:
        st.info("Keine passenden Kursempfehlungen gefunden.")
        return
    
    grouped = {}
    for rec in recommendations:
        skill = rec.get('skill') or rec.get('skill_label') or 'Unbekannter Skill'
        grouped.setdefault(skill, []).append(rec)
    
    st.markdown("**Empfohlene Kurse für fehlende Fähigkeiten:**")
    for skill, recs in grouped.items():
        expander_title = f"{skill} – {min(len(recs), 3)} Kurse"
        with st.expander(expander_title, expanded=False):
            for idx, rec in enumerate(recs[:3], 1):
                title = rec.get('course_title', 'Unbekannter Kurs')
                score = rec.get('similarity_score', 0)
                url = rec.get('course_url', '')
                headline = rec.get('course_headline', '')
                matched_list = rec.get('matched_skills_list') or []
                if not matched_list:
                    matched_list = [skill]
                num_matched = max(len(matched_list), rec.get('num_matched_skills', 0), 1)
                coverage_preview = ', '.join(matched_list[:5])
                if len(matched_list) > 5:
                    coverage_preview += " ..."
                
                st.write(f"{idx}. {title} (Score: {score:.3f}) – deckt {num_matched} Skill(s) ab")
                if headline:
                    st.caption(headline)
                st.write("**Abgedeckte Skills:**")
                for covered_skill in matched_list:
                    st.write(f"- {covered_skill}")
                if url:
                    st.markdown(f"[Zum Kurs]({url})")

def _prepare_berufsbenennungen_df(df):
    """Prepare the alphabetical index for quick searching"""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Berufsbenennungen', 'KldB_Code', 'normalized_label'])
    
    df = df.copy()
    df.columns = [col.strip().lstrip('\ufeff') for col in df.columns]
    
    if 'KldB 2010 (5-Steller)' in df.columns:
        df = df.rename(columns={'KldB 2010 (5-Steller)': 'KldB_Code'})
    
    required_cols = {'Berufsbenennungen', 'KldB_Code'}
    if not required_cols.issubset(set(df.columns)):
        return pd.DataFrame(columns=['Berufsbenennungen', 'KldB_Code', 'normalized_label'])
    
    df = df[['Berufsbenennungen', 'KldB_Code']].dropna()
    df['Berufsbenennungen'] = df['Berufsbenennungen'].astype(str).str.strip()
    df['KldB_Code'] = (
        df['KldB_Code']
        .astype(str)
        .str.extract(r'(\d{5})')[0]
    )
    df = df.dropna(subset=['KldB_Code'])
    df['KldB_Code'] = df['KldB_Code'].str.strip()
    df = df[(df['Berufsbenennungen'] != '') & (df['KldB_Code'] != '')]
    
    records = []
    for _, row in df.iterrows():
        code = row['KldB_Code']
        label = row['Berufsbenennungen']
        aliases = expand_job_aliases(label)
        if not aliases:
            aliases = [label]
        for alias in aliases:
            records.append({
                'Berufsbenennungen': alias,
                'KldB_Code': code,
                'normalized_label': normalize_job_label(alias)
            })
    
    expanded_df = pd.DataFrame(records)
    expanded_df = expanded_df.drop_duplicates(subset=['Berufsbenennungen', 'KldB_Code'])
    return expanded_df

@st.cache_data
def load_berufsbenennungen_dataset(data_dir=DATA_DIR):
    """Loads the alphabetical index with proper umlaut handling"""
    excel_path = os.path.join(data_dir, 'Alphabetisches-Verzeichnis-Berufsbenennungen.xlsx')
    csv_path = os.path.join(data_dir, 'Alphabetisches-Verzeichnis-Berufsbenennungen.csv')
    
    df = pd.DataFrame()
    source = None
    
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(
                excel_path,
                sheet_name='alphabet_Verz_Berufsb',
                header=None,
                skiprows=4,
                names=['Berufsbenennungen', 'KldB 2010 (5-Steller)']
            )
            if not df.empty:
                df = df.dropna(how='all')
            source = 'xlsx'
        except Exception as e:
            st.error(f"Fehler beim Laden der XLSX-Datei '{excel_path}': {str(e)}")
    
    if (df is None or df.empty) and os.path.exists(csv_path):
        encodings = ['utf-8', 'utf-16', 'cp1252', 'latin-1']
        best_df = None
        best_source = None
        
        for enc in encodings:
            try:
                temp_df = pd.read_csv(csv_path, sep=';', encoding=enc, engine='python')
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Fehler beim Laden der CSV-Datei '{csv_path}' mit Encoding {enc}: {str(e)}")
                continue
            
            if 'Berufsbenennungen' not in temp_df.columns:
                continue
            
            contains_replacement = temp_df['Berufsbenennungen'].astype(str).str.contains('�').any()
            best_df = temp_df
            best_source = f'csv ({enc})'
            if not contains_replacement:
                break
        
        if best_df is not None:
            df = best_df
            source = best_source
    
    df = _prepare_berufsbenennungen_df(df)
    if df.empty:
        return df, None
    
    return df, source

def match_job_in_berufsverzeichnis(job_title, berufsbenennungen_df, min_ratio=0.82):
    """Search for a KldB code using the alphabetical index"""
    if not job_title or berufsbenennungen_df is None or berufsbenennungen_df.empty:
        return None, None
    
    normalized_job = normalize_job_label(job_title)
    if not normalized_job:
        return None, None
    
    exact_matches = berufsbenennungen_df[berufsbenennungen_df['normalized_label'] == normalized_job]
    if not exact_matches.empty:
        row = exact_matches.iloc[0]
        return row['KldB_Code'], row['Berufsbenennungen']
    
    contains_matches = berufsbenennungen_df[
        berufsbenennungen_df['normalized_label'].str.contains(normalized_job, case=False, regex=False)
    ]
    if not contains_matches.empty:
        row = contains_matches.iloc[0]
        return row['KldB_Code'], row['Berufsbenennungen']
    
    candidate_df = berufsbenennungen_df
    first_char = normalized_job[0]
    first_char_matches = candidate_df[candidate_df['normalized_label'].str.startswith(first_char, na=False)]
    if not first_char_matches.empty:
        candidate_df = first_char_matches
    
    best_ratio = 0
    best_row = None
    for _, row in candidate_df.iterrows():
        ratio = SequenceMatcher(None, normalized_job, row['normalized_label']).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_row = row
    
    if best_row is not None and best_ratio >= min_ratio:
        return best_row['KldB_Code'], best_row['Berufsbenennungen']
    
    return None, None

def get_kldb_mapping_for_code(kldb_code, kldb_esco_df, preferred_label=None):
    """Returns KldB and ESCO information for a code"""
    if not kldb_code or kldb_esco_df.empty:
        return None, None, None
    
    match = get_kldb_rows(kldb_esco_df, kldb_code)
    if match.empty:
        return None, None, None
    
    if preferred_label:
        preferred_clean = normalize_display_label(preferred_label).lower()
        exact_match = match[
            match['KldB_Label'].apply(lambda x: normalize_display_label(x).lower()) == preferred_clean
        ]
        if not exact_match.empty:
            match = exact_match
        else:
            preferred_norm = normalize_job_label(preferred_label)
            preferred_match = match[match['KldB_Label'].apply(normalize_job_label) == preferred_norm]
            if not preferred_match.empty:
                match = preferred_match
            else:
                contains_match = match[
                    match['KldB_Label'].apply(lambda x: normalize_job_label(x)).str.contains(preferred_norm, na=False)
                ]
                if not contains_match.empty:
                    match = contains_match
    
    row = match.iloc[0]
    label = normalize_display_label(row.get('KldB_Label'))
    if preferred_label:
        label = normalize_display_label(preferred_label)
    return label, row.get('ESCO_Label'), row.get('ESCO_Code')

def get_kldb_rows(kldb_esco_df, kldb_code):
    """Filters rows based on the full or 5-digit KldB code"""
    if kldb_esco_df.empty or not kldb_code:
        return kldb_esco_df.iloc[0:0]
    
    code_str = str(kldb_code).strip()
    if not code_str:
        return kldb_esco_df.iloc[0:0]
    
    if code_str.startswith('B '):
        mask = kldb_esco_df['KldB_Code'].str.strip() == code_str
    else:
        pattern = rf"^B\s+{re.escape(code_str)}-"
        mask = kldb_esco_df['KldB_Code'].str.contains(pattern, case=False, regex=True, na=False)
    
    return kldb_esco_df[mask]

def get_unique_esco_roles(kldb_esco_df, kldb_code):
    """Returns unique ESCO roles for a KldB code"""
    rows = get_kldb_rows(kldb_esco_df, kldb_code)
    if rows.empty:
        return rows
    return rows.drop_duplicates(subset=['ESCO_Code', 'ESCO_Label'])

def select_groupings_for_actor(grouping_ids, groupings, grouping_resources):
    """Select preferred groupings (Skills > Skills > Resources) for a future actor"""
    if not grouping_ids:
        return []
    
    def has_keyword(gid, keyword):
        name = groupings.get(gid, '').lower()
        return keyword in name
    
    prioritized = [
        lambda gid: has_keyword(gid, 'fähigkeit'),
        lambda gid: has_keyword(gid, 'skill'),
        lambda gid: has_keyword(gid, 'resource')
    ]
    
    for predicate in prioritized:
        selected = [gid for gid in grouping_ids if predicate(gid) and grouping_resources.get(gid)]
        if selected:
            return selected
    
    return [gid for gid in grouping_ids if grouping_resources.get(gid)]

def extract_future_profiles(actors, associations, groupings, grouping_resources):
    """Extracts future employee profiles (Employees A–E) along with their skills"""
    future_profiles = []
    actor_ids_by_name = defaultdict(list)
    
    for actor in actors:
        actor_ids_by_name[actor['name']].append(actor['identifier'])
    
    # Method 1: Search for BusinessActors named “Employee A-E”"
    for actor_name, ids in actor_ids_by_name.items():
        if not re.match(r'^Mitarbeiter/in [A-E]$', actor_name):
            continue
        
        related_groupings = set()
        for assoc in associations:
            if assoc['source'] in ids and assoc['target'] in groupings:
                related_groupings.add(assoc['target'])
        
        selected_groupings = select_groupings_for_actor(list(related_groupings), groupings, grouping_resources)
        skills = set()
        for gid in selected_groupings:
            for skill in grouping_resources.get(gid, []):
                if skill:
                    skills.add(skill)
        
        skills_list = sorted(skills, key=lambda s: s.lower())
        future_profiles.append({
            'name': actor_name,
            'skills': skills_list,
            'skill_count': len(skills_list)
        })
    
    # Method 2: If no profiles were found via Actors, search for groupings that include “Target Employee Role” in their name
    if not future_profiles:
        for grouping_id, grouping_name in groupings.items():
            # Search for the pattern “Target Employee Role A,” “Target Employee Role B,” etc.
            match = re.search(r'Ziel-Mitarbeiterrolle\s+([A-E])', grouping_name, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                profile_name = f"Mitarbeiter/in {letter}"
                
                # Collect skills from this grouping
                skills = set()
                for skill in grouping_resources.get(grouping_id, []):
                    if skill:
                        skills.add(skill)
                
                skills_list = sorted(skills, key=lambda s: s.lower())
                if skills_list:  # Add only if skills are available
                    future_profiles.append({
                        'name': profile_name,
                        'skills': skills_list,
                        'skill_count': len(skills_list)
                    })
    
    future_profiles.sort(key=lambda x: x['name'])
    return future_profiles

LOAD_DATA_CACHE_VERSION = "load_data_v8"

def enhance_kldb_mapping(kldb_esco_df, berufsbenennungen_df):
    """Aligns KldB terms with the Alphabetical Index and adds synonyms"""
    if kldb_esco_df.empty:
        return kldb_esco_df, 0, 0
    
    enhanced_df = kldb_esco_df.copy()
    enhanced_df['KldB_Label'] = enhanced_df['KldB_Label'].astype(str).str.strip()
    enhanced_df['KldB_Code'] = enhanced_df['KldB_Code'].astype(str).str.strip()
    enhanced_df['KldB_Code_5'] = enhanced_df['KldB_Code'].str.extract(r'(\d{5})')
    enhanced_df = enhanced_df[enhanced_df['KldB_Label'].apply(is_valid_kldb_label)].copy()
    
    if berufsbenennungen_df is None or berufsbenennungen_df.empty:
        return enhanced_df, 0, 0
    
    alias_map = (
        berufsbenennungen_df[['KldB_Code', 'Berufsbenennungen']]
        .dropna()
        .groupby('KldB_Code')['Berufsbenennungen']
        .apply(list)
        .to_dict()
    )
    
    replacements = 0
    alias_rows = []
    
    for code, alias_list in alias_map.items():
        code_str = str(code).strip()
        if not code_str:
            continue
        code_5 = re.findall(r'\d{5}', code_str)
        target_code = code_5[0] if code_5 else code_str[-5:]
        mask = enhanced_df['KldB_Code_5'] == target_code
        if not mask.any():
            continue
        
        clean_aliases = []
        for alias in alias_list:
            alias_str = str(alias).strip()
            if not is_valid_kldb_label(alias_str):
                continue
            clean_aliases.append(alias_str)
        
        if not clean_aliases:
            continue
        
        current_label = enhanced_df.loc[mask, 'KldB_Label'].iloc[0]
        current_norm = normalize_job_label(current_label)
        
        # Prefer an alias that matches the current label; otherwise, use the shortest entry
        chosen_label = None
        for alias in clean_aliases:
            if normalize_job_label(alias) == current_norm:
                chosen_label = alias
                break
        if chosen_label is None:
            chosen_label = select_preferred_alias(clean_aliases) or clean_aliases[0]
        elif not is_informative_kldb_label(chosen_label):
            better_alias = select_preferred_alias(clean_aliases)
            if better_alias:
                chosen_label = better_alias
        
        # Update all existing lines for this code
        enhanced_df.loc[mask, 'KldB_Label'] = chosen_label
        replacements += mask.sum()
        
        # Add more synonyms as additional entries
        base_row = enhanced_df[mask].iloc[0].copy()
        for alias in clean_aliases:
            if normalize_job_label(alias) == normalize_job_label(chosen_label):
                continue
            if not is_informative_kldb_label(alias):
                continue
            alias_row = base_row.copy()
            alias_row['KldB_Label'] = alias
            alias_rows.append(alias_row)
    
    if alias_rows:
        enhanced_df = pd.concat([enhanced_df, pd.DataFrame(alias_rows)], ignore_index=True)
    
    enhanced_df = enhanced_df.drop_duplicates(subset=['KldB_Code', 'KldB_Label', 'ESCO_Code'])
    enhanced_df = enhanced_df[enhanced_df['KldB_Label'].apply(is_valid_kldb_label)]
    enhanced_df = enhanced_df.drop(columns=['KldB_Code_5'], errors='ignore')
    return enhanced_df, replacements, len(alias_rows)

@st.cache_data
def load_data(cache_buster=LOAD_DATA_CACHE_VERSION):
    """Loads all required CSV files"""
    try:
        # KldB on ESCO Mapping
        kldb_esco_df = pd.read_csv(data_path('KldB_to_ESCO_Mapping_clean.csv'))
        
        # Load the alphabetical index and integrate it into the mapping
        berufsbenennungen_df, berufsbenennungen_source = load_berufsbenennungen_dataset()
        if not berufsbenennungen_df.empty:
            kldb_esco_df, label_updates, alias_additions = enhance_kldb_mapping(kldb_esco_df, berufsbenennungen_df)
        else:
            label_updates = alias_additions = 0
        
        # ESCO Occupation-Skill Relationships (the correct file!)
        try:
            occupation_skill_relations_df = pd.read_csv(data_path('occupationSkillRelations_de.csv'), on_bad_lines='skip')
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Beruf-Skill Beziehungen: {str(e)}")
            occupation_skill_relations_df = pd.DataFrame()
        
        # ESCO Careers
        try:
            occupations_df = pd.read_csv(data_path('occupations_de.csv'), on_bad_lines='skip')
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Berufe: {str(e)}")
            occupations_df = pd.DataFrame()
        
        # ESCO Skills (German)
        try:
            skills_df = pd.read_csv(data_path('skills_de.csv'), on_bad_lines='skip')
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Skills (Deutsch): {str(e)}")
            skills_df = pd.DataFrame()
        
        # ESCO Skills (English)
        try:
            skills_en_df = pd.read_csv(data_path('skills_en.csv'), on_bad_lines='skip')
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Skills (Englisch): {str(e)}")
            skills_en_df = pd.DataFrame()
        
        # EURES Skills Mapping
        try:
            eures_skills_df = pd.read_csv(data_path('EURESmapping_skills_DE.csv'), on_bad_lines='skip')
        except Exception as e:
            st.error(f"Fehler beim Laden des EURES Skills Mappings: {str(e)}")
            eures_skills_df = pd.DataFrame()
        
        # Udemy Courses
        try:
            udemy_courses_df = pd.read_csv(data_path('Udemy_Course_Desc.csv'), on_bad_lines='skip')
        except Exception as e:
            st.error(f"Fehler beim Laden der Udemy Kurse: {str(e)}")
            udemy_courses_df = pd.DataFrame()
        
        # Employee data - Load from employees_data.csv first, then fall back to employee_input.csv
        try:
            employees_df = load_employees_from_csv()
            if employees_df.empty:
                # Fallback to employee_input.csv
                employees_df = pd.read_csv(data_path('employee_input.csv'))
        except Exception as e:
            st.error(f"Fehler beim Laden der Mitarbeiterdaten: {str(e)}")
            employees_df = pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role'])
        
        # Create an ESCO Career-Skill Mapping
        occupation_skills_mapping = get_all_occupation_skills_direct(occupation_skill_relations_df, skills_df)
        
        # Load XML data from Archi
        archi_xml_path = data_path('DigiVan.xml')
        archi_data = None
        if os.path.exists(archi_xml_path):
            
            try:
                archi_data = parse_archi_xml(archi_xml_path)
                if not archi_data:
                    st.warning("Archi XML-Daten konnten nicht geparst werden")
            except Exception as e:
                st.error(f"Fehler beim Laden der XML-Daten: {str(e)}")
                archi_data = None
        else:
            st.warning(f"Archi XML-Datei nicht gefunden: {archi_xml_path}")
            st.write(f"**Erwarteter Pfad:** {archi_xml_path}")
        
        # Load XML Data for Skill Matching
        kompetenzabgleich_data = None
        kompetenzabgleich_xml_path = data_path("Kompetenzabgleich.xml")
        if os.path.exists(kompetenzabgleich_xml_path):
            try:
                kompetenzabgleich_data = parse_kompetenzabgleich_xml(kompetenzabgleich_xml_path)
                if not (kompetenzabgleich_data and kompetenzabgleich_data.get('success')):
                    st.warning("Kompetenzabgleich XML-Daten konnten nicht geladen werden")
            except Exception as e:
                st.error(f"Fehler beim Laden der Kompetenzabgleich XML-Daten: {str(e)}")
                kompetenzabgleich_data = None
        else:
            # File not found - do not display a warning
            pass
        
        return (employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, 
                eures_skills_df, udemy_courses_df, occupations_df, occupation_skills_mapping, skills_en_df, archi_data, kompetenzabgleich_data, berufsbenennungen_df)
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {str(e)}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame())

@st.cache_data
def get_all_occupation_skills_direct(occupation_skill_relations_df, skills_df):
    """Creates a direct mapping of all occupations to their skills"""
    if occupation_skill_relations_df.empty or skills_df.empty:
        return {}
    
    # Create a mapping of skill URIs to skill labels
    skill_uri_to_label = {}
    for _, skill_row in skills_df.iterrows():
        skill_uri = str(skill_row.get('conceptUri', ''))
        skill_label = str(skill_row.get('preferredLabel', ''))
        if skill_uri and skill_label and not pd.isna(skill_uri) and not pd.isna(skill_label):
            skill_uri_to_label[skill_uri] = skill_label
    
    # Create the Job-to-Skills Mapping
    occupation_skills = {}
    
    for _, relation_row in occupation_skill_relations_df.iterrows():
        occupation_uri = str(relation_row.get('occupationUri', ''))
        skill_uri = str(relation_row.get('skillUri', ''))
        relation_type = str(relation_row.get('relationType', ''))
        skill_type = str(relation_row.get('skillType', ''))
        
        if occupation_uri and skill_uri and not pd.isna(occupation_uri) and not pd.isna(skill_uri):
            if occupation_uri not in occupation_skills:
                occupation_skills[occupation_uri] = []
            
            # Get the Skill Badge
            skill_label = skill_uri_to_label.get(skill_uri, skill_uri)
            
            occupation_skills[occupation_uri].append({
                'skill_uri': skill_uri,
                'skill_label': skill_label,
                'relation_type': relation_type,
                'skill_type': skill_type,
                'is_essential': relation_type.lower() == 'essential'
            })
    
    return occupation_skills

def find_occupation_by_label(occupations_df, target_label):
    """Find a career based on its label"""
    if occupations_df.empty:
        return None
    
    # Search for an exact match
    exact_match = occupations_df[
        occupations_df['preferredLabel'].astype(str).str.contains(target_label, case=False, na=False)
    ]
    
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    # Search by partial match
    partial_match = occupations_df[
        occupations_df['preferredLabel'].astype(str).str.contains(target_label.split()[0], case=False, na=False)
    ]
    
    if not partial_match.empty:
        return partial_match.iloc[0]
    
    return None

def get_skills_for_occupation_simple(occupation_label, occupation_skills_mapping, occupations_df, skill_mapping_with_english=None):
    """A simple tool for finding skills related to a profession (with German and English labels)"""
    
    # 1. Find the occupation in the occupations_df
    occupation = find_occupation_by_label(occupations_df, occupation_label)
    
    if occupation is None:
        return []
    
    # 2. Acquire the skills needed for this profession
    occupation_uri = str(occupation['conceptUri'])
    
    if occupation_uri in occupation_skills_mapping:
        skills = occupation_skills_mapping[occupation_uri]
        
        # 3. Add English labels to skills, if available
        if skill_mapping_with_english:
            enhanced_skills = []
            for skill in skills:
                skill_uri = skill.get('skill_uri', '')
                enhanced_skill = skill.copy()
                
                # Add English labels, if available
                if skill_uri in skill_mapping_with_english:
                    english_label = skill_mapping_with_english[skill_uri]['english']
                    enhanced_skill['skill_label_english'] = english_label
                    enhanced_skill['skill_labels_combined'] = f"{skill['skill_label']} | {english_label}"
                else:
                    enhanced_skill['skill_label_english'] = skill['skill_label']
                    enhanced_skill['skill_labels_combined'] = skill['skill_label']
                
                enhanced_skills.append(enhanced_skill)
            
            return enhanced_skills
        
        return skills
    
    return []

def render_skills_two_columns(skills, left_title="Essentiell", right_title="Optional"):
    """Displays skills in a two-column layout: essential skills on the left, optional skills on the right.

    Supports both skill dictionaries with ‘skill_label’/'skill_name' and raw strings.
    """
    if skills is None:
        skills = []

    def extract_label(skill):
        if isinstance(skill, dict):
            return str(skill.get('skill_label') or skill.get('skill_name') or '').strip()
        return str(skill).strip()

    essential_skills = [s for s in skills if isinstance(s, dict) and s.get('is_essential', False)]
    optional_skills = [s for s in skills if not (isinstance(s, dict) and s.get('is_essential', False))]

    col_left, col_right = st.columns(2)
    with col_left:
        st.write(f"**{left_title}:**")
        if essential_skills:
            for s in essential_skills:
                label = extract_label(s)
                if label:
                    st.write(f"• {label}")
        else:
            st.write("—")

def render_skills_two_columns_table(skills, left_title="Essentiell", right_title="Optional"):
    """Displays skills in a table with two columns: essential on the left, optional on the right."""
    if skills is None:
        skills = []
    def extract_label(skill):
        if isinstance(skill, dict):
            return str(skill.get('skill_label') or skill.get('skill_name') or '').strip()
        return str(skill).strip()
    essential = [extract_label(s) for s in skills if isinstance(s, dict) and s.get('is_essential', False)]
    optional = [extract_label(s) for s in skills if not (isinstance(s, dict) and s.get('is_essential', False))]
    max_len = max(len(essential), len(optional)) if (essential or optional) else 0
    rows = []
    for i in range(max_len):
        left_val = essential[i] if i < len(essential) else ""
        right_val = optional[i] if i < len(optional) else ""
        rows.append({left_title: left_val, right_title: right_val})
    df = pd.DataFrame(rows, columns=[left_title, right_title]) if rows else pd.DataFrame(columns=[left_title, right_title])
    # Display without an index column
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        # Fallback for older versions of Streamlit that do not support `hide_index`
        df = df.reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

def render_missing_skills_with_favorites(missing_skills, session_key_prefix="favorite"):
    """Displays missing skills in a table and allows users to mark favorites directly in the table."""
    if not missing_skills:
        return set()
    
    # Initialize session state for favorite skills, if not already set
    if 'favorite_skills' not in st.session_state:
        st.session_state.favorite_skills = set()
    
    # Check if anything has changed for an immediate update
    needs_rerun = False
    
    # Divide Skills into Essential and Optional
    essential_skills = [s for s in missing_skills if isinstance(s, dict) and s.get('is_essential', False)]
    optional_skills = [s for s in missing_skills if not (isinstance(s, dict) and s.get('is_essential', False))]
    
    def extract_label(skill):
        if isinstance(skill, dict):
            return str(skill.get('skill_label') or skill.get('skill_name') or '').strip()
        return str(skill).strip()
    
    max_len = max(len(essential_skills), len(optional_skills)) if (essential_skills or optional_skills) else 0
    
    # Create a custom table with checkboxes within the table
    # Table headings
    header_col1, header_col2, header_col3, header_col4 = st.columns([1, 3, 1, 3])
    with header_col1:
        st.write("**Favorit**")
    with header_col2:
        st.write("**Essentiell**")
    with header_col3:
        st.write("**Favorit**")
    with header_col4:
        st.write("**Optional**")
    
    # Create each row of the table with checkboxes
    for i in range(max_len):
        row_col1, row_col2, row_col3, row_col4 = st.columns([1, 3, 1, 3])
        
        with row_col1:
            if i < len(essential_skills):
                skill = essential_skills[i]
                skill_uri = skill.get('skill_uri', '')
                is_favorite = skill_uri in st.session_state.favorite_skills if skill_uri else False
                
                favorite_key = f"{session_key_prefix}_ess_{i}_{skill_uri if skill_uri else i}"
                checkbox_value = st.checkbox(
                    "",
                    value=is_favorite,
                    key=favorite_key,
                    label_visibility="collapsed"
                )
                
                if skill_uri:
                    was_favorite = skill_uri in st.session_state.favorite_skills
                    if checkbox_value and not was_favorite:
                        st.session_state.favorite_skills.add(skill_uri)
                        needs_rerun = True
                    elif not checkbox_value and was_favorite:
                        st.session_state.favorite_skills.discard(skill_uri)
                        needs_rerun = True
        
        with row_col2:
            if i < len(essential_skills):
                skill_label = extract_label(essential_skills[i])
                skill_uri = essential_skills[i].get('skill_uri', '')
                is_favorite = skill_uri in st.session_state.favorite_skills if skill_uri else False
                display_label = f"★ {skill_label}" if is_favorite else skill_label
                st.write(display_label)
        
        with row_col3:
            if i < len(optional_skills):
                skill = optional_skills[i]
                skill_uri = skill.get('skill_uri', '')
                is_favorite = skill_uri in st.session_state.favorite_skills if skill_uri else False
                
                favorite_key = f"{session_key_prefix}_opt_{i}_{skill_uri if skill_uri else i}"
                checkbox_value = st.checkbox(
                    "",
                    value=is_favorite,
                    key=favorite_key,
                    label_visibility="collapsed"
                )
                
                if skill_uri:
                    was_favorite = skill_uri in st.session_state.favorite_skills
                    if checkbox_value and not was_favorite:
                        st.session_state.favorite_skills.add(skill_uri)
                        needs_rerun = True
                    elif not checkbox_value and was_favorite:
                        st.session_state.favorite_skills.discard(skill_uri)
                        needs_rerun = True
        
        with row_col4:
            if i < len(optional_skills):
                skill_label = extract_label(optional_skills[i])
                skill_uri = optional_skills[i].get('skill_uri', '')
                is_favorite = skill_uri in st.session_state.favorite_skills if skill_uri else False
                display_label = f"★ {skill_label}" if is_favorite else skill_label
                st.write(display_label)
    
    # Run the script again immediately if anything has changed, so that the star is displayed right away
    if needs_rerun:
        st.rerun()
    
    # Show the number of favorite skills and highlighted skills
    favorite_count = len([s for s in missing_skills if isinstance(s, dict) and s.get('skill_uri', '') in st.session_state.favorite_skills])
    if favorite_count > 0:
        favorite_skill_labels = []
        for s in missing_skills:
            if isinstance(s, dict) and s.get('skill_uri', '') in st.session_state.favorite_skills:
                favorite_skill_labels.append(s.get('skill_label', ''))
        
        st.info(f"**{favorite_count} Skill(s) als Favoriten markiert:** {', '.join(favorite_skill_labels[:5])}{'...' if len(favorite_skill_labels) > 5 else ''}. Diese werden bei Kursempfehlungen priorisiert.")
    
    return st.session_state.favorite_skills

@st.cache_data
def create_employee_profile(employee_id, kldb_code, manual_skills, kldb_esco_df, occupation_skill_relations_df, skills_df, occupation_skills_mapping, occupations_df, saved_esco_role=None, manual_essential_skills='', manual_optional_skills='', removed_skills=''):
    """Create a competency profile for an employee based on their current role"""
    
    # Make sure all parameters are strings
    employee_id = str(employee_id)
    kldb_code = str(kldb_code).strip() if kldb_code and str(kldb_code) != 'nan' else ''
    manual_skills = str(manual_skills) if manual_skills and str(manual_skills) != 'nan' else ''
    manual_essential_skills = str(manual_essential_skills) if manual_essential_skills and str(manual_essential_skills) != 'nan' else ''
    manual_optional_skills = str(manual_optional_skills) if manual_optional_skills and str(manual_optional_skills) != 'nan' else ''
    removed_skills = str(removed_skills) if removed_skills and str(removed_skills) != 'nan' else ''
    saved_esco_role = str(saved_esco_role).strip() if saved_esco_role and str(saved_esco_role) != 'nan' else ''
    
    # Normalize KldB code (remove spaces)
    if kldb_code:
        kldb_code = kldb_code.replace(' ', '').strip()
    
    # 1. Find the employee's CURRENT role
    current_occupation = pd.DataFrame()
    
    # Priority 1: Check for ESCO role (if applicable)
    if saved_esco_role:
        current_occupation = kldb_esco_df[
            kldb_esco_df['ESCO_Label'].astype(str).str.strip() == saved_esco_role
        ]
        
        # If a KldB code is available, refine your search
        if not current_occupation.empty and kldb_code:
            current_occupation = current_occupation[
                current_occupation['KldB_Code'].astype(str).str.replace(' ', '').str.strip() == kldb_code
            ]
    
    # Priority 2: Search by KldB code (if ESCO role is not found or does not exist)
    if current_occupation.empty and kldb_code:
        # Normalize KldB codes in the DataFrame for comparison
        kldb_esco_df_normalized = kldb_esco_df.copy()
        kldb_esco_df_normalized['KldB_Code_Normalized'] = kldb_esco_df_normalized['KldB_Code'].astype(str).str.replace(' ', '').str.strip()
        
        current_occupation = kldb_esco_df_normalized[
            kldb_esco_df_normalized['KldB_Code_Normalized'] == kldb_code
        ]
        
        # Remove temporary column
        if not current_occupation.empty:
            current_occupation = current_occupation.drop(columns=['KldB_Code_Normalized'])
    
    # Priority 3: Fallback – Search for similar KldB codes (substring match)
    if current_occupation.empty and kldb_code:
        kldb_esco_df_normalized = kldb_esco_df.copy()
        kldb_esco_df_normalized['KldB_Code_Normalized'] = kldb_esco_df_normalized['KldB_Code'].astype(str).str.replace(' ', '').str.strip()
        
        current_occupation = kldb_esco_df_normalized[
            kldb_esco_df_normalized['KldB_Code_Normalized'].str.contains(kldb_code, na=False, regex=False)
        ]
        
        if not current_occupation.empty:
            current_occupation = current_occupation.drop(columns=['KldB_Code_Normalized'])
    
    # Priority 4: Fallback – Search only by ESCO role (without KldB code)
    if current_occupation.empty and saved_esco_role:
        current_occupation = kldb_esco_df[
            kldb_esco_df['ESCO_Label'].astype(str).str.strip().str.contains(saved_esco_role, case=False, na=False)
        ]
    
    # If you still can't find anything, create a basic profile with your manual skills
    if current_occupation.empty:
        # Create a minimal profile without a role, but with manual skills
        filtered_skills = []
        
        # Add manual essential skills
        manual_essential_list = [s.strip() for s in manual_essential_skills.split(';') if s.strip()]
        for skill in manual_essential_list:
            skill_uri = f"manual_essential_{skill.lower().replace(' ', '_')}"
            for _, skill_row in skills_df.iterrows():
                if str(skill_row.get('preferredLabel', '')).lower() == skill.lower():
                    skill_uri = str(skill_row.get('conceptUri', skill_uri))
                    break
            filtered_skills.append({
                'skill_uri': skill_uri,
                'skill_label': skill,
                'relation_type': 'manual_essential',
                'skill_type': 'manual_essential',
                'is_essential': True
            })
        
        # Add manual optional skills
        manual_optional_list = [s.strip() for s in manual_optional_skills.split(';') if s.strip()]
        for skill in manual_optional_list:
            skill_uri = f"manual_optional_{skill.lower().replace(' ', '_')}"
            for _, skill_row in skills_df.iterrows():
                if str(skill_row.get('preferredLabel', '')).lower() == skill.lower():
                    skill_uri = str(skill_row.get('conceptUri', skill_uri))
                    break
            filtered_skills.append({
                'skill_uri': skill_uri,
                'skill_label': skill,
                'relation_type': 'manual_optional',
                'skill_type': 'manual_optional',
                'is_essential': False
            })
        
        # Add original manual skills
        manual_skills_list = [s.strip() for s in manual_skills.split(';') if s.strip()]
        for skill in manual_skills_list:
            filtered_skills.append({
                'skill_uri': f"manual_{skill.lower().replace(' ', '_')}",
                'skill_label': skill,
                'relation_type': 'manual',
                'skill_type': 'manual',
                'is_essential': True
            })
        
        # Create a minimal profile without a roll
        return {
            'employee_id': employee_id,
            'kldb_code': kldb_code if kldb_code else '',
            'current_role': {
                'KldB_Code': kldb_code if kldb_code else '',
                'KldB_Label': 'Keine Rolle zugeordnet',
                'ESCO_Code': '',
                'ESCO_Label': saved_esco_role if saved_esco_role else 'Keine Rolle zugeordnet'
            },
            'skills': filtered_skills,
            'manual_skills': manual_skills_list,
            'manual_essential_skills': manual_essential_list,
            'manual_optional_skills': manual_optional_list,
            'removed_skills': []
        }
    
    # 2. Consider only the employee's primary role
    primary_role = current_occupation.iloc[0]
    esco_label = str(primary_role['ESCO_Label'])
    
    # 3. Identify the skills required for the employee’s CURRENT role
    current_role_skills = get_skills_for_occupation_simple(esco_label, occupation_skills_mapping, occupations_df)
    
    # 4. Process removed skills
    removed_skills_list = [s.strip().lower() for s in removed_skills.split(';') if s.strip()]
    filtered_skills = []
    for skill in current_role_skills:
        if skill['skill_label'].lower() not in removed_skills_list:
            filtered_skills.append(skill)
    
    # 5. Add manual Essential Skills
    manual_essential_list = [s.strip() for s in manual_essential_skills.split(';') if s.strip()]
    for skill in manual_essential_list:
        # Search for the corresponding ESCO skill URI
        skill_uri = f"manual_essential_{skill.lower().replace(' ', '_')}"
        
        # Try to find the actual ESCO URI
        for _, skill_row in skills_df.iterrows():
            if str(skill_row.get('preferredLabel', '')).lower() == skill.lower():
                skill_uri = str(skill_row.get('conceptUri', skill_uri))
                break
        
        filtered_skills.append({
            'skill_uri': skill_uri,
            'skill_label': skill,
            'relation_type': 'manual_essential',
            'skill_type': 'manual_essential',
            'is_essential': True
        })
    
    # 6. Add manual optional skills
    manual_optional_list = [s.strip() for s in manual_optional_skills.split(';') if s.strip()]
    for skill in manual_optional_list:
        # Search for the corresponding ESCO skill URI
        skill_uri = f"manual_optional_{skill.lower().replace(' ', '_')}"
        
        # Try to find the actual ESCO URI
        for _, skill_row in skills_df.iterrows():
            if str(skill_row.get('preferredLabel', '')).lower() == skill.lower():
                skill_uri = str(skill_row.get('conceptUri', skill_uri))
                break
        
        filtered_skills.append({
            'skill_uri': skill_uri,
            'skill_label': skill,
            'relation_type': 'manual_optional',
            'skill_type': 'manual_optional',
            'is_essential': False
        })
    
    # 7. Add original manual skills (for compatibility)
    manual_skills_list = [s.strip() for s in manual_skills.split(';') if s.strip()]
    for skill in manual_skills_list:
        filtered_skills.append({
            'skill_uri': f"manual_{skill.lower().replace(' ', '_')}",
            'skill_label': skill,
            'relation_type': 'manual',
            'skill_type': 'manual',
            'is_essential': True  # Treat manual skills as essential
        })
    
    return {
        'employee_id': employee_id,
        'kldb_code': kldb_code,
        'current_role': primary_role.to_dict(),
        'skills': filtered_skills,
        'manual_skills': manual_skills_list,
        'manual_essential_skills': manual_essential_list,
        'manual_optional_skills': manual_optional_list,
        'removed_skills': removed_skills_list
    }

@st.cache_data
def get_all_esco_occupations(kldb_esco_df):
    """List all available ESCO jobs"""
    unique_occupations = kldb_esco_df[['ESCO_Code', 'ESCO_Label']].drop_duplicates()
    return unique_occupations.to_dict('records')

@st.cache_data
def calculate_occupation_match(employee_profile, target_occupation, occupation_skill_relations_df, skills_df, occupation_skills_mapping, occupations_df):
    """Calculates the match between the employee (current role) and the new target role"""
    
    if not employee_profile or not target_occupation:
        return None
    
    # Master the skills of the NEW future role
    target_esco_label = str(target_occupation.get('ESCO_Label', ''))
    target_role_skills = get_skills_for_occupation_simple(target_esco_label, occupation_skills_mapping, occupations_df)
    
    # Retrieve the skills associated with the employee's CURRENT role
    current_role_skills = employee_profile['skills']
    
    # Compare Skills: What the employee has vs. what the new role requires
    current_skill_labels = [skill['skill_label'].lower() for skill in current_role_skills]
    target_skill_labels = [skill['skill_label'].lower() for skill in target_role_skills]
    
    # Calculate matches (skills the employee already has)
    matching_skills = []
    missing_skills = []
    
    for target_skill in target_role_skills:
        target_label = target_skill['skill_label'].lower()
        if target_label in current_skill_labels:
            matching_skills.append(target_skill)
        else:
            missing_skills.append(target_skill)
    
    # Calculate percentages
    total_target_skills = len(target_role_skills)
    match_count = len(matching_skills)
    
    if total_target_skills == 0:
        return {
            'match_percentage': 0,
            'weighted_fit_percentage': 0,
            'matching_skills': [],
            'missing_skills': [],
            'has_target_skills': False,
            'current_role': employee_profile.get('current_role', {}),
            'target_role': target_occupation
        }
    
    match_percentage = (match_count / total_target_skills) * 100
    
    # Calculate Weighted Fit (essential skills count double)
    weighted_matches = 0
    weighted_total = 0
    
    for target_skill in target_role_skills:
        weight = 2 if target_skill['is_essential'] else 1
        weighted_total += weight
        
        if target_skill['skill_label'].lower() in current_skill_labels:
            weighted_matches += weight
    
    weighted_fit_percentage = (weighted_matches / weighted_total) * 100 if weighted_total > 0 else 0
    
    return {
        'match_percentage': match_percentage,
        'weighted_fit_percentage': weighted_fit_percentage,
        'matching_skills': matching_skills,
        'missing_skills': missing_skills,
        'has_target_skills': True,
        'current_role': employee_profile.get('current_role', {}),
        'target_role': target_occupation
    }

@st.cache_data
def compare_employees_for_target_role(target_occupation, employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, occupation_skills_mapping, occupations_df):
    """Compares all employees against a target role and returns the best matches"""
    
    if target_occupation is None or employees_df.empty:
        return []
    
    employee_scores = []
    
    for _, employee in employees_df.iterrows():
        employee_id = employee['Employee_ID']
        current_kldb = employee.get('KldB_5_digit', '')
        current_manual_skills = employee.get('Manual_Skills', '')
        current_esco_role = employee.get('ESCO_Role', '')
        current_manual_essential_skills = employee.get('Manual_Essential_Skills', '')
        current_manual_optional_skills = employee.get('Manual_Optional_Skills', '')
        current_removed_skills = employee.get('Removed_Skills', '')
        
        # Create Employee Profile
        employee_profile = create_employee_profile(
            employee_id,
            current_kldb,
            current_manual_skills,
            kldb_esco_df,
            occupation_skill_relations_df,
            skills_df,
            occupation_skills_mapping,
            occupations_df,
            current_esco_role,
            current_manual_essential_skills,
            current_manual_optional_skills,
            current_removed_skills
        )
        
        if employee_profile:
            # Calculate the match for this target role
            match_result = calculate_occupation_match(
                employee_profile, 
                target_occupation, 
                occupation_skill_relations_df, 
                skills_df, 
                occupation_skills_mapping, 
                occupations_df
            )
            
            if match_result and match_result['has_target_skills']:
                employee_scores.append({
                    'employee_id': employee_id,
                    'employee_name': employee.get('Name', f'ID: {employee_id}'),
                    'current_role': employee_profile.get('current_role', {}).get('KldB_Label', 'Unbekannt'),
                    'match_percentage': match_result['match_percentage'],
                    'weighted_fit_percentage': match_result['weighted_fit_percentage'],
                    'matching_skills_count': len(match_result['matching_skills']),
                    'missing_skills_count': len(match_result['missing_skills'])
                })
    
    # Sort by weighted fit score (descending)
    employee_scores.sort(key=lambda x: x['weighted_fit_percentage'], reverse=True)
    
    return employee_scores

@st.cache_data
def preprocess_text(text):
    """Bereitet Text für Tokenisierung vor"""
    if pd.isna(text):
        return ""
    
    # Convert to a string and lowercase
    text = str(text).lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Simple tokenization without NLTK
    tokens = text.split()
    
    # Simple stopwords (German and English)
    stop_words = {
        'der', 'die', 'das', 'und', 'oder', 'aber', 'für', 'mit', 'von', 'zu', 'in', 'auf', 'an', 'bei',
        'the', 'and', 'or', 'but', 'for', 'with', 'from', 'to', 'in', 'on', 'at', 'by', 'is', 'are', 'was', 'were',
        'ein', 'eine', 'einer', 'eines', 'einem', 'einen', 'a', 'an', 'this', 'that', 'these', 'those'
    }
    
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def find_udemy_courses_for_skills(missing_skills, udemy_courses_df, top_k=5):
    """Find the right Udemy courses to fill skill gaps with improved score calculation and multi-skill matching"""
    
    try:
        if not missing_skills or udemy_courses_df.empty:
            return []
        
        # Check whether the required columns are present
        required_columns = ['Title', 'Headline', 'Description', 'URL', 'Price', 'Language']
        missing_columns = [col for col in required_columns if col not in udemy_courses_df.columns]
        if missing_columns:
            st.warning(f"Fehlende Spalten in Udemy-Daten: {missing_columns}")
            return []
        
        # Import the required libraries
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import re
        
        # Prepare course data (if you haven't already)
        if 'processed_text' not in udemy_courses_df.columns:
            udemy_courses_df['processed_text'] = (
                udemy_courses_df['Title'].fillna('') + ' ' +
                udemy_courses_df['Headline'].fillna('') + ' ' +
                udemy_courses_df['Description'].fillna('')
            ).apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        
        # TF-IDF vectorization for all courses
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        course_vectors = vectorizer.fit_transform(udemy_courses_df['processed_text'])
        
        # Retrieve preferred skills from the session state, if available
        favorite_skills_uris = set()
        if 'favorite_skills' in st.session_state:
            favorite_skills_uris = st.session_state.favorite_skills
        
        # Extract skill information
        skill_info_list = []
        for skill in missing_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('skill_label', str(skill))
                skill_uri = skill.get('skill_uri', '')
                is_essential = skill.get('is_essential', False)
            else:
                skill_name = str(skill)
                skill_uri = ''
                is_essential = False
            
            is_favorite = skill_uri in favorite_skills_uris if skill_uri else False
            
            # Extract important keywords from the skill name
            skill_text = re.sub(r'[^\w\s]', '', skill_name.lower())
            # Remove common stop words
            stop_words = {'für', 'und', 'der', 'die', 'das', 'mit', 'von', 'zu', 'in', 'auf', 'an', 'bei', 'über', 'unter'}
            skill_words = [w for w in skill_text.split() if w not in stop_words and len(w) > 2]
            skill_keywords = ' '.join(skill_words)
            
            skill_info_list.append({
                'skill_name': skill_name,
                'skill_uri': skill_uri,
                'skill_text': skill_text,
                'skill_keywords': skill_keywords,
                'skill_words': skill_words,
                'is_favorite': is_favorite,
                'is_essential': is_essential
            })
        
        # Create a dictionary for all courses and their matches
        course_matches = {}  # course_idx -> {course_data, matched_skills: [], total_score: 0}
        
        # For every skill: Find the right courses
        for skill_info in skill_info_list:
            skill_name = skill_info['skill_name']
            skill_text = skill_info['skill_text']
            skill_keywords = skill_info['skill_keywords']
            skill_words = skill_info['skill_words']
            is_favorite = skill_info['is_favorite']
            is_essential = skill_info['is_essential']
            
            # Prepare skill text for TF-IDF
            skill_vector = vectorizer.transform([skill_keywords if skill_keywords else skill_text])
            
            # Calculate cosine similarity for all courses
            similarities = cosine_similarity(skill_vector, course_vectors).flatten()
            
            # Find courses with a relevant similarity score
            for idx, similarity in enumerate(similarities):
                if similarity > 0.05:  # Higher threshold for relevance
                    course = udemy_courses_df.iloc[idx]
                    course_id = idx
                    
                    title = str(course.get('Title', '')).lower()
                    headline = str(course.get('Headline', '')).lower()
                    description = str(course.get('Description', '')).lower()
                    course_text = f"{title} {headline} {description}"
                    
                    # Improved relevance check: At least 2 important keywords must be included
                    matching_keywords = sum(1 for word in skill_words if word in course_text)
                    if matching_keywords < 2 and len(skill_words) >= 2:
                        continue  # Überspringe Kurse mit zu wenigen Keyword-Matches
                    
                    # Calculate detailed score
                    base_similarity = float(similarity)
                    
                    # Keyword Match Bonus (the more keywords, the better)
                    keyword_bonus = min(0.3, matching_keywords * 0.1 / max(len(skill_words), 1))
                    
                    # Title bonus (weighted more heavily)
                    title_bonus = 0.0
                    if any(word in title for word in skill_words):
                        title_bonus = 0.25
                        # Extra bonus if important keywords appear at the beginning
                        if any(title.startswith(word) for word in skill_words[:2]):
                            title_bonus = 0.35
                    
                    # Headline-Bonus
                    headline_bonus = 0.0
                    if any(word in headline for word in skill_words):
                        headline_bonus = 0.15
                    
                    # Favorite-Bonus
                    favorite_bonus = 0.1 if is_favorite else 0.0
                    
                    # Essential-Bonus
                    essential_bonus = 0.05 if is_essential else 0.0
                    
                    # Calculate the final score for this skill
                    skill_score = min(1.0, base_similarity + keyword_bonus + title_bonus + headline_bonus + favorite_bonus + essential_bonus)
                    
                    # Initialize the course entry if it does not exist
                    if course_id not in course_matches:
                        course_matches[course_id] = {
                            'course': course,
                            'matched_skills': [],
                            'total_score': 0.0,
                            'skill_scores': {}
                        }
                    
                    # Add Skill Match
                    course_matches[course_id]['matched_skills'].append({
                        'skill_name': skill_name,
                        'skill_uri': skill_info['skill_uri'],
                        'is_favorite': is_favorite,
                        'is_essential': is_essential,
                        'score': skill_score
                    })
                    course_matches[course_id]['skill_scores'][skill_name] = skill_score
                    
                    # Multi-Skill Bonus: Courses that cover multiple skills receive a bonus
                    num_matched_skills = len(course_matches[course_id]['matched_skills'])
                    if num_matched_skills > 1:
                        multi_skill_bonus = min(0.2, (num_matched_skills - 1) * 0.1)
                        course_matches[course_id]['total_score'] = max(
                            course_matches[course_id]['total_score'],
                            skill_score + multi_skill_bonus
                        )
                    else:
                        course_matches[course_id]['total_score'] = max(
                            course_matches[course_id]['total_score'],
                            skill_score
                        )
        
        # Convert to Recommendations format
        recommendations = []
        for course_id, match_data in course_matches.items():
            course = match_data['course']
            matched_skills = match_data['matched_skills']
            total_score = match_data['total_score']
            
            # Create an entry for each matched skill
            for skill_match in matched_skills:
                recommendations.append({
                    'skill': skill_match['skill_name'],
                    'skill_uri': skill_match['skill_uri'],
                    'is_favorite': skill_match['is_favorite'],
                    'course_title': course.get('Title', 'N/A'),
                    'course_headline': course.get('Headline', 'N/A'),
                    'course_description': str(course.get('Description', ''))[:200] + '...' if len(str(course.get('Description', ''))) > 200 else course.get('Description', ''),
                    'course_url': course.get('URL', ''),
                    'course_price': course.get('Price', 'N/A'),
                    'course_language': course.get('Language', 'N/A'),
                    'similarity_score': total_score,  # Use `total_score` for multi-skill courses
                    'skill_score': skill_match['score'],  # Score for this specific skill
                    'num_matched_skills': len(matched_skills),  # Number of skills covered
                    'matched_skills_list': [s['skill_name'] for s in matched_skills]  # List of all covered skills
                })
        
        # Sort by total_score (multi-skill courses first)
        recommendations.sort(key=lambda x: (x.get('num_matched_skills', 1), x.get('similarity_score', 0)), reverse=True)
        
        # Filter by minimum score (higher threshold)
        filtered_recommendations = [r for r in recommendations if r.get('similarity_score', 0) > 0.1]
        
        # Group by course (remove duplicates) and keep the best version
        course_dict = {}
        for rec in filtered_recommendations:
            course_title = rec['course_title']
            if course_title not in course_dict or rec['similarity_score'] > course_dict[course_title]['similarity_score']:
                course_dict[course_title] = rec
        
        # Sort by score
        final_recommendations = sorted(course_dict.values(), key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return final_recommendations[:top_k * len(skill_info_list)]  # Return more courses for multi-skill matching
        
    except Exception as e:
        st.error(f"Fehler in find_udemy_courses_for_skills: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

@st.cache_data
def create_isco_esco_mapping(occupations_df):
    """Creates a mapping table between ISCO groups and ESCO URIs"""
    if occupations_df.empty:
        return {}
    
    mapping = {}
    
    for _, row in occupations_df.iterrows():
        isco_group = str(row.get('iscoGroup', ''))
        concept_uri = str(row.get('conceptUri', ''))
        preferred_label = str(row.get('preferredLabel', ''))
        
        if isco_group and concept_uri and not pd.isna(isco_group) and not pd.isna(concept_uri):
            # Extract the UUID from the URI
            uuid = concept_uri.split('/')[-1]
            
            # Create a mapping for the ISCO group
            if isco_group not in mapping:
                mapping[isco_group] = []
            
            mapping[isco_group].append({
                'uuid': uuid,
                'uri': concept_uri,
                'label': preferred_label
            })
    
    return mapping

@st.cache_data
def create_skill_mapping_with_english(skills_df, skills_en_df):
    """Creates a mapping between German and English skills based on conceptUri"""
    if skills_df.empty or skills_en_df.empty:
        return {}
    
    skill_mapping = {}
    
    # Create a mapping from conceptUri to German and English labels
    for _, skill_row in skills_df.iterrows():
        concept_uri = str(skill_row.get('conceptUri', ''))
        german_label = str(skill_row.get('preferredLabel', ''))
        
        if concept_uri and german_label and not pd.isna(concept_uri) and not pd.isna(german_label):
            # Search for the English equivalent
            english_match = skills_en_df[skills_en_df['conceptUri'] == concept_uri]
            
            if not english_match.empty:
                english_label = str(english_match.iloc[0].get('preferredLabel', ''))
                if english_label and not pd.isna(english_label):
                    skill_mapping[concept_uri] = {
                        'german': german_label,
                        'english': english_label
                    }
    
    return skill_mapping

@st.cache_data
def create_kldb_isco_mapping(kldb_esco_df):
    """Creates a mapping table between ESCO codes and ISCO groups"""
    if kldb_esco_df.empty:
        return {}
    
    mapping = {}
    
    # Mapping between ESCO codes and ISCO groups based on the structure
    # C0110 -> 0110 (Militärberufe)
    # C0210 -> 0210 (Unteroffiziere)
    # etc.
    
    for _, row in kldb_esco_df.iterrows():
        esco_code = str(row.get('ESCO_Code', ''))
        esco_label = str(row.get('ESCO_Label', ''))
        
        if esco_code and not pd.isna(esco_code):
            # Extract the numeric component from the ESCO code
            if esco_code.startswith('C'):
                isco_group = esco_code[1:]  # Remove the ‘C’ and keep the rest
                mapping[esco_code] = isco_group
    
    return mapping

@st.cache_data
def get_all_available_esco_skills(skills_df, skills_en_df=None):
    """Loads all available ESCO skills for selection from a dropdown menu"""
    if skills_df.empty:
        return []
    
    available_skills = []
    
    for _, skill_row in skills_df.iterrows():
        skill_uri = str(skill_row.get('conceptUri', ''))
        german_label = str(skill_row.get('preferredLabel', ''))
        
        if skill_uri and german_label and not pd.isna(skill_uri) and not pd.isna(german_label):
            skill_info = {
                'uri': skill_uri,
                'german_label': german_label,
                'english_label': german_label,  # Fallback
                'display_label': german_label
            }
            
            # Add English labels, if available
            if skills_en_df is not None and not skills_en_df.empty:
                english_match = skills_en_df[skills_en_df['conceptUri'] == skill_uri]
                if not english_match.empty:
                    english_label = str(english_match.iloc[0].get('preferredLabel', ''))
                    if english_label and not pd.isna(english_label):
                        skill_info['english_label'] = english_label
                        skill_info['display_label'] = f"{german_label} | {english_label}"
            
            available_skills.append(skill_info)
    
    # Sort by German label
    available_skills.sort(key=lambda x: x['german_label'])
    
    return available_skills

@st.cache_data
def parse_archi_xml(xml_file_path):
    """Parses XML files from Archi and extracts capabilities and their associated skills"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Namespace Mapping for Archimate
        namespaces = {
            'archimate': 'http://www.opengroup.org/xsd/archimate/3.0/',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
        
        capabilities = []
        resources = []
        
        # Extract all capabilities with the correct namespace
        capability_elements = root.findall('.//archimate:element[@xsi:type="Capability"]', namespaces)
        
        # Process the identified capabilities
        for capability in capability_elements:
            cap_id = capability.get('identifier')
            cap_name = capability.find('archimate:name', namespaces)
            if cap_name is not None and cap_name.text:
                capabilities.append({
                    'id': cap_id,
                    'name': cap_name.text,
                    'type': 'Capability'
                })
        
        # Extract all resources (skills) with the correct namespace
        resource_elements = root.findall('.//archimate:element[@xsi:type="Resource"]', namespaces)
        
        # Process found resources
        for resource in resource_elements:
            res_id = resource.get('identifier')
            res_name = resource.find('archimate:name', namespaces)
            if res_name is not None and res_name.text:
                resources.append({
                    'id': res_id,
                    'name': res_name.text,
                    'type': 'Resource'
                })
        
        # Identify relationships between capabilities and resources
        relationships = []
        relationship_elements = root.findall('.//archimate:relationship', namespaces)
        
        for relation in relationship_elements:
            source = relation.get('source')
            target = relation.get('target')
            rel_type = relation.get('{http://www.w3.org/2001/XMLSchema-instance}type')
            
            if source and target and rel_type:
                relationships.append({
                    'source': source,
                    'target': target,
                    'type': rel_type
                })
        
        # Fallback: If no elements were found, try alternative strategies
        if not capabilities and not resources:
            st.write("Keine Elemente mit Namespace gefunden, versuche alternative Strategien...")
            
            # Alternative 1: Search for all elements with xsi:type
            for element in root.findall('.//*'):
                xsi_type = element.get('{http://www.w3.org/2001/XMLSchema-instance}type')
                if xsi_type == 'Capability':
                    cap_id = element.get('identifier')
                    cap_name = element.find('name')
                    if cap_name is not None and cap_name.text:
                        capabilities.append({
                            'id': cap_id,
                            'name': cap_name.text,
                            'type': 'Capability'
                        })
                elif xsi_type == 'Resource':
                    res_id = element.get('identifier')
                    res_name = element.find('name')
                    if res_name is not None and res_name.text:
                        resources.append({
                            'id': res_id,
                            'name': res_name.text,
                            'type': 'Resource'
                        })
        
        return {
            'capabilities': capabilities,
            'resources': resources,
            'relationships': relationships
        }
        
    except Exception as e:
        st.error(f"Fehler beim Parsen der XML-Datei: {str(e)}")
        st.write("**Debug-Informationen:**")
        st.write(f"• Fehlertyp: {type(e).__name__}")
        st.write(f"• Fehlermeldung: {str(e)}")
        st.write("**Versuche alternative Parsing-Strategie...**")
        
        # Alternative: Simple parsing without namespace handling
        try:
            # Load XML as text and search for keywords
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Simple text-based extraction
            capabilities = []
            resources = []
            
            # Search for capability elements
            import re
            
            # Various patterns for capabilities
            cap_patterns = [
                r'<element[^>]*xsi:type="Capability"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<element[^>]*type="Capability"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<capability[^>]*>.*?<name[^>]*>(.*?)</name>'
            ]
            
            for pattern in cap_patterns:
                cap_matches = re.findall(pattern, xml_content, re.DOTALL)
                for match in cap_matches:
                    clean_name = match.strip()
                    if clean_name and clean_name not in [cap['name'] for cap in capabilities]:
                        capabilities.append({
                            'id': f"cap_{len(capabilities)}",
                            'name': clean_name,
                            'type': 'Capability'
                        })
            
            # Various patterns for resources
            res_patterns = [
                r'<element[^>]*xsi:type="Resource"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<element[^>]*type="Resource"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<resource[^>]*>.*?<name[^>]*>(.*?)</name>'
            ]
            
            for pattern in res_patterns:
                res_matches = re.findall(pattern, xml_content, re.DOTALL)
                for match in res_matches:
                    clean_name = match.strip()
                    if clean_name and clean_name not in [res['name'] for res in resources]:
                        resources.append({
                            'id': f"res_{len(resources)}",
                            'name': clean_name,
                            'type': 'Resource'
                        })
            
            st.success(f"Alternative Parsing-Strategie erfolgreich: {len(capabilities)} Capabilities, {len(resources)} Resources gefunden")
            
            return {
                'capabilities': capabilities,
                'resources': resources,
                'relationships': []
            }
            
        except Exception as e2:
            st.error(f"Auch alternative Parsing-Strategie fehlgeschlagen: {str(e2)}")
            return None

@st.cache_data
def extract_future_skills_from_capabilities(archi_data):
    """Extract the skills needed in the future from the capabilities in the XML"""
    if not archi_data:
        return []
    
    future_skills = []
    
    # Gather all resources (skills) from the capabilities
    for resource in archi_data['resources']:
        skill_name = resource['name']
        
        # Check whether the skill is linked to a capability
        is_capability_skill = False
        for relation in archi_data['relationships']:
            if (relation['source'] == resource['id'] or relation['target'] == resource['id']) and \
               any(rel['type'] in ['Composition', 'Aggregation', 'Realization'] for rel in [relation]):
                is_capability_skill = True
                break
        
        if is_capability_skill:
            future_skills.append({
                'skill_name': skill_name,
                'source': 'Capability',
                'type': 'Future Skill'
            })
    
    return future_skills

@st.cache_data
def parse_kompetenzabgleich_xml(xml_file_path):
    """Parses the Kompetenzabgleich.xml file and extracts current roles and future skills"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        ist_rollen = []  # BusinessActor (current Role)
        soll_skills = []  # Capability (future Skills)
        
        # Strategy 1: Search for all elements and check the xsi:type
        for element in root.findall('.//element'):
            element_type = element.get('xsi:type', '')
            name_elem = element.find('name')
            
            if name_elem is not None and name_elem.text:
                name = name_elem.text
                identifier = element.get('identifier')
                
                if 'BusinessActor' in element_type:
                    ist_rollen.append({
                        'identifier': identifier,
                        'name': name,
                        'type': 'BusinessActor'
                    })
                elif 'Capability' in element_type:
                    soll_skills.append({
                        'identifier': identifier,
                        'name': name,
                        'type': 'Capability'
                    })
        
        # Strategy 2: Fallback - Search for elements with specific tags
        if not ist_rollen and not soll_skills:
            # Search for BusinessActor
            for element in root.findall('.//element'):
                if 'BusinessActor' in str(element.attrib):
                    name_elem = element.find('name')
                    if name_elem is not None and name_elem.text:
                        ist_rollen.append({
                            'identifier': element.get('identifier', ''),
                            'name': name_elem.text,
                            'type': 'BusinessActor'
                        })
            
            # Search for Capability
            for element in root.findall('.//element'):
                if 'Capability' in str(element.attrib):
                    name_elem = element.find('name')
                    if name_elem is not None and name_elem.text:
                        soll_skills.append({
                            'identifier': element.get('identifier', ''),
                            'name': name_elem.text,
                            'type': 'Capability'
                        })
        
        # Strategy 3: Regex-based search as a last resort
        if not ist_rollen and not soll_skills:
            import re
            
            # Read the entire XML content
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Search for BusinessActor elements
            business_actor_pattern = r'<element[^>]*xsi:type="BusinessActor"[^>]*>.*?<name[^>]*>([^<]+)</name>'
            business_actor_matches = re.findall(business_actor_pattern, xml_content, re.DOTALL)
            
            for i, name in enumerate(business_actor_matches):
                ist_rollen.append({
                    'identifier': f'auto_gen_{i}',
                    'name': name.strip(),
                    'type': 'BusinessActor'
                })
            
            # Search for capability elements
            capability_pattern = r'<element[^>]*xsi:type="Capability"[^>]*>.*?<name[^>]*>([^<]+)</name>'
            capability_matches = re.findall(capability_pattern, xml_content, re.DOTALL)
            
            for i, name in enumerate(capability_matches):
                soll_skills.append({
                    'identifier': f'auto_gen_{i}',
                    'name': name.strip(),
                    'type': 'Capability'
                })
        
        # Debug-Information
        
        return {
            'ist_rollen': ist_rollen,
            'soll_skills': soll_skills,
            'success': True
        }
        
    except Exception as e:
        st.error(f"Fehler beim Parsen der Kompetenzabgleich.xml: {str(e)}")
        st.info("Versuche alternative Parsing-Strategien...")
        
        try:
            # Alternative Strategy: Simple Text-Based Parsing
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            ist_rollen = []
            soll_skills = []
            
            # Search for BusinessActor
            if 'BusinessActor' in xml_content:
                lines = xml_content.split('\n')
                for i, line in enumerate(lines):
                    if 'BusinessActor' in line and i + 1 < len(lines):
                        # Search for the next name element
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '<name' in lines[j] and '</name>' in lines[j]:
                                name = lines[j].split('<name')[1].split('</name>')[0].split('>', 1)[1]
                                if name.strip():
                                    ist_rollen.append({
                                        'identifier': f'fallback_{len(ist_rollen)}',
                                        'name': name.strip(),
                                        'type': 'BusinessActor'
                                    })
                                break
            
            # Search for Capability
            if 'Capability' in xml_content:
                lines = xml_content.split('\n')
                for i, line in enumerate(lines):
                    if 'Capability' in line and i + 1 < len(lines):
                        # Search for the next name element
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '<name' in lines[j] and '</name>' in lines[j]:
                                name = lines[j].split('<name')[1].split('</name>')[0].split('>', 1)[1]
                                if name.strip():
                                    soll_skills.append({
                                        'identifier': f'fallback_{len(soll_skills)}',
                                        'name': name.strip(),
                                        'type': 'Capability'
                                    })
                                break
            
            st.success(f"Fallback-Parsing erfolgreich: {len(ist_rollen)} IST-Rollen, {len(soll_skills)} SOLL-Skills")
            
            return {
                'ist_rollen': ist_rollen,
                'soll_skills': soll_skills,
                'success': True
            }
            
        except Exception as fallback_error:
            st.error(f"Auch Fallback-Parsing fehlgeschlagen: {str(fallback_error)}")
            return {
                'ist_rollen': [],
                'soll_skills': [],
                'success': False,
                'error': f"Original: {str(e)}, Fallback: {str(fallback_error)}"
            }

FUTURE_PROFILE_CACHE_KEY = "future_profiles_v3"

@st.cache_data
def parse_ist_soll_xml(xml_file_path, cache_buster=FUTURE_PROFILE_CACHE_KEY):
    """Parses Kompetenzabgleich_neuV1.xml and extracts current employees with business roles and future employees with capabilities"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Namespace-Mapping
        namespaces = {
            'archimate': 'http://www.opengroup.org/xsd/archimate/3.0/',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
        
        # Dictionary for all elements (identifier -> name)
        elements = {}
        alle_business_actors = []  # Gather all business actors first
        business_roles = {}  # BusinessRole identifier -> name
        resources = {}  # Resource identifier -> name
        groupings = {}  # Grouping identifier -> name (for future Skills)
        
        # 1. Extract all elements
        # Try different namespace variations
        elements_list = []
        
        # Option 1: Using Archimate: Namespace
        try:
            elements_list = root.findall('.//archimate:element', namespaces)
        except:
            pass
        
        # Option 2: Without a namespace
        if not elements_list:
            elements_list = root.findall('.//element')
        
        # Variante 3: Mit allen Namespaces (falls nsmap verfügbar)
        if not elements_list:
            try:
                if hasattr(root, 'nsmap') and root.nsmap:
                    for prefix, uri in root.nsmap.items():
                        try:
                            ns = {prefix: uri}
                            elements_list = root.findall(f'.//{{{uri}}}element', ns)
                            if elements_list:
                                break
                        except:
                            continue
            except:
                pass
        
        for element in elements_list:
            element_id = element.get('identifier')
            if not element_id:
                continue
                
            # Try different ways to find the type
            element_type = ''
            if '{http://www.w3.org/2001/XMLSchema-instance}type' in element.attrib:
                element_type = element.get('{http://www.w3.org/2001/XMLSchema-instance}type', '')
            elif 'xsi:type' in element.attrib:
                element_type = element.get('xsi:type', '')
            elif 'type' in element.attrib:
                element_type = element.get('type', '')
            
            # Try different ways to find the name
            name_elem = None
            try:
                name_elem = element.find('archimate:name', namespaces)
            except:
                pass
            
            if name_elem is None:
                name_elem = element.find('name')
            
            if name_elem is not None and name_elem.text:
                name = name_elem.text.strip()
                elements[element_id] = {
                    'name': name,
                    'type': element_type
                }
                
                # Categorize items
                if 'BusinessActor' in element_type:
                    # Filter out "Client", "Customer", and "Municipality"
                    # Also filter out dummy employees such as "Employee A", "Employee B", etc.
                    name_lower = name.lower()
                    is_dummy = False
                    future_actor_whitelist = {
                        f"mitarbeiter/in {letter}" for letter in ['a', 'b', 'c', 'd', 'e']
                    }
                    
                    # Check for dummy patterns: "Employee A", "Employee B", etc.
                    dummy_patterns = [
                        r'^mitarbeiter[\/\s]*in?\s+[a-z]$',  # "Mitarbeiter/in A", "Mitarbeiter A"
                        r'^mitarbeiter[\/\s]*in?\s+[a-z]\s*$',  # With spaces
                        r'^mitarbeiter[\/\s]*in?\s+[a-z]\d*$',  # With numbers
                    ]
                    
                    for pattern in dummy_patterns:
                        if re.match(pattern, name_lower):
                            is_dummy = True
                            break
                    
                    # Additional check: The name consists only of "Employee" + a single letter
                    if not is_dummy:
                        name_parts = name_lower.replace('/', ' ').replace('in', '').split()
                        if len(name_parts) == 2 and name_parts[0] == 'mitarbeiter' and len(name_parts[1]) == 1:
                            is_dummy = True
                    
                    if name_lower in future_actor_whitelist:
                        is_dummy = False
                    
                    if name_lower not in ['kunde', 'customer', 'kommune'] and not is_dummy:
                        alle_business_actors.append({
                            'identifier': element_id,
                            'name': name
                        })
                elif 'BusinessRole' in element_type:
                    business_roles[element_id] = name
                elif 'Resource' in element_type:
                    cleaned_name = clean_skill_label(name)
                    if cleaned_name:
                        resources[element_id] = cleaned_name
                elif 'Grouping' in element_type:
                    groupings[element_id] = name
        
        # 2. Extract relationships
        assignments = []  # Assignment Relationships (BusinessActor -> BusinessRole)
        associations = []  # Association Relationships (BusinessActor -> Resource or BusinessActor -> Grouping)
        compositions = []  # Composition Relationships (Grouping -> Resource)
        
        relationships_list = []
        
        # Try different ways to find relationships
        try:
            relationships_list = root.findall('.//archimate:relationship', namespaces)
        except:
            pass
        
        if not relationships_list:
            relationships_list = root.findall('.//relationship')
        
        for relationship in relationships_list:
            source = relationship.get('source')
            target = relationship.get('target')
            
            if not source or not target:
                continue
            
            # Try different ways to find the type
            rel_type = ''
            if '{http://www.w3.org/2001/XMLSchema-instance}type' in relationship.attrib:
                rel_type = relationship.get('{http://www.w3.org/2001/XMLSchema-instance}type', '')
            elif 'xsi:type' in relationship.attrib:
                rel_type = relationship.get('xsi:type', '')
            elif 'type' in relationship.attrib:
                rel_type = relationship.get('type', '')
            
            if 'Assignment' in rel_type:
                assignments.append({
                    'source': source,
                    'target': target
                })
            elif 'Association' in rel_type:
                associations.append({
                    'source': source,
                    'target': target
                })
            elif 'Composition' in rel_type:
                compositions.append({
                    'source': source,
                    'target': target
                })
        
        # 3. Categorize business actors in the current state based on their relationships
        ist_mitarbeiter = []
        future_actor_whitelist = {
            f"mitarbeiter/in {letter}" for letter in ['a', 'b', 'c', 'd', 'e']
        }
        
        for actor in alle_business_actors:
            actor_id = actor['identifier']
            actor_name_lower = actor['name'].lower()
            
            # Check assignment relationships to business roles
            hat_assignment_zu_role = False
            for assignment in assignments:
                if assignment['source'] == actor_id and assignment['target'] in business_roles:
                    hat_assignment_zu_role = True
                    break
            
            # Current employee = has an assignment to a business role (has a job title)
            if hat_assignment_zu_role and actor_name_lower not in future_actor_whitelist:
                ist_mitarbeiter.append(actor)
            # All other business actors without a job title are ignored
            # (Future skills come directly from groupings, not from business actors)
        
        # 4. Link current employees to business roles (with deduplication by name)
        ist_mitarbeiter_mit_rollen_dict = {}  # name -> {name, identifiers, business_roles}
        
        for mitarbeiter in ist_mitarbeiter:
            mitarbeiter_id = mitarbeiter['identifier']
            mitarbeiter_name = mitarbeiter['name']
            zugeordnete_rollen = []
            
            # Find assignment relationships for this employee
            for assignment in assignments:
                if assignment['source'] == mitarbeiter_id:
                    role_id = assignment['target']
                    if role_id in business_roles:
                        role_name = business_roles[role_id]
                        # Avoid duplicate roles
                        if role_name not in zugeordnete_rollen:
                            zugeordnete_rollen.append(role_name)
            
            # Deduplicate by name - collect all roles and identifiers for each employee
            if mitarbeiter_name not in ist_mitarbeiter_mit_rollen_dict:
                ist_mitarbeiter_mit_rollen_dict[mitarbeiter_name] = {
                    'name': mitarbeiter_name,
                    'identifiers': [mitarbeiter_id],
                    'business_roles': zugeordnete_rollen.copy()
                }
            else:
                # The employee already exists - add roles and an identifier
                existing = ist_mitarbeiter_mit_rollen_dict[mitarbeiter_name]
                if mitarbeiter_id not in existing['identifiers']:
                    existing['identifiers'].append(mitarbeiter_id)
                # Add new roles (no duplicates)
                for role in zugeordnete_rollen:
                    if role not in existing['business_roles']:
                        existing['business_roles'].append(role)
        
        # Convert Dictionary back to List
        ist_mitarbeiter_mit_rollen = list(ist_mitarbeiter_mit_rollen_dict.values())
        
        # 5. Extract future capabilities from groupings (for the new corporate strategy)
        # Create a mapping: Grouping -> Resources (via Composition)
        grouping_resources = {}
        for composition in compositions:
            grouping_id = composition['source']
            resource_id = composition['target']
            if grouping_id in groupings and resource_id in resources:
                if grouping_id not in grouping_resources:
                    grouping_resources[grouping_id] = []
                grouping_resources[grouping_id].append(resources[resource_id])
        
        # Collect all skills from all groupings (especially those with "Fähigkeit" in the name)
        soll_faehigkeiten = set()
        for grouping_id, grouping_name in groupings.items():
            # Focus on groupings that include "Fähigkeit" or "Resource" in their names 
            if 'fähigkeit' in grouping_name.lower() or 'resource' in grouping_name.lower():
                if grouping_id in grouping_resources:
                    for resource in grouping_resources[grouping_id]:
                        soll_faehigkeiten.add(resource)
        
        # If no specific skill (Fähigkeit) groupings are found, use all resources from the groupings
        if not soll_faehigkeiten:
            for grouping_id in grouping_resources:
                for resource in grouping_resources[grouping_id]:
                    soll_faehigkeiten.add(resource)
        
        future_profiles = extract_future_profiles(alle_business_actors, associations, groupings, grouping_resources)
        if future_profiles:
            aggregated_skills = []
            for profile in future_profiles:
                aggregated_skills.extend(profile.get('skills', []))
            soll_faehigkeiten_liste = aggregated_skills
            soll_total_skill_count = sum(len(profile.get('skills', [])) for profile in future_profiles)
        else:
            soll_faehigkeiten_liste = sorted(list(soll_faehigkeiten))
            soll_total_skill_count = len(soll_faehigkeiten_liste)
        
        # Debug-Information
        debug_info = {
            'total_elements': len(elements),
            'alle_business_actors_count': len(alle_business_actors),
            'ist_mitarbeiter_count': len(ist_mitarbeiter),
            'business_roles_count': len(business_roles),
            'resources_count': len(resources),
            'groupings_count': len(groupings),
            'assignments_count': len(assignments),
            'associations_count': len(associations),
            'compositions_count': len(compositions),
            'ist_mit_rollen_count': len(ist_mitarbeiter_mit_rollen),
            'soll_faehigkeiten_count': len(soll_faehigkeiten_liste),
            'soll_total_skill_count': soll_total_skill_count
        }
        
        return {
            'ist_mitarbeiter': ist_mitarbeiter_mit_rollen,
            'soll_faehigkeiten': soll_faehigkeiten_liste,
            'success': True,
            'debug': debug_info,
            'soll_profile_map': future_profiles,
            'soll_total_skill_count': soll_total_skill_count
        }
        
    except Exception as e:
        import traceback
        return {
            'ist_mitarbeiter': [],
            'soll_faehigkeiten': [],
            'soll_profile_map': [],
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@st.cache_data
def find_kldb_code_for_job_title(job_title, occupations_df, kldb_esco_df, berufsbenennungen_df=None):
    """Findet den passenden KldB-Code für eine Jobbezeichnung"""
    if not job_title:
        return None, None
    
    # 1. Try searching via the alphabetical index
    kldb_code_from_lookup, lookup_label = match_job_in_berufsverzeichnis(job_title, berufsbenennungen_df)
    if kldb_code_from_lookup:
        kldb_label, _, _ = get_kldb_mapping_for_code(kldb_code_from_lookup, kldb_esco_df, preferred_label=lookup_label)
        return kldb_code_from_lookup, kldb_label or lookup_label
    
    if occupations_df.empty:
        return None, None
    
    # Standardize the job title
    normalized_job = job_title.lower().strip()
    
    # Search for exact matches in the names
    for _, row in occupations_df.iterrows():
        preferred_label = str(row.get('preferredLabel', '')).lower()
        if normalized_job in preferred_label or preferred_label in normalized_job:
            # Find the corresponding KldB code
            concept_uri = str(row.get('conceptUri', ''))
            preferred_label = str(row.get('preferredLabel', ''))
            if concept_uri and preferred_label:
                # First, try direct label matching using KldB-ESCO mapping
                for _, kldb_row in kldb_esco_df.iterrows():
                    esco_label = str(kldb_row.get('ESCO_Label', '')).lower()
                    if preferred_label.lower() == esco_label or preferred_label.lower() in esco_label:
                        kldb_code = kldb_row.get('KldB_Code', '')
                        kldb_label = kldb_row.get('KldB_Label', '')
                        if kldb_code:
                            return kldb_code, kldb_label
    
    # Fallback: Search for similar terms
    best_match = None
    best_score = 0
    
    for _, row in occupations_df.iterrows():
        preferred_label = str(row.get('preferredLabel', '')).lower()
        
        # Simple similarity calculation
        common_words = set(normalized_job.split()) & set(preferred_label.split())
        if common_words:
            score = len(common_words) / max(len(normalized_job.split()), len(preferred_label.split()))
            if score > best_score and score > 0.3:  # Minimum similarity
                best_score = score
                best_match = row
    
    if best_match is not None:
        preferred_label = str(best_match.get('preferredLabel', ''))
        if preferred_label:
            # Try label matching using KldB-ESCO mapping
            for _, kldb_row in kldb_esco_df.iterrows():
                esco_label = str(kldb_row.get('ESCO_Label', '')).lower()
                if preferred_label.lower() in esco_label or esco_label in preferred_label.lower():
                    kldb_code = kldb_row.get('KldB_Code', '')
                    kldb_label = kldb_row.get('KldB_Label', '')
                    if kldb_code:
                        return kldb_code, kldb_label
    
    return None, None

@st.cache_data
def find_kldb_code_for_business_role_semantic(business_role, kldb_esco_df, berufsbenennungen_df=None, min_similarity=0.2):
    """Find the appropriate KldB code for a business role using improved semantic similarity"""
    if not business_role or kldb_esco_df.empty:
        return None, None, None, None
    
    role_norm = normalize_job_label(business_role)
    if role_norm:
        alias_match = kldb_esco_df[
            kldb_esco_df['KldB_Label'].apply(normalize_job_label) == role_norm
        ]
        if not alias_match.empty:
            row = alias_match.iloc[0]
            return row['KldB_Code'], row['KldB_Label'], row['ESCO_Label'], row['ESCO_Code']
    
    # 1. Try searching the alphabetical directory
    lookup_code, lookup_label = match_job_in_berufsverzeichnis(business_role, berufsbenennungen_df)
    if lookup_code:
        kldb_label, esco_label, esco_code = get_kldb_mapping_for_code(lookup_code, kldb_esco_df, preferred_label=business_role)
        return lookup_code, kldb_label or lookup_label, esco_label, esco_code
    
    # Normalize the business role, remove "/in", parentheses, etc.
    normalized_role = business_role.strip()
    
    # Remove parentheses and their contents (but save the contents for later use)
    bracket_content = re.findall(r'\(([^)]+)\)', normalized_role)
    normalized_role_no_brackets = re.sub(r'\([^)]*\)', '', normalized_role).strip()
    
    # Remove "/in" variants, but keep the main term
    # First, normalize "/in" to a space, then remove isolated "in"
    normalized_role_clean = normalized_role_no_brackets.replace('/in', ' ').replace('(in)', ' ').replace(' / ', ' ').replace('/', ' ').strip()
    # Remove the isolated "in" at the end or as a separate word
    normalized_role_clean = re.sub(r'\bin\b', '', normalized_role_clean, flags=re.IGNORECASE).strip()
    # Remove hyphens and replace them with spaces
    normalized_role_clean = normalized_role_clean.replace('-', ' ').replace('–', ' ').replace(',', ' ').strip()
    # Remove extra spaces
    normalized_role_clean = re.sub(r'\s+', ' ', normalized_role_clean).strip()
    
    # Extract keywords (words with at least 4 characters to be specific)
    # But also shorter important words like "Ing" (for Ingenieur)
    role_keywords = []
    for w in normalized_role_clean.split():
        w_lower = w.lower()
        if len(w) >= 4:  # Standard keywords
            role_keywords.append(w_lower)
        elif w_lower in ['ing', 'ing.', 'ing-']:  # Abbreviation for "Ingenieur"
            role_keywords.append('ingenieur')
        elif len(w) >= 3 and w_lower not in ['der', 'die', 'das', 'und', 'oder', 'mit', 'von', 'für']:
            role_keywords.append(w_lower)
    
    # Also extract key terms (the first 1-2 words are usually the key term)
    main_terms = normalized_role_clean.lower().split()[:2]
    main_term = ' '.join(main_terms) if main_terms else ''
    
    # 2. Try exact matching (case-insensitive)
    exact_match = kldb_esco_df[
        kldb_esco_df['KldB_Label'].str.strip().str.lower() == normalized_role.lower()
    ]
    
    if not exact_match.empty:
        row = exact_match.iloc[0]
        return row['KldB_Code'], row['KldB_Label'], row['ESCO_Label'], row['ESCO_Code']
    
    # 3. Try matching with the cleaned-up version (without parentheses, without /in)
    # Also normalize the KldB labels for comparison
    def normalize_label(label):
        if pd.isna(label):
            return ''
        label_str = str(label).lower()
        # Remove /in variants
        label_str = label_str.replace('/in', ' ').replace('(in)', ' ').replace('/', ' ')
        # Remove parentheses
        label_str = re.sub(r'\([^)]*\)', '', label_str)
        # Normalize spaces
        label_str = re.sub(r'\s+', ' ', label_str).strip()
        return label_str
    
    normalized_role_clean_lower = normalized_role_clean.lower()
    
    # Search for exact matches using normalized labels
    for idx, row in kldb_esco_df.iterrows():
        kldb_label_normalized = normalize_label(row.get('KldB_Label', ''))
        if kldb_label_normalized == normalized_role_clean_lower:
            return row['KldB_Code'], row['KldB_Label'], row['ESCO_Label'], row['ESCO_Code']
    
    # 4. Search by main term (first 1-2 words) - very important for cases such as "Betriebsingenieur"
    if main_term:
        main_term_matches = []
        for idx, row in kldb_esco_df.iterrows():
            kldb_label = str(row.get('KldB_Label', '')).lower()
            if pd.isna(kldb_label):
                continue
            
            # Normalize KldB label
            kldb_label_normalized = normalize_label(kldb_label)
            kldb_main_terms = kldb_label_normalized.split()[:2]
            kldb_main_term = ' '.join(kldb_main_terms) if kldb_main_terms else ''
            
            # Check if the main term matches
            if main_term in kldb_label_normalized or kldb_main_term in normalized_role_clean_lower:
                # Calculate Score
                score = 100
                # Bonus for an exact match of the main term
                if main_term == kldb_main_term:
                    score += 50
                # Bonus for shorter labels (more specific)
                score += 30 / max(len(kldb_label), 1)
                # Bonus points if all keywords are included
                if role_keywords:
                    matching_kw = sum(1 for kw in role_keywords if kw in kldb_label_normalized)
                    score += matching_kw * 20
                
                main_term_matches.append((score, row))
        
        if main_term_matches:
            main_term_matches.sort(key=lambda x: x[0], reverse=True)
            best_score, best_match = main_term_matches[0]
            if best_score >= 80:  # High threshold for main term matches
                return best_match['KldB_Code'], best_match['KldB_Label'], best_match['ESCO_Label'], best_match['ESCO_Code']
    
    # 5. Prioritized keyword search
    # Search for KldB labels that contain all important keywords (highest priority)
    if role_keywords:
        all_keywords_match = []
        
        for idx, row in kldb_esco_df.iterrows():
            kldb_label = str(row.get('KldB_Label', '')).lower()
            if pd.isna(kldb_label):
                continue
            
            kldb_label_normalized = normalize_label(kldb_label)
            
            # Check to see if all keywords are included
            matching_keywords = [kw for kw in role_keywords if kw in kldb_label_normalized]
            
            if len(matching_keywords) == len(role_keywords):  # All keywords found
                score = 200  # Very high base score
                
                # Bonus for exact word order
                role_words = normalized_role_clean_lower.split()
                kldb_words = kldb_label_normalized.split()
                
                # Check the word order
                role_idx = 0
                for kw in role_keywords:
                    for i, kldb_word in enumerate(kldb_words):
                        if kw in kldb_word and i >= role_idx:
                            score += 50 - i  # Bonus for early positioning
                            role_idx = i
                            break
                
                # Bonus for shorter labels (more specific)
                score += 50 / max(len(kldb_label), 1)
                
                all_keywords_match.append((score, row))
        
        if all_keywords_match:
            all_keywords_match.sort(key=lambda x: x[0], reverse=True)
            best_score, best_match = all_keywords_match[0]
            return best_match['KldB_Code'], best_match['KldB_Label'], best_match['ESCO_Label'], best_match['ESCO_Code']
    
    # 6. Search for tags that contain at least the most important keywords (with prioritization)
    if role_keywords:
        keyword_matches = []
        
        # Prioritize the first keywords (they are usually more important)
        important_keywords = role_keywords[:2] if len(role_keywords) >= 2 else role_keywords
        
        for idx, row in kldb_esco_df.iterrows():
            kldb_label = str(row.get('KldB_Label', '')).lower()
            if pd.isna(kldb_label):
                continue
            
            kldb_label_normalized = normalize_label(kldb_label)
            
            # Calculate score based on matching keywords
            matching_keywords = [kw for kw in role_keywords if kw in kldb_label_normalized]
            matching_important = [kw for kw in important_keywords if kw in kldb_label_normalized]
            
            if matching_keywords:
                # Prioritize by:
                # - Number of matching keywords
                # - Important keywords count more
                # - Position of the keywords (earlier = better)
                # - Length of the label (shorter = more specific)
                score = len(matching_keywords) * 15
                score += len(matching_important) * 25  # Bonus for important keywords
                
                # Bonus for early keyword placement
                kldb_words = kldb_label_normalized.split()
                for kw in matching_keywords:
                    for i, word in enumerate(kldb_words):
                        if kw in word:
                            score += 30 - i  # Bonus for early positioning
                            break
                
                # Bonus for shorter labels (more specific)
                score += 60 / max(len(kldb_label), 1)
                
                # Penalty points for very general terms (only if not all keywords match)
                if len(matching_keywords) < len(role_keywords):
                    general_terms = ['fachkraft', 'mitarbeiter', 'assistent', 'fachkräfte', 'fachkräfte für']
                    if any(term in kldb_label_normalized for term in general_terms):
                        score -= 10
                
                keyword_matches.append((score, row))
        
        if keyword_matches:
            # Sort by score (highest first)
            keyword_matches.sort(key=lambda x: x[0], reverse=True)
            best_score, best_match = keyword_matches[0]
            
            # Return only if the score is high enough
            if best_score >= 40:  # Minimum score for an acceptable match
                return best_match['KldB_Code'], best_match['KldB_Label'], best_match['ESCO_Label'], best_match['ESCO_Code']
    
    # 7. Fallback: Semantic matching using TF-IDF and cosine similarity
    kldb_labels = []
    kldb_indices = []
    
    for idx, row in kldb_esco_df.iterrows():
        kldb_label = str(row.get('KldB_Label', '')).strip()
        if kldb_label and kldb_label != 'nan':
            kldb_labels.append(kldb_label)
            kldb_indices.append(idx)
    
    if kldb_labels:
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 2))
            
            all_texts = kldb_labels + [normalized_role_clean]
            vectors = vectorizer.fit_transform(all_texts)
            
            kldb_vectors = vectors[:-1]
            role_vector = vectors[-1:]
            
            similarities = cosine_similarity(role_vector, kldb_vectors).flatten()
            
            # Find the top 5 matches and choose the best one based on additional criteria
            top_indices = similarities.argsort()[-5:][::-1]
            
            best_match = None
            best_combined_score = -1
            
            for idx in top_indices:
                if similarities[idx] >= min_similarity:
                    original_idx = kldb_indices[idx]
                    row = kldb_esco_df.iloc[original_idx]
                    kldb_label = str(row['KldB_Label']).lower()
                    kldb_label_normalized = normalize_label(kldb_label)
                    
                    # Combine the similarity score with additional criteria
                    combined_score = similarities[idx] * 100
                    
                    # Bonus for shorter labels (more specific)
                    combined_score += 20 / max(len(kldb_label), 1)
                    
                    # Bonus for matching keywords
                    if role_keywords:
                        matching = sum(1 for kw in role_keywords if kw in kldb_label_normalized)
                        combined_score += matching * 10
                    
                    # Bonus for main term
                    if main_term and main_term in kldb_label_normalized:
                        combined_score += 30
                    
                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        best_match = row
            
            if best_match is not None and best_combined_score >= 50:
                return best_match['KldB_Code'], best_match['KldB_Label'], best_match['ESCO_Label'], best_match['ESCO_Code']
        except Exception:
            pass
    
    return None, None, None, None

@st.cache_data
def find_best_job_matches_for_capabilities(capabilities, occupations_df, kldb_esco_df, top_k=5):
    """Find the best job title matches for capabilities (future skills)"""
    job_descriptions = []
    job_indices = []
    
    for idx, row in occupations_df.iterrows():
        preferred_label = str(row.get('preferredLabel', ''))
        if preferred_label and not pd.isna(preferred_label):
            job_descriptions.append(preferred_label)
            job_indices.append(idx)
    
    if not job_descriptions:
        return []
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    job_vectors = vectorizer.fit_transform(job_descriptions)
    
    all_matches = []
    
    for capability in capabilities:
        capability_name = capability.get('name', '')
        if not capability_name:
            continue
        
        # Vectorization Capability
        capability_vector = vectorizer.transform([capability_name])
        
        # Similarity calculation
        similarities = cosine_similarity(capability_vector, job_vectors).flatten()
        
        # Top-K Matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        capability_matches = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0.01:  # Minimum similarity
                job_idx = job_indices[idx]
                job_row = occupations_df.iloc[job_idx]
                
                # Find KldB-Code
                preferred_label = str(job_row.get('preferredLabel', ''))
                kldb_code = ''
                kldb_label = ''
                
                if preferred_label:
                    # Try label matching using KldB-ESCO mapping
                    for _, kldb_row in kldb_esco_df.iterrows():
                        esco_label = str(kldb_row.get('ESCO_Label', '')).lower()
                        if preferred_label.lower() in esco_label or esco_label in preferred_label.lower():
                            kldb_code = kldb_row.get('KldB_Code', '')
                            kldb_label = kldb_row.get('KldB_Label', '')
                            break
                
                capability_matches.append({
                    'capability': capability_name,
                    'job_title': job_row.get('preferredLabel', ''),
                    'kldb_code': kldb_code,
                    'kldb_label': kldb_label,
                    'esco_uri': job_row.get('conceptUri', ''),
                    'similarity_score': similarities[idx],
                    'rank': rank + 1
                })
        
        if capability_matches:
            all_matches.extend(capability_matches)
    
    # Sort by similarity score
    all_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return all_matches

def main():
    # Copyright footer at the bottom left - always visible
    st.markdown("""
    <style>
    .streamlit-footer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        padding: 8px 15px !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #666 !important;
        font-size: 0.75rem !important;
        z-index: 999999 !important;
        width: auto !important;
        border-top: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-top-right-radius: 4px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1) !important;
        pointer-events: auto !important;
    }
    /* Make sure the footer isn't hidden */
    footer { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # Footer as a separate element
    st.markdown('<div class="streamlit-footer">© Natascha Deuter</div>', unsafe_allow_html=True)
    
    st.title("Kompetenzabgleich & Weiterbildungsempfehlungen")
    st.markdown("---")
    
    # Load data
    with st.spinner("Lade Daten..."):
        (employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, 
         udemy_courses_df, occupations_df, occupation_skills_mapping, skills_en_df, archi_data, 
         kompetenzabgleich_data, berufsbenennungen_df) = load_data(cache_buster=LOAD_DATA_CACHE_VERSION)
    
    if employees_df.empty and kldb_esco_df.empty:
        st.error("Fehler beim Laden der Daten. Bitte überprüfe die CSV-Dateien.")
        return
    
    # Create a direct skills mapping
    st.session_state.occupation_skills_mapping = occupation_skills_mapping
    
    # Create the skill mapping with English equivalents
    skill_mapping_with_english = create_skill_mapping_with_english(skills_df, skills_en_df)
    st.session_state.skill_mapping_with_english = skill_mapping_with_english
    
    # Save Archi data in session state
    st.session_state.archi_data = archi_data
    
    # Store skill matching data in the session state
    st.session_state.kompetenzabgleich_data = kompetenzabgleich_data
    
    # Save job titles in session state
    st.session_state.berufsbenennungen_df = berufsbenennungen_df
    
    # Initialize session state for employee data if it does not exist
    if 'employees_data' not in st.session_state:
        # Add a "Name" column if it doesn't exist
        if 'Name' not in employees_df.columns:
            employees_df['Name'] = 'Unbekannt'
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
    # Make sure all the required columns are included
    required_columns = ['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']
    for col in required_columns:
        if col not in st.session_state.employees_data.columns:
            st.session_state.employees_data[col] = ''
    
    # Handle NaN values in the future role columns and new skill columns
    for col in ['Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']:
        st.session_state.employees_data[col] = st.session_state.employees_data[col].fillna('')
    
    # Load saved futue roles for the current employee
    if 'current_employee_id' in st.session_state:
        current_employee_id = st.session_state.current_employee_id
        employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == current_employee_id]
        
        if not employee_data.empty:
            employee_row = employee_data.iloc[0]
            target_kldb_code = employee_row.get('Target_KldB_Code', '')
            target_kldb_label = employee_row.get('Target_KldB_Label', '')
            target_esco_code = employee_row.get('Target_ESCO_Code', '')
            target_esco_label = employee_row.get('Target_ESCO_Label', '')
            
            # If there are any saved future roles, restore them
            if target_kldb_code and target_kldb_label and target_esco_code and target_esco_label:
                st.session_state.selected_target_role = {
                    'KldB_Code': target_kldb_code,
                    'KldB_Label': target_kldb_label,
                    'ESCO_Code': target_esco_code,
                    'ESCO_Label': target_esco_label
                }
                
                # Future role has been restored (do not display notification)
                if 'target_role_restored' not in st.session_state:
                    st.session_state.target_role_restored = True
    

    
    # Use updated employee data from session state
    employees_df = st.session_state.employees_data
    
    # Global employee selection in the sidebar
    st.sidebar.title("Navigation")
    
    # Staff Selection (global for all sections)
    if not employees_df.empty:
        st.sidebar.subheader("Mitarbeiter auswählen")
        employee_options = [f"{row['Employee_ID']} - {row.get('Name', 'Unbekannt')}" for _, row in employees_df.iterrows()]
        
        # Initialize selected_employee if it does not exist
        if 'selected_employee' not in st.session_state:
            st.session_state.selected_employee = employee_options[0] if employee_options else None
        
        # Use the saved selection or the first employee
        default_index = 0
        if st.session_state.selected_employee in employee_options:
            default_index = employee_options.index(st.session_state.selected_employee)
        
        selected_employee_str = st.sidebar.selectbox(
            "Mitarbeiter:", 
            employee_options, 
            key="global_employee_select",
            index=default_index
        )
        
        # Save the selection to session state
        st.session_state.selected_employee = selected_employee_str
        
        # Extract the employee ID for future use
        if selected_employee_str:
            new_employee_id = selected_employee_str.split(" - ")[0]
            
            # Check whether the employee has changed
            if 'current_employee_id' not in st.session_state or st.session_state.current_employee_id != new_employee_id:
                st.session_state.current_employee_id = new_employee_id
                
                # Reset notifications for new employees
                if 'target_role_restored' in st.session_state:
                    del st.session_state.target_role_restored
                
                # Load saved future roles for the new employee
                employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == new_employee_id]
                
                if not employee_data.empty:
                    employee_row = employee_data.iloc[0]
                    target_kldb_code = employee_row.get('Target_KldB_Code', '')
                    target_kldb_label = employee_row.get('Target_KldB_Label', '')
                    target_esco_code = employee_row.get('Target_ESCO_Code', '')
                    target_esco_label = employee_row.get('Target_ESCO_Label', '')
                    
                    # If there are any saved future roles, restore them
                    if target_kldb_code and target_kldb_label and target_esco_code and target_esco_label:
                        st.session_state.selected_target_role = {
                            'KldB_Code': target_kldb_code,
                            'KldB_Label': target_kldb_label,
                            'ESCO_Code': target_esco_code,
                            'ESCO_Label': target_esco_label
                        }
                    else:
                        # Delete the saved future role if none exists
                        if 'selected_target_role' in st.session_state:
                            del st.session_state.selected_target_role
    
    # Navigation
    page = st.sidebar.selectbox(
        "Wählen Sie eine Sektion:",
        ["Personalplanung mit Kursempfehlung", "Mitarbeiter-Kompetenzprofile", "Berufsabgleich", "Kursempfehlungen", "Strategische Weiterbildung ", "XML-basierte Kompetenzabgleich ", "Gesamtübersicht", "Mitarbeiter-Verwaltung"]
    )
    
    # Show the corresponding page
    if page == "Mitarbeiter-Kompetenzprofile":
        show_employee_profiles(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df)
    elif page == "Berufsabgleich":
        show_occupation_matching(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df)
    elif page == "Strategische Weiterbildung ":
        show_strategic_development(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, archi_data, udemy_courses_df)
    elif page == "XML-basierte Kompetenzabgleich ":
        show_xml_based_competency_analysis(
            employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df,
            eures_skills_df, occupations_df, skills_en_df, udemy_courses_df, berufsbenennungen_df
        )
    elif page == "Personalplanung mit Kursempfehlung":
        show_ist_soll_matching(
            employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df,
            eures_skills_df, occupations_df, skills_en_df, udemy_courses_df, berufsbenennungen_df
        )
    elif page == "Kursempfehlungen":
        show_course_recommendations(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df)
    elif page == "Gesamtübersicht":
        show_overview(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df)
    elif page == "Mitarbeiter-Verwaltung":
        show_employee_management(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df)

def show_employee_profiles(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df=None):
    st.header("Mitarbeiter-Kompetenzprofile")
    
    # Use updated employee data from session state
    employees_df = st.session_state.employees_data
    
    if employees_df.empty:
        st.warning("Keine Mitarbeiterdaten gefunden.")
        return
    
    # Use the global employee selection
    if 'current_employee_id' not in st.session_state:
        st.info("Bitte wählen Sie einen Mitarbeiter in der Sidebar aus.")
        return
    
    employee_id = st.session_state.current_employee_id
    employee_data = employees_df[employees_df['Employee_ID'] == employee_id].iloc[0]
    
    st.subheader(f"Profil von {employee_data.get('Name', employee_id)}")
    
    # Retrieve current data from session state (not cached)
    current_employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
    current_kldb = current_employee_data.get('KldB_5_digit', '')
    current_manual_skills = current_employee_data.get('Manual_Skills', '')
    current_esco_role = current_employee_data.get('ESCO_Role', '')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aktuelle Rolle:**")
        
        # Find the KldB role based on the code
        kldb_match = get_unique_esco_roles(kldb_esco_df, current_kldb)
        
        if not kldb_match.empty:
            kldb_label = kldb_match.iloc[0]['KldB_Label']
            esco_label = kldb_match.iloc[0]['ESCO_Label']
            
            st.write(f"• **KldB-Code:** {current_kldb}")
            st.write(f"• **Berufsbezeichnung:** {kldb_label}")
            
            # Show the ESCO category only if it differs from the KldB label
            if esco_label.lower() != kldb_label.lower():
                st.write(f"• **ESCO-Kategorie:** {esco_label}")
        else:
            st.write(f"• **KldB-Code:** {current_kldb}")
            st.write("• **Berufsbezeichnung:** Nicht gefunden")
    
    with col2:
        st.write("**Manuelle Skills:**")
        if current_manual_skills and pd.notna(current_manual_skills) and str(current_manual_skills).strip():
            for skill in str(current_manual_skills).split(';'):
                if skill.strip():  # Show only non-empty skills
                    st.write(f"• {skill.strip()}")
        else:
            st.write("Keine manuellen Skills")
    
    # Show selected future role, if available
    if 'selected_target_role' in st.session_state:
        st.markdown("---")
        st.subheader("Ausgewählte Zielrolle")
        
        target_role = st.session_state.selected_target_role
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Zielrolle:**")
            st.write(f"• **KldB-Code:** {target_role['KldB_Code']}")
            st.write(f"• **Berufsbezeichnung:** {target_role['KldB_Label']}")
        
        with col2:
            # Show the ESCO category only if it differs from the KldB label
            if target_role['ESCO_Label'].lower() != target_role['KldB_Label'].lower():
                st.write("**ESCO-Kategorie:**")
                st.write(f"• {target_role['ESCO_Label']}")
        
        # Button to reset the future role
        if st.button("Zielrolle zurücksetzen", key="reset_target_role_profile", type="secondary"):
            # Delete the saved future role from the employee data
            st.session_state.employees_data.loc[
                st.session_state.employees_data['Employee_ID'] == employee_id, 
                ['Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label']
            ] = ['', '', '', '']
            
            # Save as CSV
            if save_employees_to_csv(st.session_state.employees_data):
                # Delete from session state
                if 'selected_target_role' in st.session_state:
                    del st.session_state.selected_target_role
                st.success("Zielrolle wurde zurückgesetzt und gespeichert!")
            else:
                st.warning("Zielrolle zurückgesetzt, aber Speichern fehlgeschlagen!")
            
            st.rerun()
    
    st.markdown("---")
    
    # View Assigned Skills (after role assignment)
    st.subheader("Zugewiesene Skills")
    
    # Check whether a current role has been assigned
    if not current_kldb:
        st.info("**Keine aktuelle Rolle zugewiesen.** Bitte weisen Sie unten eine Rolle zu, um die zugehörigen Skills anzuzeigen.")
    else:
        # Retrieve additional skill data from session state
        current_manual_essential_skills = current_employee_data.get('Manual_Essential_Skills', '')
        current_manual_optional_skills = current_employee_data.get('Manual_Optional_Skills', '')
        current_removed_skills = current_employee_data.get('Removed_Skills', '')
        
        # Create the current profile
        profile = create_employee_profile(
            employee_id,
            current_kldb,
            current_manual_skills,
            kldb_esco_df,
            occupation_skill_relations_df,
            skills_df,
            st.session_state.occupation_skills_mapping,
            occupations_df,
            current_esco_role,  # Transfer the saved ESCO role
            current_manual_essential_skills,
            current_manual_optional_skills,
            current_removed_skills
        )
        
        # A profile should always be created (even if no role is found)
        if profile:
            current_role = profile['current_role']
            st.write(f"**Anzahl Skills:** {len(profile['skills'])}")
            
            # Show role information
            if current_role.get('KldB_Code') or current_role.get('ESCO_Label'):
                st.write(f"**Aktuelle Rolle:** {current_role.get('KldB_Label', current_role.get('ESCO_Label', 'Keine Rolle'))}")
            
            # Show skills with legend
            if profile['skills']:
                # Key to skill color points
                st.markdown("**Skill-Legende:**")
                legend_col1, legend_col2 = st.columns(2)
                with legend_col1:
                    st.write("**Essential Skills** - Unverzichtbare Skills (zählen doppelt)")
                with legend_col2:
                    st.write("**Optional Skills** - Hilfreiche Skills (zählen einfach)")
                
                st.markdown("---")
                
                st.write("**Zugewiesene Skills:**")
                render_skills_two_columns(profile['skills'], left_title="Essentiell", right_title="Optional")
            else:
                st.info("Keine Skills zugewiesen. Bitte weisen Sie eine Rolle zu oder fügen Sie manuelle Skills hinzu.")
        else:
            # This shouldn't happen anymore, since we always return a profile
            st.warning("Konnte kein Kompetenzprofil erstellen. Bitte überprüfen Sie die Mitarbeiterdaten.")
    
    st.markdown("---")
    
    # Manual role assignment
    st.markdown("---")
    st.subheader("Aktuelle Rolle manuell zuweisen")
    
    # Select from available KldB roles
    st.write("**Aus verfügbaren KldB-Rollen wählen**")
    
    # Create a dropdown list containing all available KldB roles
    # Remove duplicates based on KldB_Code AND KldB_Label
    available_kldb_roles = kldb_esco_df[['KldB_Code', 'KldB_Label']].drop_duplicates(subset=['KldB_Code', 'KldB_Label'])
    available_kldb_roles = available_kldb_roles.sort_values('KldB_Label')
    
    # Create options for the dropdown box - a shorter, cleaner display
    kldb_options = []
    seen_options = set()  # Prevents duplicate entries in the display
    
    for _, row in available_kldb_roles.iterrows():
        kldb_label = str(row['KldB_Label']).strip()
        kldb_code = str(row['KldB_Code']).strip()
        
        # Skip empty entries
        if not kldb_label or not kldb_code or kldb_label == 'nan' or kldb_code == 'nan':
            continue
        
        # Shorten long labels for better readability
        display_label = kldb_label
        if len(display_label) > 40:
            display_label = display_label[:37] + "..."
        
        # Create a clean option
        option = f"{display_label} | {kldb_code}"
        
        # Prevent duplicate listings
        if option not in seen_options:
            kldb_options.append(option)
            seen_options.add(option)
    
    # Sort the options alphabetically
    kldb_options.sort()
    kldb_options.insert(0, "Bitte wählen Sie eine KldB-Rolle...")
    
    # Show the number of available roles
    st.write(f"**Verfügbare KldB-Rollen:** {len(kldb_options) - 1} Rollen")
    
    selected_kldb_role = st.selectbox("Wählen Sie eine KldB-Rolle:", kldb_options, help="Scrollen Sie durch die Liste oder tippen Sie den Anfang des Berufsnamens")
    
    if selected_kldb_role and selected_kldb_role != "Bitte wählen Sie eine KldB-Rolle...":
        # Extract KldB code from the selection
        kldb_code = selected_kldb_role.split(" | ")[1]
        kldb_label = selected_kldb_role.split(" | ")[0]
        
        # Reconstruct the complete label from the original data
        full_label = available_kldb_roles[
            (available_kldb_roles['KldB_Code'] == kldb_code) & 
            (available_kldb_roles['KldB_Label'].str.contains(kldb_label.split('...')[0] if '...' in kldb_label else kldb_label, na=False))
        ]['KldB_Label'].iloc[0] if not available_kldb_roles[
            (available_kldb_roles['KldB_Code'] == kldb_code) & 
            (available_kldb_roles['KldB_Label'].str.contains(kldb_label.split('...')[0] if '...' in kldb_label else kldb_label, na=False))
        ].empty else kldb_label
        
        # View full details
        st.write(f"**Ausgewählte KldB-Rolle:** {full_label} ({kldb_code})")
        
        # Find all related ESCO roles for this KldB role
        matching_roles = (
            get_unique_esco_roles(kldb_esco_df, kldb_code)
            .drop_duplicates(subset=['ESCO_Code', 'ESCO_Label'])
        )
        
        if not matching_roles.empty:
            st.write(f"**Verfügbare ESCO-Rollen für '{full_label}':**")
            
            for idx, role in matching_roles.iterrows():
                esco_label = role['ESCO_Label']
                esco_code = role['ESCO_Code']
                
                # Acquire the skills needed for this ESCO role
                role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                
                with st.expander(f"{esco_label}"):
                    if role_skills:
                        # Key to skill color points
                        st.markdown("**Skill-Legende:**")
                        legend_col1, legend_col2 = st.columns(2)
                        with legend_col1:
                            st.write("**Essential Skills** - Unverzichtbare Skills")
                        with legend_col2:
                            st.write("**Optional Skills** - Hilfreiche Skills")
                        
                        st.markdown("---")
                        
                        st.write("**Skills:**")
                        render_skills_two_columns_table(role_skills, left_title="Essentiell", right_title="Optional")
                        
                        # Apply button
                        if st.button(f"Als aktuelle Rolle übernehmen", key=f"assign_kldb_{idx}"):
                            # Update the KldB code in the session state data
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'KldB_5_digit'
                            ] = kldb_code
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'ESCO_Role'
                            ] = esco_label # Save the ESCO role
                            
                            # Save as CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Rolle '{esco_label}' wurde als aktuelle Rolle zugewiesen und gespeichert!")
                            else:
                                st.warning(f"Rolle zugewiesen, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    else:
                        st.write("Keine Skills für diese Rolle gefunden.")
        else:
            st.warning(f"Keine ESCO-Rollen für KldB-Rolle '{full_label}' gefunden.")
    
    # Option 3: Manually adjust essential and optional skills
    st.markdown("---")
    st.subheader("Essential und Optional Skills anpassen")
    
    # Check whether a current role has been assigned
    if not current_kldb:
        st.info("**Keine aktuelle Rolle zugewiesen.** Bitte weisen Sie oben eine Rolle zu, um die Skills anzupassen.")
    else:
        # Get the latest skill data
        current_manual_essential_skills = current_employee_data.get('Manual_Essential_Skills', '')
        current_manual_optional_skills = current_employee_data.get('Manual_Optional_Skills', '')
        current_removed_skills = current_employee_data.get('Removed_Skills', '')
        
        # Create the current profile for the skill display
        profile_for_skills = create_employee_profile(
            employee_id,
            current_kldb,
            current_manual_skills,
            kldb_esco_df,
            occupation_skill_relations_df,
            skills_df,
            st.session_state.occupation_skills_mapping,
            occupations_df,
            current_esco_role,
            current_manual_essential_skills,
            current_manual_optional_skills,
            current_removed_skills
        )
        
        if profile_for_skills:
            # Show current skills with checkboxes to remove them
            st.write("**Aktuelle Skills der Rolle:**")
            
            # Group skills by type - include manual skills in the main lists
            essential_skills = [s for s in profile_for_skills['skills'] if s.get('is_essential', False)]
            optional_skills = [s for s in profile_for_skills['skills'] if not s.get('is_essential', False)]
            
            # Essential Skills
            if essential_skills:
                st.write("**Essential Skills:**")
                
                # Checkbox for "Select all essential skills"
                select_all_essential = st.checkbox(f"Alle Essential Skills auswählen ({len(essential_skills)} Skills)", key="select_all_essential")
                
                essential_to_remove = []
                essential_checkboxes = {}
                
                for i, skill in enumerate(essential_skills):
                    # Use the "Select All" checkbox as the default
                    default_value = select_all_essential
                    
                    # A separate checkbox for each skill with a unique key
                    is_checked = st.checkbox(
                        f"Entfernen: {skill['skill_label']}", 
                        key=f"remove_essential_{i}_{skill['skill_uri']}",
                        value=default_value
                    )
                    
                    if is_checked:
                        essential_to_remove.append(skill['skill_label'])
                    essential_checkboxes[skill['skill_label']] = is_checked
                
                # Display of selected Essential Skills
                if essential_to_remove:
                    st.info(f"**{len(essential_to_remove)} von {len(essential_skills)} Essential Skills ausgewählt**")
                    st.warning(f"Folgende Essential Skills werden entfernt: {', '.join(essential_to_remove)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Essential Skills entfernen bestätigen", key="confirm_remove_essential"):
                            # Process the skills to be removed
                            current_removed_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                            current_manual_essential_list = [s.strip() for s in current_manual_essential_skills.split(';') if s.strip()]
                            
                            for skill_to_remove in essential_to_remove:
                                # Check whether it is a manual Essential Skill
                                if skill_to_remove in current_manual_essential_list:
                                    # Remove from Manual Essential Skills
                                    current_manual_essential_list.remove(skill_to_remove)
                                else:
                                    # Add to the Removed_Skills list (for automatic skills)
                                    if skill_to_remove not in current_removed_list:
                                        current_removed_list.append(skill_to_remove)
                            
                            new_removed_skills = '; '.join(current_removed_list)
                            new_manual_essential_skills = '; '.join(current_manual_essential_list)
                            
                            # Update the session state data
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Removed_Skills'
                            ] = new_removed_skills
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Essential_Skills'
                            ] = new_manual_essential_skills
            
                            # Save as CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"{len(essential_to_remove)} Essential Skills entfernt!")
                            else:
                                st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("Auswahl zurücksetzen", key="reset_essential_selection"):
                            st.rerun()
            
            # Optional Skills
            if optional_skills:
                st.write("**Optional Skills:**")
                
                # Checkbox for "Select all optional skills"
                select_all_optional = st.checkbox(f"Alle Optional Skills auswählen ({len(optional_skills)} Skills)", key="select_all_optional")
                
                optional_to_remove = []
                optional_checkboxes = {}
                
                for i, skill in enumerate(optional_skills):
                    # Use the "Select All" checkbox as the default
                    default_value = select_all_optional
                    
                    # A separate checkbox for each skill with a unique key
                    is_checked = st.checkbox(
                        f"Entfernen: {skill['skill_label']}", 
                        key=f"remove_optional_{i}_{skill['skill_uri']}",
                        value=default_value
                    )
                    
                    if is_checked:
                        optional_to_remove.append(skill['skill_label'])
                    optional_checkboxes[skill['skill_label']] = is_checked
                
                # Display of selected optional skills
                if optional_to_remove:
                    st.info(f"**{len(optional_to_remove)} von {len(optional_skills)} Optional Skills ausgewählt**")
                    st.warning(f"Folgende Optional Skills werden entfernt: {', '.join(optional_to_remove)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Optional Skills entfernen bestätigen", key="confirm_remove_optional"):
                            # Process the skills to be removed
                            current_removed_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                            current_manual_optional_list = [s.strip() for s in current_manual_optional_skills.split(';') if s.strip()]
                            
                            for skill_to_remove in optional_to_remove:
                                # Check whether it is a manual optional skill
                                if skill_to_remove in current_manual_optional_list:
                                    # Remove from Manual Optional Skills
                                    current_manual_optional_list.remove(skill_to_remove)
                                else:
                                    # Add to the Removed_Skills list (for automatic skills)
                                    if skill_to_remove not in current_removed_list:
                                        current_removed_list.append(skill_to_remove)
                            
                            new_removed_skills = '; '.join(current_removed_list)
                            new_manual_optional_skills = '; '.join(current_manual_optional_list)
                            
                            # Update the session state data
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Removed_Skills'
                            ] = new_removed_skills
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Optional_Skills'
                            ] = new_manual_optional_skills
                            
                            # Save as CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"{len(optional_to_remove)} Optional Skills entfernt!")
                            else:
                                st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("Auswahl zurücksetzen", key="reset_optional_selection"):
                            st.rerun()
            

            
            # Load all available ESCO skills for the dropdown menu
            available_esco_skills = get_all_available_esco_skills(skills_df, skills_en_df)
            
            # Add new essential skills
            st.write("**Neue Essential Skills hinzufügen:**")
            
            # Create dropdown options for Essential Skills
            essential_skill_options = ["Bitte wählen Sie einen Essential Skill..."]
            essential_skill_labels = {}
            
            for skill in available_esco_skills:
                option_label = skill['display_label']
                essential_skill_options.append(option_label)
                essential_skill_labels[option_label] = skill['german_label']
            
            # Dropdown menu for new Essential Skills
            selected_essential_skill = st.selectbox(
                "Wählen Sie einen Essential Skill aus:",
                essential_skill_options,
                key="essential_skill_dropdown"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Essential Skill hinzufügen", key="add_essential_skill"):
                    if selected_essential_skill and selected_essential_skill != "Bitte wählen Sie einen Essential Skill...":
                        # Add the skill to the list
                        skill_label = essential_skill_labels[selected_essential_skill]
                        current_essential_list = [s.strip() for s in current_manual_essential_skills.split(';') if s.strip()]
                        
                        if skill_label not in current_essential_list:
                            current_essential_list.append(skill_label)
                            new_essential_skills = '; '.join(current_essential_list)
                            
                            # Update the session state data
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Essential_Skills'
                            ] = new_essential_skills
                            
                            # Save as CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Essential Skill '{skill_label}' hinzugefügt!")
                            else:
                                st.warning("Skill hinzugefügt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                        else:
                            st.warning(f"Skill '{skill_label}' ist bereits hinzugefügt!")
                    else:
                        st.warning("Bitte wählen Sie einen Skill aus!")
            
            with col2:
                if st.button("Alle Essential Skills entfernen", key="remove_all_essential"):
                    st.session_state.employees_data.loc[
                        st.session_state.employees_data['Employee_ID'] == employee_id, 
                        'Manual_Essential_Skills'
                    ] = ''
                    
                    if save_employees_to_csv(st.session_state.employees_data):
                        st.success("Alle Essential Skills entfernt!")
                    else:
                        st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                    
                    st.rerun()
                
                st.markdown("---")
                
            # Add new optional skills
            st.write("**Neue Optional Skills hinzufügen:**")
            
            # Create dropdown options for optional skills
            optional_skill_options = ["Bitte wählen Sie einen Optional Skill..."]
            optional_skill_labels = {}
            
            for skill in available_esco_skills:
                option_label = skill['display_label']
                optional_skill_options.append(option_label)
                optional_skill_labels[option_label] = skill['german_label']
            
            # Dropdown menu for new optional skills
            selected_optional_skill = st.selectbox(
                "Wählen Sie einen Optional Skill aus:",
                optional_skill_options,
                key="optional_skill_dropdown"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Optional Skill hinzufügen", key="add_optional_skill"):
                    if selected_optional_skill and selected_optional_skill != "Bitte wählen Sie einen Optional Skill...":
                        # Add the skill to the list
                        skill_label = optional_skill_labels[selected_optional_skill]
                        current_optional_list = [s.strip() for s in current_manual_optional_skills.split(';') if s.strip()]
                        
                        if skill_label not in current_optional_list:
                            current_optional_list.append(skill_label)
                            new_optional_skills = '; '.join(current_optional_list)
                            
                            # Update the session state data
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Optional_Skills'
                            ] = new_optional_skills
                            
                            # Save as CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Optional Skill '{skill_label}' hinzugefügt!")
                            else:
                                st.warning("Skill hinzugefügt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                        else:
                            st.warning(f"Skill '{skill_label}' ist bereits hinzugefügt!")
                    else:
                        st.warning("Bitte wählen Sie einen Skill aus!")
            
            with col2:
                if st.button("Alle Optional Skills entfernen", key="remove_all_optional"):
                    st.session_state.employees_data.loc[
                        st.session_state.employees_data['Employee_ID'] == employee_id, 
                        'Manual_Optional_Skills'
                    ] = ''
                    
                    if save_employees_to_csv(st.session_state.employees_data):
                        st.success("Alle Optional Skills entfernt!")
                    else:
                        st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                    
                    st.rerun()
            
            # View removed skills
            if current_removed_skills and current_removed_skills.strip():
                st.write("**Aktuell entfernte Skills:**")
                removed_skills_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                
                # Checkbox for "Select all removed skills"
                select_all_removed = st.checkbox(f"Alle entfernten Skills auswählen ({len(removed_skills_list)} Skills)", key="select_all_removed")
                
                removed_to_restore = []
                removed_checkboxes = {}
                
                for i, skill in enumerate(removed_skills_list):
                    # Use the "Select All" checkbox as the default
                    default_value = select_all_removed
                    
                    # A separate checkbox for each removed skill
                    is_checked = st.checkbox(
                        f"Wiederherstellen: {skill}", 
                        key=f"restore_removed_{skill.lower().replace(' ', '_')}",
                        value=default_value
                    )
                    
                    if is_checked:
                        removed_to_restore.append(skill)
                    removed_checkboxes[skill] = is_checked
                
                # Display of selected remote skills
                if removed_to_restore:
                    st.info(f"**{len(removed_to_restore)} von {len(removed_skills_list)} entfernten Skills ausgewählt**")
                    st.warning(f"Folgende Skills werden wiederhergestellt: {', '.join(removed_to_restore)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Entfernte Skills wiederherstellen bestätigen", key="confirm_restore_removed"):
                            # Remove the selected skills from the Removed_Skills list
                            current_removed_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                            for skill_to_restore in removed_to_restore:
                                if skill_to_restore in current_removed_list:
                                    current_removed_list.remove(skill_to_restore)
                            
                            new_removed_skills = '; '.join(current_removed_list)
                            
                            # Update the session state data
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Removed_Skills'
                            ] = new_removed_skills
                            
                            # Save as CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"{len(removed_to_restore)} entfernte Skills wiederhergestellt!")
                            else:
                                st.warning("Skills wiederhergestellt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("Auswahl zurücksetzen", key="reset_removed_selection"):
                            st.rerun()
                else:
                    # Button to restore all removed skills (if none are selected)
                    if st.button("Alle entfernten Skills wiederherstellen", key="restore_all_removed"):
                        st.session_state.employees_data.loc[
                            st.session_state.employees_data['Employee_ID'] == employee_id, 
                            'Removed_Skills'
                        ] = ''
                        
                        if save_employees_to_csv(st.session_state.employees_data):
                            st.success("Alle entfernten Skills wiederhergestellt!")
                        else:
                            st.warning("Skills wiederhergestellt, aber Speichern fehlgeschlagen!")
                        
                        st.rerun()
            else:
                st.write("**Entfernte Skills:** Keine Skills entfernt")
            
            # General Reset Function
            st.markdown("---")
            st.write("**Alle Skill-Anpassungen zurücksetzen:**")
            if st.button("Alle Skill-Anpassungen zurücksetzen", key="reset_all_skills"):
                # Reset all skill adjustments
                st.session_state.employees_data.loc[
                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                    ['Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']
                ] = ['', '', '']
                
                # Save as CSV
                if save_employees_to_csv(st.session_state.employees_data):
                    st.success("Alle Skill-Anpassungen wurden zurückgesetzt!")
                else:
                    st.warning("Zurücksetzung gespeichert, aber CSV-Export fehlgeschlagen!")
                
                st.rerun()
        else:
            # This shouldn't happen anymore, since we always return a profile
            st.warning("Konnte kein Kompetenzprofil für Skill-Anpassungen erstellen. Bitte überprüfen Sie die Mitarbeiterdaten oder weisen Sie eine Rolle zu.")
    


def show_occupation_matching(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df):
    st.header("Berufsabgleich")
    
    # Use updated employee data from session state
    employees_df = st.session_state.employees_data
    
    if employees_df.empty:
        st.warning("Keine Mitarbeiterdaten gefunden.")
        return
    
    # Use the global employee selection
    if 'current_employee_id' not in st.session_state:
        st.info("Bitte wählen Sie einen Mitarbeiter in der Sidebar aus.")
        return
    
    employee_id = st.session_state.current_employee_id
    
    # Retrieve current data from session state (not cached)
    current_employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
    
    st.subheader(f"Berufsabgleich für {current_employee_data.get('Name', employee_id)}")
    
    # Current role of the employee
    current_kldb = current_employee_data.get('KldB_5_digit', '')
    current_manual_skills = current_employee_data.get('Manual_Skills', '')
    current_esco_role = current_employee_data.get('ESCO_Role', '')
    
    # Retrieve additional skill data from session state
    current_manual_essential_skills = current_employee_data.get('Manual_Essential_Skills', '')
    current_manual_optional_skills = current_employee_data.get('Manual_Optional_Skills', '')
    current_removed_skills = current_employee_data.get('Removed_Skills', '')
    
    # Create the current employee profile
    current_profile = create_employee_profile(
        employee_id,
        current_kldb,
        current_manual_skills,
        kldb_esco_df,
        occupation_skill_relations_df,
        skills_df,
        st.session_state.occupation_skills_mapping,
        occupations_df,
        current_esco_role,  # Transfer the saved ESCO role
        current_manual_essential_skills,
        current_manual_optional_skills,
        current_removed_skills
    )
    
    if not current_profile:
        st.error("Konnte kein aktuelles Mitarbeiterprofil erstellen.")
        return
    
    # Show the current role and the selected future role side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aktuelle Rolle:**")
        current_role = current_profile['current_role']
        
        # Display the saved ESCO role or the calculated
        if current_esco_role:
            st.write(f"• KldB: {current_role.get('KldB_Label', 'N/A')} ({current_role.get('KldB_Code', 'N/A')})")
            st.write(f"• ESCO: {current_esco_role}")
            st.write(f"• Anzahl Skills: {len(current_profile['skills'])}")
        else:
            st.write(f"• KldB: {current_role.get('KldB_Label', 'N/A')} ({current_role.get('KldB_Code', 'N/A')})")
            st.write(f"• ESCO: {current_role.get('ESCO_Label', 'N/A')}")
            st.write(f"• Anzahl Skills: {len(current_profile['skills'])}")
    
    with col2:
        st.write("**Ausgewählte Zielrolle:**")
        # Display the selected future role based on session state or the current dropdown selection
        if 'selected_target_role' in st.session_state:
            target_role = st.session_state.selected_target_role
            st.write(f"• **KldB-Code:** {target_role['KldB_Code']}")
            st.write(f"• **Berufsbezeichnung:** {target_role['KldB_Label']}")
            
            # Show the ESCO category only if it differs from the KldB label
            if target_role['ESCO_Label'].lower() != target_role['KldB_Label'].lower():
                st.write(f"• **ESCO-Kategorie:** {target_role['ESCO_Label']}")
            
            # Button to reset the future role
            if st.button("Zielrolle zurücksetzen", key="reset_target_role_matching", type="secondary"):
                # Delete the saved future role from the employee data
                st.session_state.employees_data.loc[
                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                    ['Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label']
                ] = ['', '', '', '']
                
                # Save as CSV
                if save_employees_to_csv(st.session_state.employees_data):
                    # Delete from session state
                    if 'selected_target_role' in st.session_state:
                        del st.session_state.selected_target_role
                    st.success("Zielrolle wurde zurückgesetzt und gespeichert!")
                else:
                    st.warning("Zielrolle zurückgesetzt, aber Speichern fehlgeschlagen!")
                
                st.rerun()
        else:
            st.write("• Keine Zielrolle ausgewählt")
    
    st.markdown("---")
    
    # Select a future role
    st.subheader("Neue Zielrolle auswählen")
    
    # Selection method
    selection_method = st.radio(
        "Wie möchten Sie die Zielrolle auswählen?",
        ["Zielrolle auswählen", "Zukünftige Skills auswählen"],
        help="Wählen Sie zwischen direkter Rollenauswahl oder Skill-basierter Suche"
    )
    
    st.markdown("---")
    
    if selection_method == "Zielrolle auswählen":
        # Create a dropdown list containing all available KldB roles
        # Remove duplicates based on KldB_Code AND KldB_Label
        available_kldb_roles = kldb_esco_df[['KldB_Code', 'KldB_Label']].drop_duplicates(subset=['KldB_Code', 'KldB_Label'])
        available_kldb_roles = available_kldb_roles[available_kldb_roles['KldB_Label'].apply(is_valid_kldb_label)]
        # Remove additional duplicates: If the code and label are identical, keep only one
        available_kldb_roles = available_kldb_roles.drop_duplicates(subset=['KldB_Code', 'KldB_Label'], keep='first')
        
        # IMPORTANT: Remove duplicates based on label - if multiple codes have the same label,
        # keep only the first one (to avoid duplicates in the dropdown list)
        available_kldb_roles = available_kldb_roles.drop_duplicates(subset=['KldB_Label'], keep='first')
        
        available_kldb_roles = available_kldb_roles.sort_values('KldB_Label')
        
        # Create options for the dropdown box - unique and without duplicates
        kldb_options = []
        seen_labels = set()  # Prevents duplicates based on label
        seen_codes = set()  # Prevents duplicates based on code
        
        for _, row in available_kldb_roles.iterrows():
            kldb_label = str(row['KldB_Label']).strip()
            kldb_code = str(row['KldB_Code']).strip()
            
            # Skip empty entries
            if not kldb_label or not kldb_code or kldb_label == 'nan' or kldb_code == 'nan':
                continue
            
            # Normalize labels for comparison (remove leading/trailing spaces, normalize case)
            normalized_label = kldb_label.strip()
            
            # Skip if this label already exists (prevents duplicates)
            if normalized_label in seen_labels:
                continue
            
            # Skip if this code already exists (for added security)
            if kldb_code in seen_codes:
                continue
            
            seen_labels.add(normalized_label)
            seen_codes.add(kldb_code)
            
            # Create a unique option
            option = f"{kldb_label} | {kldb_code}"
            kldb_options.append(option)
        
        # Sort the options alphabetically
        kldb_options.sort()
        kldb_options.insert(0, "Bitte wählen Sie eine neue Zielrolle...")
        
        selected_target_role = st.selectbox("KldB-Zielrolle auswählen:", kldb_options, help="Scrollen Sie durch die Liste oder tippen Sie den Anfang des Berufsnamens")
        
        # Show a dynamic overview based on the current selection
        if selected_target_role and selected_target_role != "Bitte wählen Sie eine neue Zielrolle...":
            # Extract the KldB code and label from the selection
            kldb_code = selected_target_role.split(" | ")[1]
            kldb_label = selected_target_role.split(" | ")[0]
            
            # Find the complete label in the ORIGINAL DATA (kldb_esco_df), not in available_kldb_roles
            # Search for the label in the original data to find the correct code
            matching_rows = kldb_esco_df[
                (kldb_esco_df['KldB_Label'].str.strip() == kldb_label.strip()) |
                (kldb_esco_df['KldB_Label'].str.contains(kldb_label.split('...')[0] if '...' in kldb_label else kldb_label, case=False, na=False))
            ]
            
            if not matching_rows.empty:
                # Use the first entry found
                first_match = matching_rows.iloc[0]
                full_label = first_match['KldB_Label']
                # Use the code from the original data, not from the filtered list
                kldb_code = first_match['KldB_Code']
            else:
                # Fallback: Use the extracted values
                full_label = kldb_label
            
            # Display a dynamic overview of the selected KldB role
            st.markdown("---")
            st.write("**Aktuell ausgewählte KldB-Zielrolle:**")
            st.write(f"• KldB: {full_label} ({kldb_code})")
            
            # Find all ESCO roles for the selected KldB role
            # Search for all codes associated with this label (if multiple codes share the same label)
            all_matching_codes = kldb_esco_df[
                kldb_esco_df['KldB_Label'].str.strip() == full_label.strip()
            ]['KldB_Code'].unique()
            
            # Collect all ESCO roles for all matching codes
            target_roles_list = []
            for code in all_matching_codes:
                roles = get_unique_esco_roles(kldb_esco_df, code)
                if not roles.empty:
                    target_roles_list.append(roles)
            
            # Combine all the rolls you've found
            if target_roles_list:
                target_roles = pd.concat(target_roles_list, ignore_index=True).drop_duplicates(subset=['ESCO_Code', 'ESCO_Label'])
            else:
                target_roles = pd.DataFrame()
            
            if not target_roles.empty:
                st.write(f"**Verfügbare ESCO-Rollen für Zielrolle '{full_label}':**")
                
                for idx, role in target_roles.iterrows():
                    esco_label = role['ESCO_Label']
                    esco_code = role['ESCO_Code']
                    
                    # Acquire the skills needed for this ESCO role
                    role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                    
                    with st.expander(f"{esco_label}"):
                        if role_skills:
                            st.write("**Skills:**")
                            render_skills_two_columns_table(role_skills, left_title="Essentiell", right_title="Optional")
                            
                            # Button to select as the future role for comparison
                            if st.button(f"Als Zielrolle für Vergleich auswählen", key=f"select_target_role_{idx}"):
                                # Check whether it is the same ESCO role
                                current_esco_role = current_employee_data.get('ESCO_Role', '')
                                
                                if current_esco_role and esco_label == current_esco_role:
                                    st.warning("Sie haben die gleiche ESCO-Rolle wie die aktuelle Rolle ausgewählt. Bitte wählen Sie eine andere Zielrolle.")
                                    # Delete the previous match from the session state
                                    if 'current_match' in st.session_state:
                                        del st.session_state.current_match
                                    if 'current_target_role_key' in st.session_state:
                                        del st.session_state.current_target_role_key
                                else:
                                    # Create a future role profile for comparison
                                    target_role_data = {
                                        'ESCO_Label': esco_label,
                                        'ESCO_Code': esco_code,
                                        'KldB_Label': full_label,
                                        'KldB_Code': kldb_code
                                    }
                                    
                                    # Calculate the match between the current employee and the future role
                                    match_result = calculate_occupation_match(
                                        current_profile, 
                                        target_role_data, 
                                        occupation_skill_relations_df, 
                                        skills_df, 
                                        st.session_state.occupation_skills_mapping, 
                                        occupations_df
                                    )
                                    
                                    if match_result and match_result['has_target_skills']:
                                        # Save match result in session state
                                        st.session_state.current_match = match_result
                                        st.session_state.current_target_role_key = f"select_target_role_{idx}"
                                        st.success(f"Match berechnet für {esco_label}")
                                        st.rerun()
                
                # Show the role comparison overview if a match is found and for "Select future role"
                if ('current_match' in st.session_state and st.session_state.current_match and 
                    ('current_target_role_key' not in st.session_state or 'skill_based' not in st.session_state.get('current_target_role_key', ''))):
                    match_result = st.session_state.current_match
                    
                    st.markdown("---")
                    # Show match results
                    st.subheader("Rollenvergleich")
                    
                    # Match Results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if match_result['has_target_skills']:
                            st.metric("Fit-Score % (nicht gewichtet)", f"{match_result['match_percentage']:.1f}%", help="Berechnet den prozentualen Anteil der Skills, die der Mitarbeiter bereits besitzt, im Verhältnis zu allen benötigten Skills der Zielrolle. Jeder Skill wird gleich gewichtet, unabhängig von seiner Wichtigkeit.")
                        else:
                            st.metric("Fit-Score % (nicht gewichtet)", "N/A", help="Berechnet den prozentualen Anteil der Skills, die der Mitarbeiter bereits besitzt, im Verhältnis zu allen benötigten Skills der Zielrolle. Jeder Skill wird gleich gewichtet, unabhängig von seiner Wichtigkeit.")
                    
                    with col2:
                        if match_result['has_target_skills']:
                            st.metric("Fit-Score % (gewichtet)", f"{match_result['weighted_fit_percentage']:.1f}%", help="Berechnet den gewichteten Fit-Score, bei dem Essential Skills doppelt zählen (Gewichtung 2) und Optional Skills einfach (Gewichtung 1). Diese Berechnung gibt einen realistischeren Eindruck der Eignung, da kritische Kompetenzen stärker berücksichtigt werden.")
                        else:
                            st.metric("Fit-Score % (gewichtet)", "N/A", help="Berechnet den gewichteten Fit-Score, bei dem Essential Skills doppelt zählen (Gewichtung 2) und Optional Skills einfach (Gewichtung 1). Diese Berechnung gibt einen realistischeren Eindruck der Eignung, da kritische Kompetenzen stärker berücksichtigt werden.")
                    
                    st.markdown("---")
                    
                    # Employee comparison for this future role
                    st.subheader("Vergleich mit anderen Mitarbeitern")
                    
                    # Compare with other employees
                    with st.spinner("Berechne Vergleich mit anderen Mitarbeitern..."):
                        all_employee_scores = compare_employees_for_target_role(
                            match_result['target_role'], 
                            st.session_state.employees_data, 
                            kldb_esco_df, 
                            occupation_skill_relations_df, 
                            skills_df, 
                            st.session_state.occupation_skills_mapping, 
                            occupations_df
                        )
                    
                    if all_employee_scores:
                        # Find the current employee in the list
                        current_employee_id = st.session_state.current_employee_id
                        current_employee_rank = None
                        current_employee_score = None
                        
                        for i, score in enumerate(all_employee_scores):
                            if score['employee_id'] == current_employee_id:
                                current_employee_rank = i + 1
                                current_employee_score = score
                                break
                        
                        if current_employee_rank and current_employee_score:
                            # Filter only employees with a higher score
                            better_employees = [score for score in all_employee_scores 
                                             if score['weighted_fit_percentage'] > current_employee_score['weighted_fit_percentage']]
                            
                            if better_employees:
                                st.info(f"**Ihr aktueller Mitarbeiter belegt Platz {current_employee_rank} von {len(all_employee_scores)} Mitarbeitern für diese Zielrolle.**")
                                st.warning(f"**Es gibt {len(better_employees)} Mitarbeiter mit einem besseren Fit-Score für diese Zielrolle:**")
                                
                                # Show employees with higher scores
                                st.write("**Mitarbeiter mit besserem Fit-Score:**")
                                
                                # Create a DataFrame for better visualization
                                better_employees_data = []
                                for i, score in enumerate(better_employees):
                                    better_employees_data.append({
                                        'Rang': all_employee_scores.index(score) + 1,
                                        'Mitarbeiter': score['employee_name'],
                                        'Aktuelle Rolle': score['current_role'],
                                        'Fit-Score (gewichtet)': f"{score['weighted_fit_percentage']:.1f}%",
                                        'Fit-Score (nicht gewichtet)': f"{score['match_percentage']:.1f}%",
                                        'Matching Skills': score['matching_skills_count'],
                                        'Fehlende Skills': score['missing_skills_count'],
                                        'Score-Differenz': f"+{score['weighted_fit_percentage'] - current_employee_score['weighted_fit_percentage']:.1f}%"
                                    })
                                
                                comparison_df = pd.DataFrame(better_employees_data)
                                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                                
                                # Show additional information
                                best_score = better_employees[0]
                                score_difference = best_score['weighted_fit_percentage'] - current_employee_score['weighted_fit_percentage']
                                st.warning(f"Der beste Mitarbeiter ({best_score['employee_name']}) hat einen {score_difference:.1f}% höheren Fit-Score.")
                            else:
                                st.success("**Dieser Mitarbeiter ist der beste Kandidat für diese Zielrolle!**")
                                st.info(f"Von {len(all_employee_scores)} Mitarbeitern hat keiner einen besseren Fit-Score.")
                        else:
                            st.warning("Aktueller Mitarbeiter konnte nicht in der Vergleichsliste gefunden werden.")
                    else:
                        st.info("Keine anderen Mitarbeiter für den Vergleich verfügbar.")
                    
                    st.markdown("---")
                    
                    # Skills that are missing and skills that are present
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if match_result['missing_skills']:
                            st.write("**Fehlende Skills für neue Rolle:**")
                            render_missing_skills_with_favorites(match_result['missing_skills'], session_key_prefix="favorite_main")
                        else:
                            st.write("**Fehlende Skills:**")
                            st.write("Alle benötigten Skills sind vorhanden!")
                    
                    with col2:
                        if match_result['matching_skills']:
                            st.write("**Bereits vorhandene Skills:**")
                            render_skills_two_columns_table(match_result['matching_skills'], left_title="Essentiell", right_title="Optional")
                        else:
                            st.write("**Bereits vorhandene Skills:**")
                            st.write("Keine Übereinstimmungen gefunden.")
            else:
                st.warning(f"Keine ESCO-Rollen für Zielrolle '{full_label}' gefunden.")
        else:
            st.info("Bitte wählen Sie eine neue Zielrolle aus der Dropdown-Box.")
    
    elif selection_method == "Zukünftige Skills auswählen":
        st.subheader("Skills für zukünftige Rolle auswählen")
        
        # Get all available skills
        all_skills = skills_df[['preferredLabel', 'conceptUri']].drop_duplicates().sort_values('preferredLabel')
        
        # Multi-select for skills
        selected_skills = st.multiselect(
            "Wählen Sie die Skills für die zukünftige Rolle:",
            options=all_skills['preferredLabel'].tolist(),
            help="Wählen Sie mehrere Skills aus, die für die zukünftige Rolle relevant sind"
        )
        
        if selected_skills:
            st.write(f"**Ausgewählte Skills ({len(selected_skills)}):**")
            for skill in selected_skills:
                st.write(f"• {skill}")
            
            # Find suitable roles based on selected skills
            if st.button("Passende Rollen suchen"):
                with st.spinner("Suche passende Rollen..."):
                    # Find roles that include these skills
                    matching_roles = []
                    
                    for _, role in kldb_esco_df.iterrows():
                        esco_label = role['ESCO_Label']
                        role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                        
                        if role_skills:
                            role_skill_labels = [skill['skill_label'].lower() for skill in role_skills]
                            selected_skill_labels = [skill.lower() for skill in selected_skills]
                            
                            # Calculate Match
                            matches = sum(1 for skill in selected_skill_labels if skill in role_skill_labels)
                            match_percentage = (matches / len(selected_skills)) * 100 if selected_skills else 0
                            
                            if match_percentage > 0:
                                matching_roles.append({
                                    'role': role,
                                    'match_percentage': match_percentage,
                                    'matches': matches,
                                    'total_skills': len(role_skills)
                                })
                    
                    # Sort by match percentage
                    matching_roles.sort(key=lambda x: x['match_percentage'], reverse=True)
                    
                    if matching_roles:
                        st.success(f"**{len(matching_roles)} passende Rollen gefunden:**")
                        
                        # Show Top 10 Roles
                        for i, match in enumerate(matching_roles[:10]):
                            role = match['role']
                            with st.expander(f"{role['KldB_Label']} ({role['KldB_Code']}) - {match['match_percentage']:.1f}% Übereinstimmung"):
                                st.write(f"**ESCO-Rolle:** {role['ESCO_Label']}")
                                st.write(f"**Übereinstimmung:** {match['matches']}/{len(selected_skills)} Skills ({match['match_percentage']:.1f}%)")
                                
                                # Show skills for the role
                                role_skills = get_skills_for_occupation_simple(role['ESCO_Label'], st.session_state.occupation_skills_mapping, occupations_df)
                                if role_skills:
                                    st.write("**Skills der Rolle:**")
                                    render_skills_two_columns_table(role_skills, left_title="Essentiell", right_title="Optional")
                                
                                # Button to select as the future role
                                if st.button(f"Als Zielrolle auswählen", key=f"select_skill_based_role_{i}"):
                                    # Create a future role profile for comparison
                                    target_role_data = {
                                        'ESCO_Label': role['ESCO_Label'],
                                        'ESCO_Code': role['ESCO_Code'],
                                        'KldB_Label': role['KldB_Label'],
                                        'KldB_Code': role['KldB_Code']
                                    }
                                    
                                    # Calculate the match between the current employee and the future role
                                    match_result = calculate_occupation_match(
                                        current_profile, 
                                        target_role_data, 
                                        occupation_skill_relations_df, 
                                        skills_df, 
                                        st.session_state.occupation_skills_mapping, 
                                        occupations_df
                                    )
                                    
                                    if match_result and match_result['has_target_skills']:
                                        # Save match result in session state
                                        st.session_state.current_match = match_result
                                        st.session_state.current_target_role_key = f"select_skill_based_role_{i}"
                                        st.success(f"Match berechnet für {role['ESCO_Label']}")
                                        st.rerun()
            
            # Show the role comparison overview for skill-based selection if a match is found
            if 'current_match' in st.session_state and st.session_state.current_match and 'current_target_role_key' in st.session_state and 'skill_based' in st.session_state.current_target_role_key:
                match_result = st.session_state.current_match
                
                st.markdown("---")
                # Show match results
                st.subheader("Rollenvergleich")
                
                # Match Results
                col1, col2 = st.columns(2)
                
                with col1:
                    if match_result['has_target_skills']:
                        st.metric("Fit-Score % (nicht gewichtet)", f"{match_result['match_percentage']:.1f}%", help="Berechnet den prozentualen Anteil der Skills, die der Mitarbeiter bereits besitzt, im Verhältnis zu allen benötigten Skills der Zielrolle. Jeder Skill wird gleich gewichtet, unabhängig von seiner Wichtigkeit.")
                    else:
                        st.metric("Fit-Score % (nicht gewichtet)", "N/A", help="Berechnet den prozentualen Anteil der Skills, die der Mitarbeiter bereits besitzt, im Verhältnis zu allen benötigten Skills der Zielrolle. Jeder Skill wird gleich gewichtet, unabhängig von seiner Wichtigkeit.")
                
                with col2:
                    if match_result['has_target_skills']:
                        st.metric("Fit-Score % (gewichtet)", f"{match_result['weighted_fit_percentage']:.1f}%", help="Berechnet den gewichteten Fit-Score, bei dem Essential Skills doppelt zählen (Gewichtung 2) und Optional Skills einfach (Gewichtung 1). Diese Berechnung gibt einen realistischeren Eindruck der Eignung, da kritische Kompetenzen stärker berücksichtigt werden.")
                    else:
                        st.metric("Fit-Score % (gewichtet)", "N/A", help="Berechnet den gewichteten Fit-Score, bei dem Essential Skills doppelt zählen (Gewichtung 2) und Optional Skills einfach (Gewichtung 1). Diese Berechnung gibt einen realistischeren Eindruck der Eignung, da kritische Kompetenzen stärker berücksichtigt werden.")
                
                st.markdown("---")
                
                # Employee comparison for this future role
                st.subheader("Vergleich mit anderen Mitarbeitern")
                
                # Compare with other employees
                with st.spinner("Berechne Vergleich mit anderen Mitarbeitern..."):
                    all_employee_scores = compare_employees_for_target_role(
                        match_result['target_role'], 
                        st.session_state.employees_data, 
                        kldb_esco_df, 
                        occupation_skill_relations_df, 
                        skills_df, 
                        st.session_state.occupation_skills_mapping, 
                        occupations_df
                    )
                
                if all_employee_scores:
                    # Find the current employee in the list
                    current_employee_id = st.session_state.current_employee_id
                    current_employee_rank = None
                    current_employee_score = None
                    
                    for i, score in enumerate(all_employee_scores):
                        if score['employee_id'] == current_employee_id:
                            current_employee_rank = i + 1
                            current_employee_score = score
                            break
                    
                    if current_employee_rank and current_employee_score:
                        # Filter only employees with a higher score
                        better_employees = [score for score in all_employee_scores 
                                         if score['weighted_fit_percentage'] > current_employee_score['weighted_fit_percentage']]
                        
                        if better_employees:
                            st.info(f"**Ihr aktueller Mitarbeiter belegt Platz {current_employee_rank} von {len(all_employee_scores)} Mitarbeitern für diese Zielrolle.**")
                            st.warning(f"**Es gibt {len(better_employees)} Mitarbeiter mit einem besseren Fit-Score für diese Zielrolle:**")
                            
                            # Show employees with higher scores
                            st.write("**Mitarbeiter mit besserem Fit-Score:**")
                            
                            # Create a DataFrame for better visualization
                            better_employees_data = []
                            for i, score in enumerate(better_employees):
                                better_employees_data.append({
                                    'Rang': all_employee_scores.index(score) + 1,
                                    'Mitarbeiter': score['employee_name'],
                                    'Aktuelle Rolle': score['current_role'],
                                    'Fit-Score (gewichtet)': f"{score['weighted_fit_percentage']:.1f}%",
                                    'Fit-Score (nicht gewichtet)': f"{score['match_percentage']:.1f}%",
                                    'Matching Skills': score['matching_skills_count'],
                                    'Fehlende Skills': score['missing_skills_count'],
                                    'Score-Differenz': f"+{score['weighted_fit_percentage'] - current_employee_score['weighted_fit_percentage']:.1f}%"
                                })
                            
                            comparison_df = pd.DataFrame(better_employees_data)
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Show additional information
                            best_score = better_employees[0]
                            score_difference = best_score['weighted_fit_percentage'] - current_employee_score['weighted_fit_percentage']
                            st.warning(f"Der beste Mitarbeiter ({best_score['employee_name']}) hat einen {score_difference:.1f}% höheren Fit-Score.")
                        else:
                            st.success("**Dieser Mitarbeiter ist der beste Kandidat für diese Zielrolle!**")
                            st.info(f"Von {len(all_employee_scores)} Mitarbeitern hat keiner einen besseren Fit-Score.")
                    else:
                        st.warning("Aktueller Mitarbeiter konnte nicht in der Vergleichsliste gefunden werden.")
                else:
                    st.info("Keine anderen Mitarbeiter für den Vergleich verfügbar.")
                
                st.markdown("---")
                
                # Skills that are missing and skills that are present
                col1, col2 = st.columns(2)
                
                with col1:
                    if match_result['missing_skills']:
                        st.write("**Fehlende Skills für neue Rolle:**")
                        render_missing_skills_with_favorites(match_result['missing_skills'], session_key_prefix="favorite_skill_based")
                    else:
                        st.write("**Fehlende Skills:**")
                        st.write("Alle benötigten Skills sind vorhanden!")
                
                with col2:
                    if match_result['matching_skills']:
                        st.write("**Bereits vorhandene Skills:**")
                        render_skills_two_columns_table(match_result['matching_skills'], left_title="Essentiell", right_title="Optional")
                    else:
                        st.write("**Bereits vorhandene Skills:**")
                        st.write("Keine Übereinstimmungen gefunden.")
                
                # The match result is already stored in the session state

def show_course_recommendations(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df):
    st.header("Kursempfehlungen")
    
    # Initialize session state for employee data if it does not exist
    if 'employees_data' not in st.session_state:
        # Add the "Name" column if it doesn't exist
        if 'Name' not in employees_df.columns:
            employees_df['Name'] = 'Unbekannt'
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
    # Use updated employee data from session state
    employees_df = st.session_state.employees_data
    
    # Check if a match result is available
    if 'current_match' not in st.session_state:
        st.info("**Bitte führe zuerst einen Berufsabgleich durch.**")
        st.info("Gehe zu 'Berufsabgleich' und wähle einen Mitarbeiter und eine Zielrolle aus.")
        return
    
    match_result = st.session_state.current_match
    
    if not match_result['has_target_skills']:
        st.warning("Keine Skills für den Zielberuf verfügbar. Kursempfehlungen können nicht generiert werden.")
        st.info("Tipp: Wähle einen anderen Zielberuf oder überprüfe die ESCO-Daten.")
        return
    
    if not match_result['missing_skills']:
        st.success("**Alle benötigten Skills sind bereits vorhanden!**")
        return
    
    # Show context information
    st.subheader("Kontext der Kursempfehlungen")
    
    # Retrieve employee information from the match results
    current_role = match_result.get('current_role', {})
    target_role = match_result.get('target_role', {})
    
    # Create a complete employee profile to access all skills
    employee_id = match_result.get('employee_id')
    if not employee_id and 'current_employee_id' in st.session_state:
        employee_id = st.session_state.current_employee_id
    
    all_employee_skills = []
    if employee_id:
        employee_data = employees_df[employees_df['Employee_ID'] == employee_id]
        if not employee_data.empty:
            employee_row = employee_data.iloc[0]
            current_kldb = employee_row.get('KldB_5_digit', '')
            current_esco_role = employee_row.get('ESCO_Role', '')
            current_manual_skills = employee_row.get('Manual_Skills', '')
            current_manual_essential_skills = employee_row.get('Manual_Essential_Skills', '')
            current_manual_optional_skills = employee_row.get('Manual_Optional_Skills', '')
            current_removed_skills = employee_row.get('Removed_Skills', '')
            
            # Create the complete employee profile
            employee_profile = create_employee_profile(
                employee_id,
                current_kldb,
                current_manual_skills,
                kldb_esco_df,
                occupation_skill_relations_df,
                skills_df,
                st.session_state.occupation_skills_mapping,
                occupations_df,
                current_esco_role,
                current_manual_essential_skills,
                current_manual_optional_skills,
                current_removed_skills
            )
            
            if employee_profile:
                all_employee_skills = employee_profile.get('skills', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mitarbeiter:**")
        st.write(f"• **Aktuelle Rolle:** {current_role.get('KldB_Label', 'N/A')}")
        st.write(f"• **Aktuelle ESCO-Rolle:** {current_role.get('ESCO_Label', 'N/A')}")
        st.write(f"• **Aktuelle Skills (gesamt):** {len(all_employee_skills)}")
        st.write(f"• **Übereinstimmende Skills (mit Zielrolle):** {len(match_result.get('matching_skills', []))}")
    
    with col2:
        st.write("**Zielrolle:**")
        st.write(f"• **Neue Rolle:** {target_role.get('KldB_Label', 'N/A')}")
        st.write(f"• **Neue ESCO-Rolle:** {target_role.get('ESCO_Label', 'N/A')}")
        st.write(f"• **Benötigte Skills:** {len(match_result.get('matching_skills', []) + match_result.get('missing_skills', []))}")
    
    st.markdown("---")
    
    # Show all of the employee's current skills
    if all_employee_skills:
        st.subheader("Aktuelle Skills des Mitarbeiters")
        st.write(f"**Anzahl aller Skills:** {len(all_employee_skills)}")
        render_skills_two_columns(all_employee_skills, left_title="Essentiell", right_title="Optional")
        st.markdown("---")

    # Show Missing Skills
    st.subheader("Fehlende Skills")

    missing_skills = match_result['missing_skills']
    st.write(f"**Anzahl fehlender Skills:** {len(missing_skills)}")

    render_skills_two_columns_table(missing_skills, left_title="Essentiell", right_title="Optional")
    
    st.markdown("---")
    
    # Extract advanced skills from the missing_skills dictionaries
    missing_skills_enhanced = []
    for skill in match_result['missing_skills']:
        # Add English labels to skills, if available
        enhanced_skill = skill.copy()
        skill_uri = skill.get('skill_uri', '')
        
        if skill_uri in st.session_state.skill_mapping_with_english:
            english_label = st.session_state.skill_mapping_with_english[skill_uri]['english']
            enhanced_skill['skill_label_english'] = english_label
            enhanced_skill['skill_labels_combined'] = f"{skill['skill_label']} | {english_label}"
        else:
            enhanced_skill['skill_label_english'] = skill['skill_label']
            enhanced_skill['skill_labels_combined'] = skill['skill_label']
        
        missing_skills_enhanced.append(enhanced_skill)
    
    # Use the advanced skills for course recommendations
    missing_skill_labels = missing_skills_enhanced
    
    # Find Course Recommendations
    with st.spinner("Suche passende Kurse..."):
        recommendations = find_udemy_courses_for_skills(
            missing_skill_labels,
            udemy_courses_df,
            top_k=3
        )
    
    if recommendations:
        st.subheader(f"Top-Kursempfehlungen für fehlende Skills")
        
        # Retrieve preferred skills from session state
        favorite_skills_uris = set()
        if 'favorite_skills' in st.session_state:
            favorite_skills_uris = st.session_state.favorite_skills
        
        # Group by Skill
        skill_groups = {}
        for rec in recommendations:
            skill_data = rec.get('skill', {})
            skill_uri = rec.get('skill_uri', '')
            
            # Extract skill names
            if isinstance(skill_data, dict):
                skill_name = skill_data.get('skill_label', str(skill_data))
                # If skill_uri isn't in rec, try retrieving it from skill_data
                if not skill_uri:
                    skill_uri = skill_data.get('skill_uri', '')
            else:
                skill_name = str(skill_data)
            
            if skill_name not in skill_groups:
                # Check whether this skill is favored
                is_fav = skill_uri in favorite_skills_uris if skill_uri else False
                skill_groups[skill_name] = {
                    'courses': [],
                    'skill_uri': skill_uri,
                    'is_favorite': is_fav
                }
            skill_groups[skill_name]['courses'].append(rec)
        
        # Set the skill type for all skills
        for skill_name, skill_data in skill_groups.items():
            skill_type = "Optional"
            for s in missing_skills:
                if isinstance(s, dict):
                    if s.get('skill_label') == skill_name:
                        skill_type = "Essential" if s.get('is_essential', False) else "Optional"
                        break
            skill_data['skill_type'] = skill_type
        
        # Sort skills: Favorites first, then Essential, then Optional; all sorted by highest course score
        def sort_skill_key(item):
            skill_name, skill_data = item
            is_favorite = skill_data['is_favorite']
            skill_type = skill_data.get('skill_type', 'Optional')
            is_essential = (skill_type == "Essential")
            
            # Sort courses within the group by score
            courses = skill_data['courses']
            courses.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Calculate the highest score
            max_score = max([c.get('similarity_score', 0) for c in courses]) if courses else 0
            
            # Sort by priority:
            # 1. Favorites first (0 = favorite, 1 = not a favorite)
            # 2. Essential before optional (but only if not a favorite)
            # 3. Highest score (negative for descending sort)
            if is_favorite:
                # Favorites: Sort by score only
                return (0, 0, -max_score)
            else:
                # Non-favorites: Essential (0) before Optional (1), then by score
                return (1, 1 if not is_essential else 0, -max_score)
        
        sorted_skill_groups = sorted(skill_groups.items(), key=sort_skill_key)
        
        for skill_name, skill_data in sorted_skill_groups:
            courses = skill_data['courses']
            is_favorite = skill_data['is_favorite']
            skill_type = skill_data.get('skill_type', 'Optional')
            
            # Show Favorite Status
            favorite_badge = " ★ FAVORIT" if is_favorite else ""
            st.write(f"**Für Skill: {skill_name}** ({skill_type}){favorite_badge}")
            
            for i, course in enumerate(courses[:3], 1):  # Top 3 Courses per Skill
                # Show multi-skill info if available
                num_matched = course.get('num_matched_skills', 1)
                matched_skills_list = course.get('matched_skills_list', [])
                multi_skill_info = ""
                if num_matched > 1:
                    multi_skill_info = f" ◉ Deckt {num_matched} Skills ab"
                
                with st.expander(f"{i}. {course['course_title']} (Score: {course['similarity_score']:.3f}){multi_skill_info}"):
                    if num_matched > 1:
                        st.info(f"◉ Dieser Kurs deckt {num_matched} fehlende Skills ab: {', '.join(matched_skills_list)}")
                    st.write(f"**Headline:** {course['course_headline']}")
                    st.write(f"**Beschreibung:** {course['course_description']}")
                    st.write(f"**Preis:** {course['course_price']}")
                    st.write(f"**Sprache:** {course['course_language']}")
                    if course.get('skill_score'):
                        st.caption(f"Relevanz für '{skill_name}': {course['skill_score']:.3f}")
                    st.markdown(f"[Zum Kurs auf Udemy]({course['course_url']})")
            
            st.markdown("---")
    else:
        st.warning("Keine passenden Kurse gefunden.")
        st.info("Tipp: Überprüfe die Udemy-Kursdaten oder versuche es mit einem anderen Zielberuf.")
    
    # Debugging Section: Show all skills and their associated courses (including those below the threshold)
    st.markdown("---")
    st.subheader("Debugging: Alle gefundenen Kurse pro Skill")
    st.info("Diese Sektion zeigt alle Kurse, die für jeden Skill gefunden wurden, auch wenn sie unter der Ähnlichkeits-Schwelle von 0.01 liegen.")
    
    # Create an advanced debugging feature
    def find_all_courses_for_skill_debug(skill, udemy_courses_df, top_k=10):
        """Findet alle Kurse für einen Skill mit Debugging-Informationen"""
        if udemy_courses_df.empty:
            return []
        
        # Prepare course data (if you haven't already)
        if 'processed_text' not in udemy_courses_df.columns:
            udemy_courses_df['processed_text'] = (
                udemy_courses_df['Title'].fillna('') + ' ' +
                udemy_courses_df['Headline'].fillna('') + ' ' +
                udemy_courses_df['Description'].fillna('')
            ).apply(preprocess_text)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        course_vectors = vectorizer.fit_transform(udemy_courses_df['processed_text'])
        
        # Prepare the skill text
        skill_text = preprocess_text(skill)
        skill_vector = vectorizer.transform([skill_text])
        
        # Calculate Similarities
        similarities = cosine_similarity(skill_vector, course_vectors).flatten()
        
        # Top-K Courses (including those below the threshold)
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        debug_recommendations = []
        for idx in top_indices:
            course = udemy_courses_df.iloc[idx]
            similarity_score = similarities[idx]
            
            # Determine status based on threshold
            status = "Empfohlen" if similarity_score > 0.01 else "Unter Schwelle"
            
            debug_recommendations.append({
                'skill': skill,
                'course_title': course['Title'],
                'course_headline': course['Headline'],
                'course_description': course['Description'][:200] + '...' if len(str(course['Description'])) > 200 else course['Description'],
                'course_url': course['URL'],
                'course_price': course['Price'],
                'course_language': course['Language'],
                'similarity_score': similarity_score,
                'status': status
            })
        
        return debug_recommendations
    
    # Show debugging information for each missing skill
    for skill in missing_skill_labels:
        # Determine the skill type based on the skill object
        if isinstance(skill, dict):
            skill_name = skill.get('skill_label', str(skill))
            skill_type = "Essential" if skill.get('is_essential', False) else "Optional"
        else:
            skill_name = str(skill)
            skill_type = "Essential" if any(s['skill_label'] == skill and s.get('is_essential', False) for s in missing_skills) else "Optional"
        
        with st.expander(f"Debug: {skill_name} ({skill_type})"):
            # Use the combined label for the course search
            if isinstance(skill, dict):
                search_skill = skill.get('skill_labels_combined', skill.get('skill_label', str(skill)))
            else:
                search_skill = str(skill)
            
            debug_courses = find_all_courses_for_skill_debug(search_skill, udemy_courses_df, top_k=10)
            
            if debug_courses:
                st.write(f"**Gefundene Kurse für '{skill_name}':**")
                
                # Show Statistics
                recommended_count = sum(1 for course in debug_courses if course['status'] == "Empfohlen")
                total_count = len(debug_courses)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Empfohlene Kurse", recommended_count)
                with col2:
                    st.metric("Gefundene Kurse (gesamt)", total_count)
                
                st.markdown("---")
                
                # Show all courses with the status
                for i, course in enumerate(debug_courses, 1):
                    status_color = "Empfohlen" if course['status'] == "Empfohlen" else "Unter Schwelle"
                    st.write(f"**{i}. {status_color} {course['course_title']} (Score: {course['similarity_score']:.4f})**")
                    st.write(f"**Status:** {course['status']}")
                    st.write(f"**Ähnlichkeits-Score:** {course['similarity_score']:.4f}")
                    st.write(f"**Schwelle:** 0.01")
                    st.write(f"**Headline:** {course['course_headline']}")
                    st.write(f"**Beschreibung:** {course['course_description']}")
                    st.write(f"**Preis:** {course['course_price']}")
                    st.write(f"**Sprache:** {course['course_language']}")
                    st.markdown(f"[Zum Kurs auf Udemy]({course['course_url']})")
                    st.markdown("---")
            else:
                st.warning(f"Keine Kurse für Skill '{skill_name}' gefunden.")
                st.info("Mögliche Gründe:")
                st.write("• Skill-Text konnte nicht verarbeitet werden")
                st.write("• Keine passenden Kurse in der Datenbank")
                st.write("• TF-IDF Vektorisierung fehlgeschlagen")

def show_overview(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df):
    st.header("Gesamtübersicht")
    
    # Initialize session state for employee data if it does not exist
    if 'employees_data' not in st.session_state:
        # Add the "Name" column if it doesn't exist
        if 'Name' not in employees_df.columns:
            employees_df['Name'] = 'Unbekannt'
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
    # Use updated employee data from session state
    employees_df = st.session_state.employees_data
    
    # Display current employee and target role information
    if 'current_employee_id' in st.session_state and not employees_df.empty:
        employee_id = st.session_state.current_employee_id
        employee_data = employees_df[employees_df['Employee_ID'] == employee_id]
        
        if not employee_data.empty:
            employee_data = employee_data.iloc[0]
            st.subheader(f"Aktueller Mitarbeiter: {employee_data.get('Name', employee_id)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Aktuelle Rolle:**")
                current_kldb = employee_data.get('KldB_5_digit', '')
                current_esco_role = employee_data.get('ESCO_Role', '')
                
                if current_esco_role:
                    st.write(f"• KldB-Code: {current_kldb}")
                    st.write(f"• ESCO-Rolle: {current_esco_role}")
                    
                    # Find the KldB role based on the code
                    kldb_match = get_unique_esco_roles(kldb_esco_df, current_kldb)
                    if not kldb_match.empty:
                        st.write(f"• KldB-Rolle: {kldb_match.iloc[0]['KldB_Label']}")
                else:
                    st.write(f"• KldB-Code: {current_kldb}")
                    st.write("• ESCO-Rolle: Nicht zugewiesen")
            
            with col2:
                st.write("**Ausgewählte Zielrolle:**")
                if 'selected_target_role' in st.session_state:
                    target_role = st.session_state.selected_target_role
                    st.write(f"• **KldB-Code:** {target_role['KldB_Code']}")
                    st.write(f"• **Berufsbezeichnung:** {target_role['KldB_Label']}")
                    
                    # Show the ESCO category only if it differs from the KldB label
                    if target_role['ESCO_Label'].lower() != target_role['KldB_Label'].lower():
                        st.write(f"• **ESCO-Kategorie:** {target_role['ESCO_Label']}")
                else:
                    st.write("• Keine Zielrolle ausgewählt")
            
            st.markdown("---")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mitarbeiter", len(employees_df))
    
    with col2:
        st.metric("ESCO-Berufe", len(get_all_esco_occupations(kldb_esco_df)))
    
    with col3:
        st.metric("KldB-ESCO Mappings", len(kldb_esco_df))
    
    with col4:
        st.metric("Udemy-Kurse", len(udemy_courses_df) if not udemy_courses_df.empty else 0)
    
    # Export Function
    st.subheader("Export")
    
    if st.button("Exportiere alle Ergebnisse als CSV", key="export_all_results_csv"):
        # The export logic could be implemented here
        st.success("Export-Funktion wird implementiert...")
    
    # Data Quality
    st.subheader("Datenqualität")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**KldB-ESCO Mapping:**")
        st.write(f"• Eindeutige KldB-Codes: {kldb_esco_df['KldB_Code'].nunique()}")
        st.write(f"• Eindeutige ESCO-Codes: {kldb_esco_df['ESCO_Code'].nunique()}")
        st.write(f"• ESCO Beruf-Skill Beziehungen: {len(occupation_skill_relations_df)}")
    
    with col2:
        st.write("**Udemy-Kurse:**")
        if not udemy_courses_df.empty:
            st.write(f"• Kurse mit Preis: {udemy_courses_df['Price'].notna().sum()}")
            st.write(f"• Sprachen: {udemy_courses_df['Language'].nunique()}")
        else:
            st.write("• Keine Udemy-Daten verfügbar")

def show_employee_management(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df):
    st.header("Mitarbeiter-Verwaltung")
    
    # Initialize Session State for Employee Data
    if 'employees_data' not in st.session_state:
        # Import employee data from a CSV file or use default data
        csv_employees = load_employees_from_csv()
        if not csv_employees.empty:
            st.session_state.employees_data = csv_employees
            st.success("Mitarbeiterdaten aus CSV geladen")
        else:
            # Add the "Name" column if it doesn't exist
            if 'Name' not in employees_df.columns:
                employees_df['Name'] = 'Unbekannt'
            st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
    # Sidebar for Navigation
    st.sidebar.subheader("Verwaltungsoptionen")
    management_option = st.sidebar.selectbox(
        "Wählen Sie eine Aktion:",
        ["Mitarbeiter anzeigen", "Neuen Mitarbeiter anlegen", "Mitarbeiter bearbeiten", "Mitarbeiter löschen"]
    )
    
    if management_option == "Mitarbeiter anzeigen":
        st.subheader("Alle Mitarbeiter")
        
        if st.session_state.employees_data.empty:
            st.info("Keine Mitarbeiter vorhanden. Legen Sie einen neuen Mitarbeiter an.")
        else:
            # Display all employees in a table
            display_columns = ['Employee_ID', 'Name', 'KldB_5_digit', 'ESCO_Role', 'Manual_Skills', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']
            available_columns = [col for col in display_columns if col in st.session_state.employees_data.columns]
            
            st.dataframe(
                st.session_state.employees_data[available_columns],
                use_container_width=True
            )
            
            # Download Feature
            csv = st.session_state.employees_data.to_csv(index=False)
            st.download_button(
                label="Mitarbeiterdaten als CSV herunterladen",
                data=csv,
                file_name="mitarbeiter_daten.csv",
                mime="text/csv"
            )
    
    elif management_option == "Neuen Mitarbeiter anlegen":
        st.subheader("Neuen Mitarbeiter anlegen")
        
        with st.form("employee_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("Vorname *", key="first_name")
            
            with col2:
                last_name = st.text_input("Nachname *", key="last_name")
            
            # Automatically generate an Employee_ID
            if first_name and last_name:
                # Create a unique ID based on the name and timestamp
                timestamp = int(time.time())
                employee_id = f"{first_name[:3].upper()}{last_name[:3].upper()}{timestamp % 10000:04d}"
                st.info(f"**Generierte Employee-ID:** {employee_id}")
            
            submitted = st.form_submit_button("Anlegen")
            
            if submitted:
                if first_name and last_name:
                    # Create a new employee
                    new_employee = {
                        'Employee_ID': employee_id,
                        'Name': f"{first_name} {last_name}",
                        'KldB_5_digit': '',
                        'Manual_Skills': '',
                        'ESCO_Role': '',
                        'Target_KldB_Code': '',
                        'Target_KldB_Label': '',
                        'Target_ESCO_Code': '',
                        'Target_ESCO_Label': '',
                        'Manual_Essential_Skills': '',
                        'Manual_Optional_Skills': '',
                        'Removed_Skills': ''
                    }
                    
                    # Add to session state
                    st.session_state.employees_data = pd.concat([
                        st.session_state.employees_data,
                        pd.DataFrame([new_employee])
                    ], ignore_index=True)
                    
                    # Save as CSV
                    if save_employees_to_csv(st.session_state.employees_data):
                        st.success(f"Mitarbeiter '{first_name} {last_name}' erfolgreich angelegt und gespeichert!")
                    else:
                        st.warning(f"Mitarbeiter angelegt, aber Speichern fehlgeschlagen!")
                    
                    # Reset Form
                    st.rerun()
                else:
                    st.error("Vorname und Nachname sind Pflichtfelder!")
    
    elif management_option == "Mitarbeiter bearbeiten":
        st.subheader("Mitarbeiter bearbeiten")
        
        if st.session_state.employees_data.empty:
            st.info("Keine Mitarbeiter zum Bearbeiten vorhanden.")
        else:
            # Select employees
            employee_options = [f"{row['Employee_ID']} - {row.get('Name', 'Unbekannt')}" for _, row in st.session_state.employees_data.iterrows()]
            selected_employee_str = st.selectbox("Mitarbeiter auswählen:", employee_options)
            
            if selected_employee_str:
                employee_id = selected_employee_str.split(" - ")[0]
                employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
                
                with st.form("edit_employee_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Extract first and last name
                        full_name = employee_data.get('Name', 'Unbekannt')
                        name_parts = full_name.split(' ', 1)
                        first_name = name_parts[0] if len(name_parts) > 0 else ""
                        last_name = name_parts[1] if len(name_parts) > 1 else ""
                        
                        new_first_name = st.text_input("Vorname *", value=first_name, key="edit_first_name")
                    
                    with col2:
                        new_last_name = st.text_input("Nachname *", value=last_name, key="edit_last_name")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Speichern")
                    with col2:
                        if st.form_submit_button("Abbrechen"):
                            st.rerun()
                    
                    if submitted:
                        if new_first_name and new_last_name:
                            # Update Employee Information
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Name'
                            ] = f"{new_first_name} {new_last_name}"
                            
                            # Save as CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Mitarbeiter '{new_first_name} {new_last_name}' erfolgreich aktualisiert und gespeichert!")
                            else:
                                st.warning(f"Mitarbeiter aktualisiert, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                        else:
                            st.error("Vorname und Nachname sind Pflichtfelder!")
    
    elif management_option == "Mitarbeiter löschen":
        st.subheader("Mitarbeiter löschen")
        
        if st.session_state.employees_data.empty:
            st.info("Keine Mitarbeiter zum Löschen vorhanden.")
        else:
            # Select employees
            employee_options = [f"{row['Employee_ID']} - {row.get('Name', 'Unbekannt')}" for _, row in st.session_state.employees_data.iterrows()]
            selected_employee_str = st.selectbox("Mitarbeiter zum Löschen auswählen:", employee_options, key="delete_employee")
            
            if selected_employee_str:
                employee_id = selected_employee_str.split(" - ")[0]
                employee_name = selected_employee_str.split(" - ")[1]
                
                st.warning(f"Sie sind dabei, den Mitarbeiter '{employee_name}' zu löschen!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Endgültig löschen", key="delete_employee_confirm", type="primary"):
                        # Delete Employee
                        st.session_state.employees_data = st.session_state.employees_data[
                            st.session_state.employees_data['Employee_ID'] != employee_id
                        ]
                        
                        # Save as CSV
                        if save_employees_to_csv(st.session_state.employees_data):
                            st.success(f"Mitarbeiter '{employee_name}' erfolgreich gelöscht und Änderungen gespeichert!")
                        else:
                            st.warning(f"Mitarbeiter gelöscht, aber Speichern fehlgeschlagen!")
                        
                        st.rerun()
                
                with col2:
                    if st.button("Abbrechen", key="cancel_delete_employee"):
                        st.rerun()
    
    # Update the global `employees_df` for other sections
    if 'employees_data' in st.session_state:
        globals()['employees_df'] = st.session_state.employees_data

def show_strategic_development(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, archi_data, udemy_courses_df):
    st.header("Strategische Weiterbildung ")
    st.info("**Neue Funktionalität:** Diese Sektion nutzt XML-Daten aus Archi, um strategische Weiterbildungsempfehlungen basierend auf Geschäftsmodellen zu generieren.")
    
    # Display debug information
    st.write("**Debug-Informationen:**")
    st.write(f"• Archi-Daten verfügbar: {archi_data is not None}")
    if archi_data:
        st.write(f"• Capabilities gefunden: {len(archi_data.get('capabilities', []))}")
        st.write(f"• Resources gefunden: {len(archi_data.get('resources', []))}")
        st.write(f"• Beziehungen gefunden: {len(archi_data.get('relationships', []))}")
    
    # Use updated employee data from session state
    employees_df = st.session_state.employees_data
    
    if employees_df.empty:
        st.warning("Keine Mitarbeiterdaten gefunden.")
        return
    
    # Use the global employee selection
    if 'current_employee_id' not in st.session_state:
        st.info("Bitte wählen Sie einen Mitarbeiter in der Sidebar aus.")
        return
    
    employee_id = st.session_state.current_employee_id
    employee_data = employees_df[employees_df['Employee_ID'] == employee_id].iloc[0]
    
    st.subheader(f"Strategische Weiterbildung für {employee_data.get('Name', employee_id)}")
    
    # Check whether Archi data is available
    if not archi_data:
        st.error("Keine Archi XML-Daten verfügbar. Bitte stellen Sie sicher, dass die DigiVan.xml Datei im data-Ordner liegt.")
        st.write("**Mögliche Ursachen:**")
        st.write("• Die DigiVan.xml Datei fehlt im data-Ordner")
        st.write("• Die XML-Datei konnte nicht geparst werden")
        st.write("• Ein Fehler ist beim Laden der Daten aufgetreten")
        
        # Try loading the XML file manually
        xml_path = data_path('DigiVan.xml')
        if os.path.exists(xml_path):
            
            # Try manual parsing
            if st.button("XML-Datei manuell neu laden", key="reload_xml_manual"):
                try:
                    manual_archi_data = parse_archi_xml(xml_path)
                    if manual_archi_data:
                        st.session_state.archi_data = manual_archi_data
                        st.success("XML-Datei erfolgreich geladen!")
                        st.rerun()
                    else:
                        st.error("Manuelles Laden fehlgeschlagen")
                except Exception as e:
                    st.error(f"Fehler beim manuellen Laden: {str(e)}")
        else:
            st.error(f"XML-Datei nicht gefunden: {xml_path}")
        
        return
    
    # Employee's Current Role
    current_kldb = employee_data.get('KldB_5_digit', '')
    current_manual_skills = employee_data.get('Manual_Skills', '')
    current_esco_role = employee_data.get('ESCO_Role', '')
    
    # Retrieve additional skill data from Session State
    current_manual_essential_skills = employee_data.get('Manual_Essential_Skills', '')
    current_manual_optional_skills = employee_data.get('Manual_Optional_Skills', '')
    current_removed_skills = employee_data.get('Removed_Skills', '')
    
    if not current_kldb:
        st.info("**Keine aktuelle Rolle zugewiesen.** Bitte weisen Sie zuerst eine Rolle in 'Mitarbeiter-Kompetenzprofile' zu.")
        return
    
    # Create the current employee profile
    current_profile = create_employee_profile(
        employee_id,
        current_kldb,
        current_manual_skills,
        kldb_esco_df,
        occupation_skill_relations_df,
        skills_df,
        st.session_state.occupation_skills_mapping,
        occupations_df,
        current_esco_role,
        current_manual_essential_skills,
        current_manual_optional_skills,
        current_removed_skills
    )
    
    if not current_profile:
        st.error("Konnte kein aktuelles Mitarbeiterprofil erstellen.")
        return
    
    # Identify the skills needed in the future based on the capabilities
    future_skills = extract_future_skills_from_capabilities(archi_data)
    
    if not future_skills:
        st.warning("Keine zukünftig benötigten Skills aus den Capabilities gefunden.")
        return
    
    # Show current and future skills
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aktuelle Skills:**")
        current_skills = current_profile['skills']
        st.write(f"• Anzahl: {len(current_skills)}")
        render_skills_two_columns(current_skills, left_title="Essentiell", right_title="Optional")
    
    with col2:
        st.write("**Zukünftig benötigte Skills (aus Capabilities):**")
        st.write(f"• Anzahl: {len(future_skills)}")
    
    st.markdown("---")
    
    # Strategic Competency Alignment
    st.subheader("Strategischer Kompetenzabgleich")
    
    # Compare current skills with skills that will be needed in the future
    current_skill_labels = [skill['skill_label'].lower() for skill in current_skills]
    future_skill_labels = [skill['skill_name'].lower() for skill in future_skills]
    
    # Find matching skills
    matching_skills = []
    missing_skills = []
    
    for future_skill in future_skills:
        future_label = future_skill['skill_name'].lower()
        
        # Search for exact matches
        exact_match = None
        for current_skill in current_skills:
            if current_skill['skill_label'].lower() == future_label:
                exact_match = current_skill
                break
        
        if exact_match:
            matching_skills.append({
                'future_skill': future_skill,
                'current_skill': exact_match,
                'match_type': 'exakt'
            })
        else:
            # Search for Similar Skills (Semantic Matching)
            similar_skill = None
            for current_skill in current_skills:
                if any(word in current_skill['skill_label'].lower() for word in future_label.split()):
                    similar_skill = current_skill
                    break
            
            if similar_skill:
                matching_skills.append({
                    'future_skill': future_skill,
                    'current_skill': similar_skill,
                    'match_type': 'ähnlich'
                })
            else:
                missing_skills.append(future_skill)
    
    # Show results
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Übereinstimmende Skills:**")
        st.write(f"• Anzahl: {len(matching_skills)}")
        
        if matching_skills:
            for match in matching_skills[:5]:
                match_type_text = "Exakt" if match['match_type'] == 'exakt' else "Ähnlich"
                st.write(f"[{match_type_text}] {match['future_skill']['skill_name']}")
                if match['match_type'] == 'ähnlich':
                    st.write(f"  → Ähnlich zu: {match['current_skill']['skill_label']}")
            if len(matching_skills) > 5:
                st.write(f"  ... und {len(matching_skills) - 5} weitere")
    
    with col2:
        st.write("**Fehlende Skills für digitale Transformation:**")
        st.write(f"• Anzahl: {len(missing_skills)}")
        
        if missing_skills:
            for skill in missing_skills[:5]:
                st.write(f"Fehlt: {skill['skill_name']}")
            if len(missing_skills) > 5:
                st.write(f"  ... und {len(missing_skills) - 5} weitere")
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.subheader("Strategische Empfehlungen")
    
    if missing_skills:
        st.write("**Prioritäre Weiterbildungsbereiche:**")
        
        # Group missing skills by category
        skill_categories = {}
        for skill in missing_skills:
            skill_name = skill['skill_name'].lower()
            
            # Simple categorization based on keywords
            if any(word in skill_name for word in ['data', 'analytics', 'mining', 'visualization']):
                category = "Datenanalyse & Business Intelligence"
            elif any(word in skill_name for word in ['python', 'sql', 'programming', 'coding']):
                category = "Programmierung & Technische Skills"
            elif any(word in skill_name for word in ['customer', 'service', 'communication', 'relationship']):
                category = "Kundenorientierung & Kommunikation"
            elif any(word in skill_name for word in ['maintenance', 'equipment', 'troubleshoot', 'technical']):
                category = "Technische Wartung & Problemlösung"
            else:
                category = "Sonstige Skills"
            
            if category not in skill_categories:
                skill_categories[category] = []
            skill_categories[category].append(skill)
        
        # Show categorized recommendations
        for category, skills in skill_categories.items():
            with st.expander(f"{category} ({len(skills)} Skills)"):
                for skill in skills:
                    st.write(f"• {skill['skill_name']}")
                
                # Generate semantic course recommendations for this category
                st.write("**Semantische Kursempfehlungen von Udemy:**")
                
                # Create skill objects for course recommendations
                skill_objects = []
                for skill in skills:
                    skill_objects.append({
                        'skill_label': skill['skill_name'],
                        'skill_uri': f"strategic_{skill['skill_name'].lower().replace(' ', '_')}",
                        'is_essential': True
                    })
                
                # Find course recommendations for these skills
                with st.spinner(f"Suche Kurse für {category}..."):
                    try:
                        # Use the provided Udemy data
                        if udemy_courses_df is not None and not udemy_courses_df.empty:
                            
                            # Find the right courses
                            recommendations = find_udemy_courses_for_skills(
                                skill_objects,
                                udemy_courses_df,
                                top_k=3  # Show the Top 3 Courses per Category
                            )
                            
                            if recommendations:
                                st.write(f"**Top {len(recommendations)} Kurse gefunden:**")
                                
                                for i, course in enumerate(recommendations[:3], 1):
                                    with st.container():
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.write(f"**{i}. {course['course_title']}**")
                                            if course.get('course_headline'):
                                                st.write(f"*{course['course_headline']}*")
                                            if course.get('course_description'):
                                                # Shorten the description
                                                desc = course['course_description'][:200] + "..." if len(course['course_description']) > 200 else course['course_description']
                                                st.write(f"*{desc}*")
                                            
                                            # Show relevant skills
                                            if course.get('skill'):
                                                st.write("**Relevanter Skill:**")
                                                st.write(f"• {course['skill']}")
                                            
                                            # Show Similarity Score
                                            if course.get('similarity_score'):
                                                st.write(f"**Relevanz:** {course['similarity_score']:.2f}")
                                        
                                        with col2:
                                            if course.get('course_price'):
                                                st.write(f"**Preis:** {course['course_price']}")
                                            if course.get('course_language'):
                                                st.write(f"**Sprache:** {course['course_language']}")
                                            if course.get('course_url'):
                                                st.write(f"[Kurs öffnen]({course['course_url']})")
                                        
                                        st.markdown("---")
                            else:
                                st.info("Keine spezifischen Kurse für diese Kategorie gefunden.")
                                st.write("**Allgemeine Empfehlungen:**")
                                
                                # Fallback recommendations based on category
                                if category == "Datenanalyse & Business Intelligence":
                                    st.write("• Kurse in Data Analytics und Business Intelligence")
                                    st.write("• Schulungen in Datenvisualisierung")
                                    st.write("• Workshops zu datengetriebener Entscheidungsfindung")
                                elif category == "Programmierung & Technische Skills":
                                    st.write("• Python-Programmierkurse")
                                    st.write("• SQL-Datenbankkurse")
                                    st.write("• Einführung in maschinelles Lernen")
                                elif category == "Kundenorientierung & Kommunikation":
                                    st.write("• Kundenservice-Schulungen")
                                    st.write("• Kommunikationstraining")
                                    st.write("• Beziehungsmanagement")
                                elif category == "Technische Wartung & Problemlösung":
                                    st.write("• Wartungsverfahren und -standards")
                                    st.write("• Problemlösungstechniken")
                                    st.write("• Technische Dokumentation")
                                else:
                                    st.write("• Allgemeine Weiterbildungsmaßnahmen")
                                    st.write("• Spezifische Schulungen je nach Skill")
                                    st.write("• On-the-Job Training")
                        else:
                            st.warning("Udemy-Kursdaten nicht verfügbar. Verwende allgemeine Empfehlungen.")
                            # Fallback Recommendations
                            if category == "Datenanalyse & Business Intelligence":
                                st.write("• Kurse in Data Analytics und Business Intelligence")
                                st.write("• Schulungen in Datenvisualisierung")
                                st.write("• Workshops zu datengetriebener Entscheidungsfindung")
                            elif category == "Programmierung & Technische Skills":
                                st.write("• Python-Programmierkurse")
                                st.write("• SQL-Datenbankkurse")
                                st.write("• Einführung in maschinelles Lernen")
                            elif category == "Kundenorientierung & Kommunikation":
                                st.write("• Kundenservice-Schulungen")
                                st.write("• Kommunikationstraining")
                                st.write("• Beziehungsmanagement")
                            elif category == "Technische Wartung & Problemlösung":
                                st.write("• Wartungsverfahren und -standards")
                                st.write("• Problemlösungstechniken")
                                st.write("• Technische Dokumentation")
                            else:
                                st.write("• Allgemeine Weiterbildungsmaßnahmen")
                                st.write("• Spezifische Schulungen je nach Skill")
                                st.write("• On-the-Job Training")
                                
                    except Exception as e:
                        st.error(f"Fehler bei der Kursempfehlung: {str(e)}")
                        st.write("**Verwende allgemeine Empfehlungen:**")
                        if category == "Datenanalyse & Business Intelligence":
                            st.write("• Kurse in Data Analytics und Business Intelligence")
                            st.write("• Schulungen in Datenvisualisierung")
                            st.write("• Workshops zu datengetriebener Entscheidungsfindung")
                        elif category == "Programmierung & Technische Skills":
                            st.write("• Python-Programmierkurse")
                            st.write("• SQL-Datenbankkurse")
                            st.write("• Einführung in maschinelles Lernen")
                        elif category == "Kundenorientierung & Kommunikation":
                            st.write("• Kundenservice-Schulungen")
                            st.write("• Kommunikationstraining")
                            st.write("• Beziehungsmanagement")
                        elif category == "Technische Wartung & Problemlösung":
                            st.write("• Wartungsverfahren und -standards")
                            st.write("• Problemlösungstechniken")
                            st.write("• Technische Dokumentation")
                        else:
                            st.write("• Allgemeine Weiterbildungsmaßnahmen")
                            st.write("• Spezifische Schulungen je nach Skill")
                            st.write("• On-the-Job Training")
    else:
        st.success("**Alle zukünftig benötigten Skills sind bereits vorhanden!**")
        st.write("Der Mitarbeiter ist bereits gut auf die digitale Transformation vorbereitet.")
    
    # Export the Strategic Analysis
    st.markdown("---")
    st.subheader("Export der strategischen Analyse")
    
    if st.button("Strategische Analyse als CSV exportieren", key="export_strategic_analysis_csv"):
        # Create a DataFrame for export
        analysis_data = []
        
        for skill in future_skills:
            skill_name = skill['skill_name']
            status = "Vorhanden" if any(s['skill_name'].lower() == skill_name.lower() for s in matching_skills) else "Fehlt"
            match_type = next((m['match_type'] for m in matching_skills if m['future_skill']['skill_name'].lower() == skill_name.lower()), "Kein Match")
            
            analysis_data.append({
                'Zukünftig_benötigter_Skill': skill_name,
                'Status': status,
                'Match_Typ': match_type,
                'Quelle': 'Capability Map'
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # CSV-Download
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"strategische_analyse_{employee_data.get('Name', employee_id)}_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def show_xml_based_competency_analysis(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df, udemy_courses_df, berufsbenennungen_df):
    """Zeigt die XML-basierte Kompetenzabgleich-Funktionalität"""
    st.header("XML-basierte Kompetenzabgleich ")
    st.info("**Neue Funktionalität:** Diese Sektion nutzt XML-Daten aus Archi, um automatisch IST-Rollen und SOLL-Skills zu extrahieren und Mitarbeiterprofile zu erstellen.")
    
    # Load both XML files
    kompetenzabgleich_path = data_path("Kompetenzabgleich.xml")
    digivan_path = data_path("DigiVan.xml")
    
    if not os.path.exists(kompetenzabgleich_path):
        # File not found - do not display an error message
        return
    
    if not os.path.exists(digivan_path):
        st.error(f"DigiVan XML-Datei nicht gefunden: {digivan_path}")
        return
    
    # Parse both XML files
    with st.spinner("Lade und parse XML-Dateien..."):
        kompetenzabgleich_data = parse_kompetenzabgleich_xml(kompetenzabgleich_path)
        digivan_data = parse_archi_xml(digivan_path)
    
    if not kompetenzabgleich_data['success']:
        st.error("Fehler beim Parsen der Kompetenzabgleich.xml")
        return
    
    # Check whether DigiVan data was loaded successfully
    if not digivan_data or 'capabilities' not in digivan_data:
        st.error("Fehler beim Parsen der DigiVan.xml")
        return
    
    # Current roles (Ist-Rollen) from Kompetenzabgleich.xml
    ist_rollen = kompetenzabgleich_data['ist_rollen']
    
    # Future Capabilities (Soll-Skills) from DigiVan.xml
    soll_skills = digivan_data['capabilities']
    
    st.success(f"Beide XML-Dateien erfolgreich geladen!")
    st.info(f"""
    **XML-Daten:**
    - **IST-Rollen** (Kompetenzabgleich.xml): {len(ist_rollen)} BusinessActor
    - **SOLL-Skills** (DigiVan.xml): {len(soll_skills)} Capabilities
    """)
    
    # Tabs for various functions
    tab1, tab2 = st.tabs(["IST-Rollen aus XML", "SOLL-Skills zu Jobs"])
    
    with tab1:
        st.subheader("IST-Rollen aus XML (BusinessActor)")
        
        if ist_rollen:
            st.write(f"**Gefundene IST-Rollen:** {len(ist_rollen)}")
            
            # Table of Current Roles
            ist_data = []
            for rolle in ist_rollen:
                ist_data.append({
                    'Rollenname': rolle['name'],
                    'Typ': rolle['type'],
                    'ID': rolle.get('id', rolle.get('identifier', 'N/A'))
                })
            
            ist_df = pd.DataFrame(ist_data)
            st.dataframe(ist_df, use_container_width=True)
            
            # New Feature: Match Actual Roles with KldB Roles and Assign ESCO Roles
            st.markdown("---")
            st.subheader("IST-Rollen mit KldB-Rollen matchen und ESCO-Rollen zuweisen")
            
            st.write("**Automatisches Matching der gefundenen IST-Rollen mit passenden KldB-Rollen:**")
            
            # Create a table listing current roles and matched KldB roles
            ist_kldb_matches = []
            
            for rolle in ist_rollen:
                rolle_name = rolle['name']
                
                # Attempt automatic matching using KldB roles
                kldb_code, kldb_label = find_kldb_code_for_job_title(
                    rolle_name, occupations_df, kldb_esco_df, berufsbenennungen_df
                )
                
                ist_kldb_matches.append({
                    'IST-Rolle': rolle_name,
                    'Gematcher KldB-Code': kldb_code if kldb_code else 'Nicht gefunden',
                    'Gematcher KldB-Rolle': kldb_label if kldb_label else 'Nicht gefunden',
                    'Matching-Status': 'Gefunden' if kldb_code else 'Nicht gefunden'
                })
            
            # Show matching results
            st.write("**Matching-Ergebnisse:**")
            matches_df = pd.DataFrame(ist_kldb_matches)
            st.dataframe(matches_df, use_container_width=True)
            
            # Select a matched current role for ESCO role assignment
            st.markdown("---")
            st.subheader("ESCO-Rollen für gematchte IST-Rollen auswählen")
            
            # Filter only successfully matched roles
            successful_matches = [match for match in ist_kldb_matches if match['Gematcher KldB-Code'] != 'Nicht gefunden']
            
            if successful_matches:
                st.success(f"{len(successful_matches)} IST-Rollen erfolgreich mit KldB-Rollen gematcht!")
                
                # Dropdown for successfully matched roles
                match_options = [f"{match['IST-Rolle']} → {match['Gematcher KldB-Rolle']} ({match['Gematcher KldB-Code']})" for match in successful_matches]
                match_options.insert(0, "Bitte wählen Sie eine gematchte IST-Rolle...")
                
                selected_match = st.selectbox(
                    "Wählen Sie eine gematchte IST-Rolle für ESCO-Rollen-Zuweisung:",
                    match_options,
                    key="xml_match_select"
                )
                
                if selected_match and selected_match != "Bitte wählen Sie eine gematchte IST-Rolle...":
                    # Extract KldB code from the selection
                    selected_kldb_code = selected_match.split("(")[1].split(")")[0]
                    
                    # Find all ESCO roles associated with this KldB role
                    matching_esco_roles = get_unique_esco_roles(kldb_esco_df, selected_kldb_code)
                    
                    if not matching_esco_roles.empty:
                        st.write(f"**Verfügbare ESCO-Rollen für KldB-Code '{selected_kldb_code}':**")
                        
                        # Dropdown menu for selecting ESCO roles
                        esco_role_options = [f"{role['ESCO_Label']}" for _, role in matching_esco_roles.iterrows()]
                        selected_esco_role = st.selectbox(
                            "Wählen Sie eine ESCO-Rolle:",
                            ["Bitte wählen Sie eine ESCO-Rolle..."] + esco_role_options,
                            key=f"esco_role_select_{selected_kldb_code}"
                        )
                        
                        if selected_esco_role and selected_esco_role != "Bitte wählen Sie eine ESCO-Rolle...":
                            # Extract the ESCO label and find the corresponding code
                            esco_label = selected_esco_role
                            # Find the ESCO code based on the label
                            matching_role = matching_esco_roles[matching_esco_roles['ESCO_Label'] == esco_label]
                            esco_code = matching_role.iloc[0]['ESCO_Code'] if not matching_role.empty else ""
                            
                            # Acquire the skills needed for this ESCO role
                            role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                            
                            st.write(f"**Ausgewählte ESCO-Rolle:** {esco_label}")
                            
                            if role_skills:
                                # Key to Skill Color Points
                                st.markdown("**Skill-Legende:**")
                                legend_col1, legend_col2 = st.columns(2)
                                with legend_col1:
                                    st.write("**Essential Skills** - Unverzichtbare Skills")
                                with legend_col2:
                                    st.write("**Optional Skills** - Hilfreiche Skills")
                                
                                st.markdown("---")
                                
                                st.write("**Skills:**")
                                render_skills_two_columns_table(role_skills, left_title="Essentiell", right_title="Optional")
                                
                                # Apply button
                                if st.button(f"Als aktuelle Rolle übernehmen", key=f"xml_assign_esco_{esco_code}"):
                                    # Update the KldB code in the session state data
                                    if 'current_employee_id' in st.session_state:
                                        employee_id = st.session_state.current_employee_id
                                        st.session_state.employees_data.loc[
                                            st.session_state.employees_data['Employee_ID'] == employee_id, 
                                            'KldB_5_digit'
                                        ] = selected_kldb_code
                                        st.session_state.employees_data.loc[
                                            st.session_state.employees_data['Employee_ID'] == employee_id, 
                                            'ESCO_Role'
                                        ] = esco_label # Save the ESCO role
                                        
                                        # Save as CSV
                                        if save_employees_to_csv(st.session_state.employees_data):
                                            st.success(f"Rolle '{esco_label}' wurde als aktuelle Rolle zugewiesen und gespeichert!")
                                        else:
                                            st.warning(f"Rolle zugewiesen, aber Speichern fehlgeschlagen!")
                                        
                                        st.rerun()
                                    else:
                                        st.warning("Bitte wählen Sie zuerst einen Mitarbeiter in der Sidebar aus.")
                            else:
                                st.write("Keine Skills für diese Rolle gefunden.")
                        else:
                            st.info("Wählen Sie eine ESCO-Rolle aus, um die Details zu sehen.")
                    else:
                        st.warning(f"Keine ESCO-Rollen für KldB-Code '{selected_kldb_code}' gefunden.")
            else:
                st.warning("Keine IST-Rollen konnten erfolgreich mit KldB-Rollen gematcht werden.")
                st.info("Tipp: Überprüfen Sie die Schreibweise der Rollennamen oder fügen Sie manuell KldB-Codes hinzu.")
            
                        # Manuelle KldB-Zuordnung für nicht gematchte Rollen
            if any(match['Matching-Status'] == 'Nicht gefunden' for match in ist_kldb_matches):
                st.markdown("---")
                st.subheader("Manuelle KldB-Zuordnung für nicht gematchte Rollen")
                
                # Show Unmatched Roles
                unmatched_roles = [match for match in ist_kldb_matches if match['Matching-Status'] == 'Nicht gefunden']
                st.write(f"**Nicht gematchte Rollen:** {len(unmatched_roles)}")
                
                # Drop-down menu for role selection
                selected_unmatched_role = st.selectbox(
                    "Wählen Sie eine nicht gematchte Rolle:",
                    [f"{unmatched['IST-Rolle']}" for unmatched in unmatched_roles],
                    index=0,
                    key="unmatched_role_select"
                )
                
                if selected_unmatched_role:
                    # Find the selected role
                    selected_role_data = next((unmatched for unmatched in unmatched_roles if unmatched['IST-Rolle'] == selected_unmatched_role), None)
                    
                    if selected_role_data:
                        st.write(f"**Ausgewählte IST-Rolle:** {selected_role_data['IST-Rolle']}")
                        
                        # Drop-down menu for selecting KldB codes, with codes and descriptions (no duplicates)
                        available_kldb_options = []
                        seen_combinations = set()  # Prevents duplicates
                        
                        for _, row in kldb_esco_df.iterrows():
                            kldb_code = str(row.get('KldB_Code', ''))
                            kldb_label = str(row.get('KldB_Label', ''))
                            if kldb_code and kldb_label and not pd.isna(kldb_code) and not pd.isna(kldb_label):
                                # Shorten long labels for better readability
                                display_label = kldb_label
                                if len(display_label) > 50:
                                    display_label = display_label[:47] + "..."
                                option = f"{display_label} | {kldb_code}"
                                
                                # Add only if the combination does not already exist
                                if option not in seen_combinations:
                                    available_kldb_options.append(option)
                                    seen_combinations.add(option)
                        
                        # Sort by name for better clarity
                        available_kldb_options = sorted(available_kldb_options)
                        
                        selected_kldb_option = st.selectbox(
                            f"Wählen Sie einen KldB-Code für '{selected_role_data['IST-Rolle']}':",
                            ["Bitte wählen Sie einen KldB-Code..."] + available_kldb_options,
                            key=f"kldb_code_select_{selected_role_data['IST-Rolle']}"
                        )
                        
                        # Extract the KldB code from the selected option
                        selected_kldb_code = None
                        if selected_kldb_option and selected_kldb_option != "Bitte wählen Sie einen KldB-Code...":
                            selected_kldb_code = selected_kldb_option.split(" | ")[1]
                        
                        if selected_kldb_code and selected_kldb_code != "Bitte wählen Sie einen KldB-Code...":
                            # Find related ESCO roles
                            manual_esco_roles = get_unique_esco_roles(kldb_esco_df, selected_kldb_code)
                            
                            if not manual_esco_roles.empty:
                                st.success(f"KldB-Code '{selected_kldb_code}' gefunden!")
                                st.write(f"**Verfügbare ESCO-Rollen:**")
                                
                                # Dropdown menu for selecting ESCO roles
                                esco_role_options = [f"{role['ESCO_Label']}" for _, role in manual_esco_roles.iterrows()]
                                selected_esco_role = st.selectbox(
                                    "Wählen Sie eine ESCO-Rolle:",
                                    ["Bitte wählen Sie eine ESCO-Rolle..."] + esco_role_options,
                                    key=f"esco_role_select_{selected_kldb_code}"
                                )
                                
                                if selected_esco_role and selected_esco_role != "Bitte wählen Sie eine ESCO-Rolle...":
                                    # Extract the ESCO label and find the corresponding code
                                    esco_label = selected_esco_role
                                    # Find the ESCO code based on the label
                                    matching_role = manual_esco_roles[manual_esco_roles['ESCO_Label'] == esco_label]
                                    esco_code = matching_role.iloc[0]['ESCO_Code'] if not matching_role.empty else ""
                                    
                                    # Acquire the skills needed for this ESCO role
                                    role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                                    
                                    if role_skills:
                                        st.write(f"**Skills für {esco_label}:**")
                                        render_skills_two_columns_table(role_skills, left_title="Essentiell", right_title="Optional")
                                        
                                        # Apply button
                                        if st.button(f"Als aktuelle Rolle übernehmen", key=f"manual_assign_esco_{selected_kldb_code}_{esco_code}"):
                                            if 'current_employee_id' in st.session_state:
                                                employee_id = st.session_state.current_employee_id
                                                st.session_state.employees_data.loc[
                                                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                                                    'KldB_5_digit'
                                                ] = selected_kldb_code
                                                st.session_state.employees_data.loc[
                                                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                                                    'ESCO_Role'
                                                ] = esco_label
                                                
                                                # Save the selection to the session state for Tab 2
                                                st.session_state.manual_selection_tab1 = f"{selected_role_data['IST-Rolle']} → {esco_label} ({selected_kldb_code})"
                                                st.session_state.selected_kldb_code_tab2 = selected_kldb_code
                                                st.session_state.selected_esco_role_tab2 = esco_label
                                                st.session_state.selected_role_data_name = selected_role_data['IST-Rolle']
                                                
                                                if save_employees_to_csv(st.session_state.employees_data):
                                                    st.success(f"Rolle '{esco_label}' wurde als aktuelle Rolle zugewiesen und gespeichert!")
                                                    st.info("Wechseln Sie jetzt zu Tab 'SOLL Skills zu Jobs' um den Kompetenzabgleich durchzuführen!")
                                                else:
                                                    st.warning(f"Rolle zugewiesen, aber Speichern fehlgeschlagen!")
                                                
                                                st.rerun()
                                            else:
                                                st.warning("Bitte wählen Sie zuerst einen Mitarbeiter in der Sidebar aus.")
                                    else:
                                        st.write("Keine Skills für diese Rolle gefunden.")
                            else:
                                st.warning(f"Kein ESCO-Code für KldB-Code '{selected_kldb_code}' gefunden.")
            
            # Option to select a role for automatic job assignment
            st.markdown("---")
            st.subheader("Automatische Job-Zuordnung testen")
            
            selected_role = st.selectbox(
                "Wählen Sie eine Rolle für den Test:",
                [rolle['name'] for rolle in ist_rollen],
                index=0,
                key="xml_role_test_select"
            )
            
            if st.button("Job-Zuordnung finden", key="job_zuordnung_test_tab1"):
                with st.spinner("Suche passenden KldB-Code..."):
                    kldb_code, kldb_label = find_kldb_code_for_job_title(
                        selected_role, occupations_df, kldb_esco_df, berufsbenennungen_df
                    )
                
                if kldb_code and kldb_label:
                    st.success(f"Job-Zuordnung gefunden!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Rollenname:** {selected_role}")
                        st.write(f"**KldB-Code:** {kldb_code}")
                    with col2:
                        st.write(f"**KldB-Rolle:** {kldb_label}")
                        
                        # Show Related Skills
                        if kldb_code:
                            kldb_match = get_unique_esco_roles(kldb_esco_df, kldb_code)
                            if not kldb_match.empty:
                                esco_uri = kldb_match.iloc[0].get('ESCO_Code', '')
                                if esco_uri:
                                    role_skills = occupation_skill_relations_df[
                                        occupation_skill_relations_df['occupationUri'] == esco_uri
                                    ]
                                    if not role_skills.empty:
                                        st.write(f"**Verfügbare Skills:** {len(role_skills)}")
                else:
                    st.warning("Keine passende Job-Zuordnung gefunden.")
                    st.info("Tipp: Überprüfen Sie die Schreibweise oder fügen Sie manuell einen KldB-Code hinzu.")
        else:
            st.warning("Keine IST-Rollen in der XML gefunden.")
    
    with tab2:
        st.subheader("SOLL-Skills zu Jobs - Kompetenzabgleich & Kursempfehlungen")
        
        if soll_skills:
            st.write(f"**Zukünftig benötigte Skills (aus Capabilities):** {len(soll_skills)}")
            
            # Check whether an current (Ist) role has been selected from Tab 1
            if ('xml_match_select' not in st.session_state or 
                not st.session_state.get('xml_match_select') or 
                st.session_state.xml_match_select == "Bitte wählen Sie eine gematchte IST-Rolle...") and \
               ('selected_kldb_code_tab2' not in st.session_state or 
                'selected_esco_role_tab2' not in st.session_state):
                
                st.info("**Bitte wählen Sie zuerst eine IST-Rolle im Tab 'IST-Rollen aus XML' aus, um den Kompetenzabgleich durchzuführen.**")
                st.write("**Verfügbare SOLL-Skills (Capabilities):**")
                
                # Table of future (Soll) skills
                soll_data = []
                for skill in soll_skills:
                    soll_data.append({
                        'Capability': skill['name'],
                        'Typ': skill['type'],
                        'ID': skill.get('id', skill.get('identifier', 'N/A'))
                    })
                
                soll_df = pd.DataFrame(soll_data)
                st.dataframe(soll_df, use_container_width=True)
                return
            
            # Retrieve the selected current role from Tab 1 or the manual selection
            selected_match = st.session_state.get('xml_match_select', '')
            manual_selection = st.session_state.get('manual_selection_tab1', '')
            selected_kldb_code = None
            selected_esco_role = None
            
            # Check whether a manual selection has been made on Tab 1
            if 'selected_kldb_code_tab2' in st.session_state and 'selected_esco_role_tab2' in st.session_state:
                selected_kldb_code = st.session_state.selected_kldb_code_tab2
                selected_esco_role = st.session_state.selected_esco_role_tab2
                st.success(f"**Ausgewählte IST-Rolle:** {st.session_state.get('selected_role_data_name', 'Manuell ausgewählte Rolle')} → {selected_esco_role}")
            elif manual_selection:
                # Use the manual selection from Tab 1
                selected_kldb_code = manual_selection.split("(")[1].split(")")[0]
                selected_esco_role = manual_selection.split(" → ")[1].split(" (")[0] if " → " in manual_selection else ""
                st.success(f"**Ausgewählte IST-Rolle:** {manual_selection}")
            elif selected_match and selected_match != "Bitte wählen Sie eine gematchte IST-Rolle...":
                # Extract KldB code from the original selection
                selected_kldb_code = selected_match.split("(")[1].split(")")[0]
                selected_esco_role = selected_match.split(" → ")[1].split(" (")[0] if " → " in selected_match else ""
                st.success(f"**Ausgewählte IST-Rolle:** {selected_match}")
            else:
                st.warning("Keine IST-Rolle ausgewählt. Bitte wechseln Sie zu Tab 1 und wählen Sie eine Rolle aus.")
                return
            
            if selected_kldb_code:
                # Create the current employee profile based on the selected current role
                if 'current_employee_id' in st.session_state:
                    employee_id = st.session_state.current_employee_id
                    current_employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
                    
                    # Use the selected KldB role and ESCO role
                    current_kldb = selected_kldb_code
                    current_esco_role = selected_esco_role or current_employee_data.get('ESCO_Role', '')
                    current_manual_skills = current_employee_data.get('Manual_Skills', '')
                    current_manual_essential_skills = current_employee_data.get('Manual_Essential_Skills', '')
                    current_manual_optional_skills = current_employee_data.get('Manual_Optional_Skills', '')
                    current_removed_skills = current_employee_data.get('Removed_Skills', '')
                    
                    # Create the current employee profile
                    current_profile = create_employee_profile(
                        employee_id,
                        current_kldb,
                        current_manual_skills,
                        kldb_esco_df,
                        occupation_skill_relations_df,
                        skills_df,
                        st.session_state.occupation_skills_mapping,
                        occupations_df,
                        current_esco_role,
                        current_manual_essential_skills,
                        current_manual_optional_skills,
                        current_removed_skills
                    )
                    
                    if current_profile:
                        st.markdown("---")
                        st.subheader("Kompetenzabgleich: IST vs. SOLL")
                        
                        # Current skills for the current role
                        current_skills = current_profile['skills']
                        current_skill_labels = [skill['skill_label'].lower() for skill in current_skills]
                        
                        # Target Skills (Capabilities) - Use the same data source as in Strategic Professional Development
                        future_skills_from_archi = extract_future_skills_from_capabilities(st.session_state.archi_data)
                        soll_skill_names = [skill['skill_name'].lower() for skill in future_skills_from_archi]
                        
                        # Calculate matches and missing skills
                        matching_skills = []
                        missing_skills = []
                        
                        for soll_skill in future_skills_from_archi:
                            soll_name = soll_skill['skill_name'].lower()
                            if soll_name in current_skill_labels:
                                # Find the corresponding current skill
                                current_skill = next((s for s in current_skills if s['skill_label'].lower() == soll_name), None)
                                if current_skill:
                                    matching_skills.append({
                                        'skill_name': soll_skill['skill_name'],
                                        'current_skill': current_skill,
                                        'is_essential': current_skill.get('is_essential', False)
                                    })
                            else:
                                missing_skills.append(soll_skill)
                        
                        # NEW SECTION: View Assigned Skills
                        st.markdown("---")
                        st.subheader("🔧 Zugewiesene Skills des aktuellen Mitarbeiters")
                        
                        if current_skills:
                            st.success(f"**Anzahl zugewiesener Skills:** {len(current_skills)}")
                            
                            # Group skills by type
                            essential_skills = [s for s in current_skills if s.get('is_essential', False)]
                            optional_skills = [s for s in current_skills if not s.get('is_essential', False)]
                            manual_skills = [s for s in current_skills if s.get('relation_type') in ['manual', 'manual_essential', 'manual_optional']]
                            automatic_skills = [s for s in current_skills if s.get('relation_type') not in ['manual', 'manual_essential', 'manual_optional']]
                            
                            # Show skills in columns (display)
                            col1, col2 = st.columns(2)

                            with col1:
                                render_skills_two_columns(current_skills, left_title="Essentiell", right_title="Optional")

                            with col2:
                                st.write("**Skill-Statistiken:**")
                                st.write(f"• **Essential Skills:** {len(essential_skills)}")
                                st.write(f"• **Optional Skills:** {len(optional_skills)}")
                                st.write(f"• **Manuelle Skills:** {len(manual_skills)}")
                                st.write(f"• **Automatische Skills:** {len(automatic_skills)}")
                                
                                # Show current role
                                if current_profile.get('current_role'):
                                    current_role = current_profile['current_role']
                                    st.write("**Aktuelle Rolle:**")
                                    st.write(f"• **KldB-Code:** {current_role.get('KldB_Code', 'N/A')}")
                                    st.write(f"• **KldB-Rolle:** {current_role.get('KldB_Label', 'N/A')}")
                                    st.write(f"• **ESCO-Rolle:** {current_role.get('ESCO_Label', 'N/A')}")
                        else:
                            st.warning("**Keine Skills zugewiesen.** Bitte weisen Sie zuerst eine Rolle in 'Mitarbeiter-Kompetenzprofile' zu.")
                        

                        
                        # Calculate the percentage of agreement
                        match_percentage = (len(matching_skills) / len(future_skills_from_archi)) * 100 if future_skills_from_archi else 0
                        

                        
                        # NEW SECTION: Skills needed in the future (from Capabilities) - Identical to the "Strategic Professional Development" section
                        st.markdown("---")
                        st.subheader("Zukünftig benötigte Skills (aus Capabilities)")
                        
                        # Use the same data source as in the Strategic Continuing Education program
                        future_skills_from_archi = extract_future_skills_from_capabilities(st.session_state.archi_data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Zukünftig benötigte Skills (aus Capabilities):**")
                            st.write(f"• Anzahl: {len(future_skills_from_archi)}")
                            
                            # Show all future skills
                            for skill in future_skills_from_archi[:10]:  # Zeige nur die ersten 10
                                st.write(f"  - {skill['skill_name']}")
                            if len(future_skills_from_archi) > 10:
                                st.write(f"  ... und {len(future_skills_from_archi) - 10} weitere")
                        
                        with col2:
                            st.write("**Übereinstimmende Skills:**")
                            if matching_skills:
                                st.success(f"{len(matching_skills)} von {len(future_skills_from_archi)} SOLL-Skills bereits vorhanden")
                                transformed = [
                                    {
                                        'skill_label': s['skill_name'],
                                        'is_essential': s.get('is_essential', False)
                                    }
                                    for s in matching_skills
                                ]
                                render_skills_two_columns(transformed, left_title="Essentiell", right_title="Optional")
                            else:
                                st.warning("Keine Übereinstimmungen gefunden")
                        
                        # Course recommendations for missing skills
                        if missing_skills:
                            st.markdown("---")
                            st.subheader("Kursempfehlungen für fehlende Skills")
                            
                            st.info(f"**Generiere Kursempfehlungen für {len(missing_skills)} fehlende Skills...**")
                            
                            if st.button("Kursempfehlungen generieren", key="generate_course_recommendations_tab2"):
                                with st.spinner("Generiere Kursempfehlungen für fehlende Skills..."):
                                    try:
                                        # Convert missing skills to the correct format
                                        missing_skill_names = [skill['skill_name'] for skill in missing_skills]
                                        
                                        # Limit the number of skills for better performance
                                        max_skills_to_process = min(10, len(missing_skill_names))
                                        st.info(f"Verarbeite die ersten {max_skills_to_process} fehlenden Skills für bessere Performance")
                                        
                                        # Call the function
                                        recommendations = find_udemy_courses_for_skills(
                                            missing_skill_names[:max_skills_to_process],
                                            udemy_courses_df,
                                            top_k=3
                                        )
                                        
                                        if recommendations:
                                            st.success(f"{len(recommendations)} Kursempfehlungen gefunden!")
                                            
                                            # Group by skill
                                            skill_groups = {}
                                            for rec in recommendations:
                                                skill = rec.get('skill', 'Unbekannt')
                                                if skill not in skill_groups:
                                                    skill_groups[skill] = []
                                                skill_groups[skill].append(rec)
                                            
                                            # Show recommendations grouped by skill level with scores
                                            for skill_name, skill_recs in skill_groups.items():
                                                st.write(f"**Kurse für: {skill_name}**")
                                                
                                                # Sort by Similarity Score
                                                skill_recs.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                                                
                                                for i, course in enumerate(skill_recs, 1):
                                                    with st.expander(f"{i}. {course.get('course_title', 'Unbekannter Kurs')} - Score: {course.get('similarity_score', 0):.3f}", expanded=False):
                                                        col1, col2, col3 = st.columns([3, 1, 1])
                                                        
                                                        with col1:
                                                            st.write(f"**Kursname:** {course.get('course_title', 'N/A')}")
                                                            if course.get('course_headline') and course.get('course_headline') != 'N/A':
                                                                st.write(f"**Beschreibung:** {course.get('course_headline')}")
                                                            if course.get('course_description') and course.get('course_description') != 'N/A':
                                                                desc = course.get('course_description')
                                                                if len(str(desc)) > 200:
                                                                    desc = str(desc)[:200] + "..."
                                                                st.write(f"**Details:** {desc}")
                                                        
                                                        with col2:
                                                            if course.get('course_price') and course.get('course_price') != 'N/A':
                                                                st.write(f"**Preis:** {course.get('course_price')}")
                                                            if course.get('course_language') and course.get('course_language') != 'N/A':
                                                                st.write(f"**Sprache:** {course.get('course_language')}")
                                                        
                                                        with col3:
                                                            if course.get('course_url'):
                                                                st.write(f"**Link:** [Zum Kurs]({course.get('course_url')})")
                                                            if course.get('similarity_score'):
                                                                # Farbkodierung basierend auf Score
                                                                score = course.get('similarity_score', 0)
                                                                if score >= 0.8:
                                                                    st.success(f"**Match-Score:** {score:.3f} ★★★")
                                                                elif score >= 0.6:
                                                                    st.info(f"**Match-Score:** {score:.3f} ★★")
                                                                else:
                                                                    st.warning(f"**Match-Score:** {score:.3f} ★")
                                                
                                                st.markdown("---")
                                        else:
                                            st.warning("Keine Kursempfehlungen gefunden!")
                                            st.info("Versuchen Sie es mit anderen Skills oder überprüfen Sie die Udemy-Daten.")
                                            
                                            # Fallback: Show sample manual courses
                                            st.subheader("Beispiel-Kursempfehlungen")
                                            st.info("Da keine automatischen Empfehlungen gefunden wurden, zeigen wir Ihnen allgemeine Kursempfehlungen:")
                                            
                                            for skill_name in missing_skill_names[:5]:
                                                st.write(f"**Für Skill: {skill_name}**")
                                                st.write("- Grundlagen und Einführungskurse")
                                                st.write("- Fortgeschrittene Techniken")
                                                st.write("- Praktische Anwendungen")
                                                st.write("- Zertifizierungskurse")
                                                st.markdown("---")
                                    
                                    except Exception as e:
                                        st.error(f"Fehler bei der Kursempfehlung: {str(e)}")
                                        st.info("Bitte versuchen Sie es erneut oder kontaktieren Sie den Administrator.")
                                        st.exception(e)
                        else:
                            st.success("🎉 Alle SOLL-Skills sind bereits vorhanden! Keine Kursempfehlungen nötig.")
                    
                    else:
                        st.error("Konnte kein Mitarbeiterprofil für die ausgewählte IST-Rolle erstellen.")
                else:
                    st.warning("Bitte wählen Sie zuerst einen Mitarbeiter in der Sidebar aus.")
        else:
            st.warning("Keine SOLL-Skills in der XML gefunden.")
    


def show_ist_soll_matching(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df, udemy_courses_df, berufsbenennungen_df):
    """Demonstrates the current-future matching functionality with integrated course recommendations"""
    st.header("Personalplanung mit Kursempfehlung")
    
    # Logo path - direct logo from project directory
    logo_path = os.path.join(BASE_DIR, "LOGO_FutureFit_Planning.png")
    
    # Check if the logo exists; otherwise, use fallback paths
    if not os.path.exists(logo_path):
        possible_logo_paths = [
            os.path.join(BASE_DIR, "LOGO_FutureFit_Planning.png"),
            data_path("LOGO_FutureFit_Planning.png"),
            os.path.join(BASE_DIR, "futurefit_logo.png"),
            data_path("futurefit_logo.png"),
        ]
        for path in possible_logo_paths:
            if os.path.exists(path):
                logo_path = path
                break
    
    # Compact layout: logo on the right, subheading on the left
    if logo_path and os.path.exists(logo_path):
        # Compact container without a background
        # Logo and subheading side by side
        col1, col2 = st.columns([2.5, 1.3])
        with col1:
            # Subheading (left)
            st.markdown("""
            <div style="
                display: flex;
                align-items: center;
                height: 100%;
                padding-right: 15px;
                margin: 5px 0;
            ">
                <h2 style="
                    margin: 0; 
                    color: #6BB6FF; 
                    font-weight: 700; 
                    font-size: 1.65rem;
                    line-height: 1.4;
                    text-shadow: 0 1px 2px rgba(107, 182, 255, 0.2);
                ">
                    FutureFit Planning: Automatisierter Kompetenzabgleich für morgen
                </h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            # Logo in the top right corner - large and prominent
            st.markdown("""
            <div style="
                display: flex;
                justify-content: flex-end;
                align-items: flex-start;
                padding: 0;
                margin: -5px 0 5px 0;
            ">
            """, unsafe_allow_html=True)
            st.image(logo_path, width=500, use_container_width=False, output_format='PNG')
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Fallback without image - compact without background area
        st.markdown("""
        <div style="
            padding: 5px 0; 
            margin: 5px 0;
        ">
            <h2 style="
                margin: 0; 
                color: #6BB6FF; 
                font-weight: 700;
                font-size: 1.65rem;
                line-height: 1.4;
                text-shadow: 0 1px 2px rgba(107, 182, 255, 0.2);
            ">
                FutureFit Planning: Automatisierter Kompetenzabgleich für morgen
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tutorial video as a pop-up
    tutorial_video_path = os.path.join(BASE_DIR, "@TUTORIAL_Personalplanung_FutureFit.mp4 .mp4")
    
    # Check if a tutorial video exists; otherwise, use fallback paths
    if not os.path.exists(tutorial_video_path):
        possible_tutorial_paths = [
            os.path.join(BASE_DIR, "@TUTORIAL_Personalplanung_FutureFit.mp4 .mp4"),
            os.path.join(BASE_DIR, "TUTORIAL_Personalplanung_FutureFit.mp4"),
            data_path("@TUTORIAL_Personalplanung_FutureFit.mp4 .mp4"),
            data_path("TUTORIAL_Personalplanung_FutureFit.mp4"),
        ]
        for path in possible_tutorial_paths:
            if os.path.exists(path):
                tutorial_video_path = path
                break
    
    # Promotional video as a pop-up
    werbe_video_path = os.path.join(BASE_DIR, "@WERBEVIDEO_Kompetenzabgleich.mp4 .mp4")
    
    # Check if a promotional video exists; otherwise, use fallback paths
    if not os.path.exists(werbe_video_path):
        possible_werbe_paths = [
            os.path.join(BASE_DIR, "@WERBEVIDEO_Kompetenzabgleich.mp4 .mp4"),
            os.path.join(BASE_DIR, "WERBEVIDEO_Kompetenzabgleich.mp4"),
            data_path("@WERBEVIDEO_Kompetenzabgleich.mp4 .mp4"),
            data_path("WERBEVIDEO_Kompetenzabgleich.mp4"),
        ]
        for path in possible_werbe_paths:
            if os.path.exists(path):
                werbe_video_path = path
                break
    
    # Buttons for the tutorial and promotional video (top right, next to "Info")
    if (tutorial_video_path and os.path.exists(tutorial_video_path)) or (werbe_video_path and os.path.exists(werbe_video_path)):
        # Initialize session state for videos
        if 'show_tutorial' not in st.session_state:
            st.session_state.show_tutorial = False
        if 'show_werbevideo' not in st.session_state:
            st.session_state.show_werbevideo = False
        
        # Buttons to open the videos (side by side in a row, same height)
        # CSS for single-line layout, same height, and background color #6BB6FF
        st.markdown("""
        <style>
        /* Selectors for both buttons */
        button[data-testid="baseButton-secondary"][key="werbe_button"],
        button[data-testid="baseButton-secondary"][key="tutorial_button"],
        button.stButton > button[key="werbe_button"],
        button.stButton > button[key="tutorial_button"],
        div[data-testid="column"]:nth-of-type(2) button,
        div[data-testid="column"]:nth-of-type(3) button,
        button:has-text("Jetzt Einführung ansehen"),
        button:has-text("Tutorial") {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            height: 38px !important;
            min-height: 38px !important;
            max-height: 38px !important;
            padding: 0.25rem 0.75rem !important;
            font-size: 0.85rem !important;
            line-height: 1.2 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Background color #6BB6FF for both buttons - robust selector */
        div[data-testid="column"]:nth-of-type(2) .stButton > button,
        div[data-testid="column"]:nth-of-type(3) .stButton > button,
        div[data-testid="column"]:nth-of-type(2) button,
        div[data-testid="column"]:nth-of-type(3) button {
            background-color: #6BB6FF !important;
            background: #6BB6FF !important;
            color: white !important;
            border: 1px solid #6BB6FF !important;
            border-color: #6BB6FF !important;
        }
        
        div[data-testid="column"]:nth-of-type(2) .stButton > button:hover,
        div[data-testid="column"]:nth-of-type(3) .stButton > button:hover,
        div[data-testid="column"]:nth-of-type(2) button:hover,
        div[data-testid="column"]:nth-of-type(3) button:hover {
            background-color: #5BA3F5 !important;
            background: #5BA3F5 !important;
            border-color: #5BA3F5 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Buttons with the color #6BB6FF - combined approach
        col1, col2, col3 = st.columns([7, 2, 1])
        with col2:
            if werbe_video_path and os.path.exists(werbe_video_path):
                if st.button("Jetzt Einführung ansehen", help="Klicken Sie hier, um das Werbevideo zu öffnen", use_container_width=True, key="werbe_button"):
                    st.session_state.show_werbevideo = True
                    st.session_state.show_tutorial = False
                    st.rerun()
        with col3:
            if tutorial_video_path and os.path.exists(tutorial_video_path):
                if st.button("Tutorial", help="Klicken Sie hier, um das Tutorial-Video zu öffnen", use_container_width=True, key="tutorial_button"):
                    st.session_state.show_tutorial = True
                    st.session_state.show_werbevideo = False
                    st.rerun()
        
        # Aggressive JavaScript to set the button color - runs continuously
        st.markdown("""
        <style>
        /* CSS für Buttons */
        button[key="werbe_button"],
        button[key="tutorial_button"] {
            white-space: nowrap !important;
            height: 38px !important;
        }
        </style>
        <script>
        (function() {
            function applyBlueColor() {
                const buttons = document.querySelectorAll('button');
                buttons.forEach(function(btn) {
                    const text = (btn.textContent || btn.innerText || '').trim();
                    if (text === 'Jetzt Einführung ansehen' || text === 'Tutorial') {
                        // Setze Farbe direkt im Style-Attribut
                        btn.setAttribute('style', 'background-color: #6BB6FF !important; background: #6BB6FF !important; color: white !important; border: 1px solid #6BB6FF !important; white-space: nowrap !important; height: 38px !important;');
                        // Setze auch direkt auf dem Element
                        btn.style.backgroundColor = '#6BB6FF';
                        btn.style.background = '#6BB6FF';
                        btn.style.color = 'white';
                        btn.style.borderColor = '#6BB6FF';
                        btn.style.border = '1px solid #6BB6FF';
                    }
                });
            }
            
            // Execute immediately
            applyBlueColor();
            
            // Run continuously (every 50 ms)
            setInterval(applyBlueColor, 50);
            
            // MutationObserver
            const observer = new MutationObserver(applyBlueColor);
            observer.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class']});
        })();
        </script>
        """, unsafe_allow_html=True)
        
        # Tutorial video as a pop-up in Expander (if enabled)
        if st.session_state.show_tutorial and tutorial_video_path and os.path.exists(tutorial_video_path):
            st.markdown("---")
            with st.expander("🎬 **Tutorial: Personalplanung mit Kursempfehlung**", expanded=True):
                st.video(tutorial_video_path)
                st.markdown("**Hinweis:** Sie können das Video mit den Steuerelementen unten im Player steuern.")
                if st.button("❌ Tutorial schließen", use_container_width=True):
                    st.session_state.show_tutorial = False
                    st.rerun()
            st.markdown("---")
        
        # Promotional video as a pop-up in Expander (if enabled)
        if st.session_state.show_werbevideo and werbe_video_path and os.path.exists(werbe_video_path):
            st.markdown("---")
            with st.expander("🎥 **Einführung: Kompetenzabgleich**", expanded=True):
                st.video(werbe_video_path)
                st.markdown("**Hinweis:** Sie können das Video mit den Steuerelementen unten im Player steuern.")
                if st.button("❌ Einführung schließen", use_container_width=True):
                    st.session_state.show_werbevideo = False
                    st.rerun()
            st.markdown("---")
    
    st.info("**Modellbasierte Personalplanung:** Diese Sektion liest IST- und SOLL-Mitarbeiter aus der XML-Datei ein und führt ein automatisches Matching basierend auf Skill-Übereinstimmungen durch. Für fehlende Fähigkeiten werden passende Kursempfehlungen angezeigt.")
    
    # File upload area
    st.markdown("### Archi XML-Datei auswählen")
    
    # Initialize session state for XML path
    if 'selected_xml_path' not in st.session_state:
        st.session_state.selected_xml_path = None
    
    # File Upload Widget
    uploaded_file = st.file_uploader(
        "Archi XML-Datei auswählen",
        type=['xml'],
        help="Wählen Sie eine Archi XML-Datei aus, die IST- und SOLL-Mitarbeiter enthält. Sie können auch mehrere Dateien nacheinander auswählen und verarbeiten."
    )
    
    # Process uploaded file
    xml_path = None
    if uploaded_file is not None:
        # Delete any old temporary files, if present
        if st.session_state.selected_xml_path and os.path.exists(st.session_state.selected_xml_path):
            try:
                # Check whether it is a temporary file (not the default path)
                if st.session_state.selected_xml_path != data_path("Kompetenzabgleich_neuV1.xml"):
                    os.remove(st.session_state.selected_xml_path)
            except Exception as e:
                pass  # Ignore errors when deleting
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='wb') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Save the path in the session state
        st.session_state.selected_xml_path = tmp_path
        xml_path = tmp_path
        st.success(f"Datei erfolgreich geladen: {uploaded_file.name}")
    elif st.session_state.selected_xml_path and os.path.exists(st.session_state.selected_xml_path):
        # Use the previously selected file
        xml_path = st.session_state.selected_xml_path
        st.info(f"Verwende zuvor ausgewählte Datei: {os.path.basename(st.session_state.selected_xml_path)}")
    else:
        # No file selected - set xml_path to None
        xml_path = None
    
    # Check if the XML file exists
    if not xml_path or not os.path.exists(xml_path):
        if uploaded_file is not None:
            # The file was uploaded but could not be saved
            st.error(f"Fehler beim Speichern der hochgeladenen Datei.")
            return
        # No file selected - show info only, no error message
        return
    
    # Save the current XML path to the session state for processing
    st.session_state.current_xml_path = xml_path
    
    # Refresh button
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        update_button = st.button("Aktualisierung durchführen", type="primary", use_container_width=True)
    
    # Show a notification when a new file has been uploaded
    if uploaded_file is not None:
        st.success(f"Datei '{uploaded_file.name}' erfolgreich geladen. Klicken Sie auf 'Aktualisierung durchführen', um die Verarbeitung zu starten.")
    
    # Check whether processed data is already present in the session state
    processing_data_key = 'xml_processing_data'
    has_cached_data = processing_data_key in st.session_state and st.session_state[processing_data_key] is not None
    
    # If data is already available and the button hasn't been clicked, use the cached data
    if has_cached_data and not update_button:
        cached_data = st.session_state[processing_data_key]
        xml_data = cached_data['xml_data']
        ist_mitarbeiter = cached_data['ist_mitarbeiter']
        soll_faehigkeiten = cached_data['soll_faehigkeiten']
        total_soll_skills = cached_data['total_soll_skills']
        xml_path = cached_data['xml_path']
        # Use the cached `is_employee_with_details` if available
        if 'ist_mitarbeiter_mit_details' in st.session_state:
            ist_mitarbeiter_mit_details = st.session_state['ist_mitarbeiter_mit_details']
        else:
            ist_mitarbeiter_mit_details = []
        # Skip the XML processing and use the cached data
        skip_processing = True
    else:
        skip_processing = False
    
    # Only execute the processing if the button has been clicked
    if not update_button and not skip_processing:
        if uploaded_file is None:
            st.info("Bitte wählen Sie eine XML-Datei aus und klicken Sie dann auf 'Aktualisierung durchführen', um die SOLL-Profile & Rankings zu aktualisieren.")
        return
    
    # Perform normal processing unless skipped
    if not skip_processing:
        # Create a dynamic cache buster based on the file path and timestamp
        # This ensures that the cache is invalidated when a new file is created
        cache_buster_value = f"{FUTURE_PROFILE_CACHE_KEY}_{xml_path}_{os.path.getmtime(xml_path) if os.path.exists(xml_path) else time.time()}"
        
        # Parse XML file
        with st.spinner("Lade und parse XML-Datei..."):
            xml_data = parse_ist_soll_xml(xml_path, cache_buster=cache_buster_value)
        
        if not xml_data['success']:
            st.error(f"Fehler beim Parsen der XML-Datei: {xml_data.get('error', 'Unbekannter Fehler')}")
            if 'traceback' in xml_data:
                with st.expander("Fehlerdetails anzeigen"):
                    st.code(xml_data['traceback'])
            return
        
        ist_mitarbeiter = xml_data['ist_mitarbeiter']
        soll_faehigkeiten = xml_data['soll_faehigkeiten']
        total_soll_skills = xml_data.get('soll_total_skill_count', len(soll_faehigkeiten))
        
        # Save all important data to the session state for restoration
        st.session_state.xml_processing_data = {
            'xml_data': xml_data,
            'ist_mitarbeiter': ist_mitarbeiter,
            'soll_faehigkeiten': soll_faehigkeiten,
            'total_soll_skills': total_soll_skills,
            'xml_path': xml_path
        }
    
    st.info(f"""
    **XML-Daten:**
    - **IST-Mitarbeiter:** {len(ist_mitarbeiter)} Mitarbeiter mit Business Roles
    - **SOLL-Fähigkeiten:** {total_soll_skills} Fähigkeiten für neue Unternehmensstrategie
    """)
    
    if not ist_mitarbeiter:
        st.warning("Keine IST-Mitarbeiter in der XML-Datei gefunden.")
        return
    
    if not soll_faehigkeiten:
        st.warning("Keine SOLL-Fähigkeiten in der XML-Datei gefunden.")
        return
    
    # Show current employees with complete information (name, job title, KldB, ESCO, skills)
    st.markdown("---")
    st.subheader("IST-Mitarbeiter (aus XML)")
    
    # Prepare current employee data using Skills (only if not skipped)
    # Always initialize roles_not_found, even if skipped
    nicht_gefundene_rollen = []
    
    if not skip_processing:
        ist_mitarbeiter_mit_details = []
        
        for mitarbeiter in ist_mitarbeiter:
            mitarbeiter_name = mitarbeiter['name']
            business_roles = mitarbeiter['business_roles']
            
            # Gather all the information for this employee
            kldb_codes = []
            esco_labels = []
            esco_codes = []
            role_matches = []
            alle_skills_set = set()
            
            for role in business_roles:
                # Use semantic matching for Business Role -> KldB Code -> ESCO Role
                kldb_code, kldb_label, esco_label, esco_code = find_kldb_code_for_business_role_semantic(
                    role, kldb_esco_df, berufsbenennungen_df, min_similarity=0.2
                )
                
                if kldb_code:
                    display_role_label = normalize_display_label(role)
                    kldb_codes.append(f"{kldb_code} ({display_role_label})")
                    esco_labels.append(esco_label)
                    esco_codes.append(esco_code)
                    role_matches.append({
                        'business_role': role,
                        'kldb_code': kldb_code,
                        'kldb_label': display_role_label,
                        'esco_label': esco_label,
                        'esco_code': esco_code
                    })
                    
                    # Acquire the skills needed for this ESCO role
                    role_skills = get_skills_for_occupation_simple(
                        esco_label,
                        st.session_state.occupation_skills_mapping,
                        occupations_df
                    )
                    
                    if role_skills:
                        for skill in role_skills:
                            skill_label = skill.get('skill_label', '')
                            if skill_label:
                                alle_skills_set.add(skill_label)
                else:
                    # No match found - save warning
                    nicht_gefundene_rollen.append({
                        'mitarbeiter': mitarbeiter_name,
                        'role': role
                    })
            
            # Also check the manual skills in employees_df
            employee_match = employees_df[employees_df['Name'].str.contains(mitarbeiter_name, case=False, na=False)]
            if not employee_match.empty:
                manual_skills = employee_match.iloc[0].get('Manual_Skills', '')
                if manual_skills and pd.notna(manual_skills):
                    for skill in str(manual_skills).split(';'):
                        if skill.strip():
                            alle_skills_set.add(skill.strip())
            
            ist_mitarbeiter_mit_details.append({
                'name': mitarbeiter_name,
                'business_roles': business_roles,
                'kldb_codes': kldb_codes,
                'esco_labels': esco_labels,
                'esco_codes': esco_codes,
                'skills': sorted(list(alle_skills_set)),
                'role_matches': role_matches
            })
        
        # Remove duplicate records for current employees by name (if duplicates still exist)
        ist_mitarbeiter_unique = {}
        for mitarbeiter_detail in ist_mitarbeiter_mit_details:
            name = mitarbeiter_detail['name']
            if name not in ist_mitarbeiter_unique:
                ist_mitarbeiter_unique[name] = mitarbeiter_detail
            else:
                # The employee already exists - merge business roles and skills
                existing = ist_mitarbeiter_unique[name]
                # Add missing business roles
                for role in mitarbeiter_detail['business_roles']:
                    if role not in existing['business_roles']:
                        existing['business_roles'].append(role)
                # Add missing KldB codes
                for kldb in mitarbeiter_detail['kldb_codes']:
                    if kldb not in existing['kldb_codes']:
                        existing['kldb_codes'].append(kldb)
                # Add missing ESCO labels
                for esco in mitarbeiter_detail['esco_labels']:
                    if esco not in existing['esco_labels']:
                        existing['esco_labels'].append(esco)
                # Merge Skills (The set automatically maintains uniqueness)
                existing_skills_set = set(existing['skills'])
                new_skills_set = set(mitarbeiter_detail['skills'])
                existing['skills'] = sorted(list(existing_skills_set | new_skills_set))
                # Merge Matches
                existing_matches = existing.get('role_matches', [])
                for match in mitarbeiter_detail.get('role_matches', []):
                    if not any(
                        m['business_role'] == match['business_role'] and
                        m['kldb_code'] == match['kldb_code'] and
                        m.get('esco_code') == match.get('esco_code')
                        for m in existing_matches
                    ):
                        existing_matches.append(match)
                existing['role_matches'] = existing_matches
        
        # Convert back to list
        ist_mitarbeiter_mit_details = list(ist_mitarbeiter_unique.values())
        
        # Save to session state
        st.session_state.ist_mitarbeiter_mit_details = ist_mitarbeiter_mit_details
    
    # Automatically sync automatically detected KldB codes with employee data
    def sync_employee_roles_with_data(ist_details):
        if st.session_state.employees_data.empty:
            return []
        
        updates = []
        employees_df = st.session_state.employees_data
        
        for detail in ist_details:
            role_matches = detail.get('role_matches', [])
            if not role_matches:
                continue
            
            normalized_name = detail['name'].strip().lower()
            employee_rows = employees_df[
                employees_df['Name'].astype(str).str.strip().str.lower() == normalized_name
            ]
            
            if employee_rows.empty:
                continue
            
            employee_id = employee_rows.iloc[0]['Employee_ID']
            best_match = role_matches[0]
            current_code = employee_rows.iloc[0].get('KldB_5_digit', '')
            current_esco = employee_rows.iloc[0].get('ESCO_Role', '')
            
            changed = False
            if best_match['kldb_code'] and current_code != best_match['kldb_code']:
                st.session_state.employees_data.loc[
                    st.session_state.employees_data['Employee_ID'] == employee_id,
                    'KldB_5_digit'
                ] = best_match['kldb_code']
                changed = True
            
            if best_match['esco_label'] and current_esco != best_match['esco_label']:
                st.session_state.employees_data.loc[
                    st.session_state.employees_data['Employee_ID'] == employee_id,
                    'ESCO_Role'
                ] = best_match['esco_label']
                changed = True
            
            if changed:
                updates.append({
                    'employee_id': employee_id,
                    'name': detail['name'],
                    'business_role': best_match['business_role'],
                    'kldb_code': best_match['kldb_code'],
                    'kldb_label': best_match['kldb_label'],
                    'esco_label': best_match['esco_label']
                })
        
        if updates:
            save_employees_to_csv(st.session_state.employees_data)
        return updates
    
    synchronization_updates = sync_employee_roles_with_data(ist_mitarbeiter_mit_details)
    if synchronization_updates:
        st.success(f"{len(synchronization_updates)} Mitarbeiterprofile mit Business-Rollen synchronisiert.")
        with st.expander("Details der automatischen Zuordnung", expanded=False):
            for update in synchronization_updates:
                st.write(f"- {update['name']} → {update['business_role']} → {update['kldb_code']} ({update['kldb_label']})")
    
    # Display current employees in a table
    ist_df_data = []
    for mitarbeiter_detail in ist_mitarbeiter_mit_details:
        ist_df_data.append({
            'Name': mitarbeiter_detail['name'],
            'Business Role(s)': ', '.join(mitarbeiter_detail['business_roles']) if mitarbeiter_detail['business_roles'] else 'Keine Rollen',
            'KldB-Code(s)': ', '.join(mitarbeiter_detail['kldb_codes']) if mitarbeiter_detail['kldb_codes'] else 'Nicht gefunden',
            'ESCO-Rolle(n)': ', '.join(mitarbeiter_detail['esco_labels']) if mitarbeiter_detail['esco_labels'] else 'Nicht gefunden',
            'Anzahl Skills': len(mitarbeiter_detail['skills'])
        })
    
    ist_df = pd.DataFrame(ist_df_data)
    st.dataframe(ist_df, use_container_width=True, hide_index=True)
    
    # Show warnings for roles that were not found
    if nicht_gefundene_rollen:
        with st.expander("△ Warnung: Nicht zugeordnete Business Roles", expanded=False):
            st.warning(f"Für {len(nicht_gefundene_rollen)} Business Role(s) konnte kein KldB-Code gefunden werden:")
            for warnung in nicht_gefundene_rollen:
                st.write(f"- **{warnung['mitarbeiter']}**: Business Role '{warnung['role']}' konnte nicht zugeordnet werden")
            st.info("ℹ Tipp: Überprüfen Sie die Schreibweise der Business Roles in der XML-Datei oder fügen Sie manuell KldB-Codes hinzu.")
    
    # Show details about current employees and their skills
    if ist_mitarbeiter_mit_details:
        with st.expander("▢ Details zu IST-Mitarbeitern mit Skills", expanded=False):
            for mitarbeiter_detail in ist_mitarbeiter_mit_details:
                st.write(f"**{mitarbeiter_detail['name']}**")
                st.write(f"**Business Role(s):** {', '.join(mitarbeiter_detail['business_roles'])}")
                
                if mitarbeiter_detail['kldb_codes']:
                    st.write(f"**KldB-Code(s):** {', '.join(mitarbeiter_detail['kldb_codes'])}")
                
                if mitarbeiter_detail['esco_labels']:
                    st.write(f"**ESCO-Rolle(n):** {', '.join(mitarbeiter_detail['esco_labels'])}")
                
                if mitarbeiter_detail['skills']:
                    st.write(f"**Skills ({len(mitarbeiter_detail['skills'])}):**")
                    # Show skills in columns
                    cols = st.columns(min(3, len(mitarbeiter_detail['skills'])))
                    for i, skill in enumerate(mitarbeiter_detail['skills']):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            st.write(f"• {skill}")
                else:
                    st.write("Keine Skills gefunden")
                st.markdown("---")
    
    # Save the current employee details in the session state for matching
    st.session_state.ist_mitarbeiter_mit_details = ist_mitarbeiter_mit_details
    
    # Option to import current employees into central administration
    st.markdown("---")
    st.subheader("Import in zentrale Mitarbeiter-Verwaltung")
    
    # Check whether it has already been imported
    import_status_key = 'ist_employees_imported'
    import_message_key = 'import_success_message'
    
    if import_status_key not in st.session_state:
        st.session_state[import_status_key] = False
    
    if st.session_state[import_status_key]:
        st.success("✅ IST-Mitarbeiter wurden bereits in die zentrale Mitarbeiter-Verwaltung importiert.")
        if st.button("🔄 Erneut importieren", key="reimport_button"):
            st.session_state[import_status_key] = False
    else:
        if st.button("IST-Mitarbeiter in zentrale Mitarbeiter-Verwaltung importieren", key="import_button", type="primary"):
            # Use the data from the session state
            import_data = st.session_state.get('ist_mitarbeiter_mit_details', [])
            
            if not import_data:
                st.error("Keine IST-Mitarbeiter-Daten verfügbar. Bitte führen Sie zuerst die Aktualisierung durch.")
                st.info("Klicken Sie auf 'Aktualisierung durchführen', um die XML-Datei zu verarbeiten.")
            else:
                with st.spinner("Importiere IST-Mitarbeiter..."):
                    # Load employees_df from session state, if available; otherwise, use the parameter
                    current_employees_df = st.session_state.get('employees_data', employees_df.copy())
                    
                    imported_count = 0
                    updated_count = 0
                    
                    # Use the details already calculated with KldB and ESCO
                    for mitarbeiter_detail in import_data:
                        mitarbeiter_name = mitarbeiter_detail['name']
                        
                        # Extract the KldB code and ESCO role from the details
                        # Take the first KldB/ESCO combination found (if multiple roles exist)
                        kldb_code = None
                        esco_role = None
                        
                        if mitarbeiter_detail['kldb_codes']:
                            # Extract the KldB code from the first entry (format: "B 12345-101 (Label)" or "B12345-101 (Label)")
                            first_kldb_entry = mitarbeiter_detail['kldb_codes'][0]
                            # Extract the code - various formats available
                            # Format 1: "B 12345-101 (Label)""
                            # Format 2: "B12345-101 (Label)""
                            # Format 3: "B12345-101" (without label)
                            kldb_match = re.search(r'([B]\s*\d+\s*-\s*\d+)', first_kldb_entry)
                            if kldb_match:
                                # Normalize format: Remove spaces, keep hyphens
                                kldb_code = kldb_match.group(1).replace(' ', '').strip()
                            else:
                                # Fallback: Try searching for the format directly
                                kldb_match = re.search(r'(B\d+-\d+)', first_kldb_entry)
                                if kldb_match:
                                    kldb_code = kldb_match.group(1)
                        
                        if mitarbeiter_detail['esco_labels']:
                            # Take on the first ESCO role
                            esco_role = mitarbeiter_detail['esco_labels'][0]
                        
                        # If no KldB/ESCO is found, try extracting it directly from the business roles
                        if not kldb_code or not esco_role:
                            business_roles = mitarbeiter_detail.get('business_roles', [])
                            if business_roles:
                                # Use the first business role for matching
                                role = business_roles[0]
                                kldb_code_found, kldb_label_found, esco_label_found, esco_code_found = find_kldb_code_for_business_role_semantic(
                                    role, kldb_esco_df, berufsbenennungen_df, min_similarity=0.2
                                )
                                if kldb_code_found and not kldb_code:
                                    kldb_code = kldb_code_found
                                if esco_label_found and not esco_role:
                                    esco_role = esco_label_found
                        
                        # Check if the employee already exists
                        existing = current_employees_df[current_employees_df['Name'].str.strip().str.lower() == mitarbeiter_name.strip().lower()]
                        
                        if existing.empty:
                            # New Employee
                            new_employee_id = f"EMP_{len(current_employees_df) + 1}"
                            new_row = {
                                'Employee_ID': new_employee_id,
                                'Name': mitarbeiter_name,
                                'KldB_5_digit': kldb_code if kldb_code else '',
                                'Manual_Skills': '',
                                'ESCO_Role': esco_role if esco_role else '',
                                'Target_KldB_Code': '',
                                'Target_KldB_Label': '',
                                'Target_ESCO_Code': '',
                                'Target_ESCO_Label': '',
                                'Manual_Essential_Skills': '',
                                'Manual_Optional_Skills': '',
                                'Removed_Skills': ''
                            }
                            current_employees_df = pd.concat([current_employees_df, pd.DataFrame([new_row])], ignore_index=True)
                            imported_count += 1
                        else:
                            # Update existing employee
                            idx = existing.index[0]
                            if kldb_code:
                                current_employees_df.at[idx, 'KldB_5_digit'] = kldb_code
                            if esco_role:
                                current_employees_df.at[idx, 'ESCO_Role'] = esco_role
                            updated_count += 1
                    
                    # Update session state
                    st.session_state.employees_data = current_employees_df
                    
                    # Save as CSV
                    if save_employees_to_csv(current_employees_df):
                        st.session_state[import_status_key] = True
                        # Display success message immediately (without rerunning)
                        st.success(f"✅ Import abgeschlossen: {imported_count} neue Mitarbeiter importiert, {updated_count} Mitarbeiter aktualisiert.")
                        st.info("ℹ Die Mitarbeiter sind jetzt in der zentralen Mitarbeiter-Verwaltung verfügbar und können in allen Funktionen (Mitarbeiter-Kompetenzprofile, Berufsabgleich, Strategische Weiterbildung, Kursempfehlungen) verwendet werden.")
                    else:
                        st.warning("△ Import durchgeführt, aber Speichern in CSV fehlgeschlagen!")
                        st.info("ℹ Die Mitarbeiter sind im Session State verfügbar, aber nicht dauerhaft gespeichert.")
    
    # Note: Matching is calculated automatically for each future profile in the following section.
    
    # Ranking by future future role
    future_profiles = xml_data.get('soll_profile_map', [])
    if not future_profiles:
        future_profiles = [{
            'name': 'Aggregiertes SOLL-Profil',
            'skills': soll_faehigkeiten,
            'skill_count': len(soll_faehigkeiten)
        }]
    
    for detail in ist_mitarbeiter_mit_details:
        normalized_set = set()
        for skill in detail.get('skills', []):
            norm = normalize_skill_name(skill)
            if norm:
                normalized_set.add(norm)
        detail['normalized_skills_set'] = normalized_set
    
    st.markdown("---")
    st.subheader("SOLL-Profile & Ranking der IST-Mitarbeiter")
    total_required_skills = sum(len(profile.get('skills', [])) for profile in future_profiles)
    st.write(f"**Erfasste zukünftige Skills:** {total_required_skills} (Erwartung laut Modell: 61)")
    
    skill_lookup = build_skill_lookup(skills_df)
    
    for profile in future_profiles:
        render_future_profile_section(
            profile,
            ist_mitarbeiter_mit_details,
            udemy_courses_df,
            skill_lookup
        )


if __name__ == "__main__":
    main()