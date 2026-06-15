import pandas as pd

df = pd.read_csv('data/KldB_to_ESCO_Mapping_clean.csv')
identical = df[df['KldB_Label'].str.lower() == df['ESCO_Label'].str.lower()]

print("=" * 80)
print("BEISPIELE WO KLDB UND ESCO IDENTISCH SIND (für Tests):")
print("=" * 80)
print()

for i, row in identical.head(15).iterrows():
    print(f"KldB-Code: {row['KldB_Code']}")
    print(f"Bezeichnung: {row['KldB_Label']}")
    print("-" * 80)
    
print()
print(f"Insgesamt {len(identical)} identische Einträge von {len(df)} gefunden.")

