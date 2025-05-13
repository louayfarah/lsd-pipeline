import pandas as pd
import requests
import re
import time
import sys
import numpy as np

# After running LibreTranslate contianer...

# ——— Configuration ———
INPUT_CSV = 'openbook_qa_en.csv'
OUTPUT_CSV = 'openbook_qa_fr.csv'
TRANSLATE_URL = 'http://127.0.0.1:5000/translate'
SOURCE_LANG = 'en'
TARGET_LANG = 'fr'
SLEEP_BETWEEN_REQUESTS = 1  # seconds

# Make NumPy’s repr wrap arrays similarly to your example
np.set_printoptions(linewidth=75)

# ——— Helpers ———
def translate_text(text: str) -> str:
    """Call LibreTranslate and return the translated text."""
    payload = {'q': text, 'source': SOURCE_LANG, 'target': TARGET_LANG}
    resp = requests.post(TRANSLATE_URL, json=payload)
    resp.raise_for_status()
    return resp.json().get('translatedText', '')

def parse_texts(choices_str: str) -> list:
    """
    Extract only the 'text' entries from the choices field, e.g.
      "{'text': array(['opt1',
                       'opt2',
                       …], dtype=object), 'label': …}"
    Returns a list of the text strings.
    """
    m = re.search(
        r"'text':\s*array\(\s*\[([\s\S]*?)\]\s*,\s*dtype=object\s*\)",
        choices_str
    )
    if not m:
        return []
    inner = m.group(1)
    # find all single- or double-quoted items
    pairs = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
    # pairs is list of tuples; pick whichever group matched
    return [a or b for a, b in pairs]

def parse_labels(choices_str: str) -> list:
    """
    Extract the 'label' entries from the choices field.
    """
    m = re.search(
        r"'label':\s*array\(\s*\[([\s\S]*?)\]\s*,\s*dtype=object\s*\)",
        choices_str
    )
    if not m:
        return []
    inner = m.group(1)
    return re.findall(r"'([^']*)'", inner)

# ——— Main ———
def main():
    df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    total = len(df)

    try:
        for count, (idx, row) in enumerate(df.iterrows(), start=1):
            print(f"\n--- Translating row {count}/{total} (index {idx}) ---")

            # 1) question_stem
            orig_q = row['question_stem']
            trans_q = translate_text(orig_q)
            print(f"Q:\n  EN → {orig_q}\n  FR → {trans_q}")
            df.at[idx, 'question_stem'] = trans_q

            # 2) fact1
            orig_f = row['fact1']
            trans_f = translate_text(orig_f)
            print(f"Fact1:\n  EN → {orig_f}\n  FR → {trans_f}")
            df.at[idx, 'fact1'] = trans_f

            # 3) choices array: extract texts and labels separately
            texts = parse_texts(row['choices'])
            labels = parse_labels(row['choices'])
            translated = []
            print("Choices:")
            for lbl, opt in zip(labels, texts):
                tr = translate_text(opt)
                print(f"  {lbl} EN → {opt}\n    FR → {tr}")
                translated.append(tr)

            # Build genuine numpy arrays and use repr to get proper formatting
            arr_text  = np.array(translated, dtype=object)
            arr_label = np.array(labels, dtype=object)
            df.at[idx, 'choices'] = repr({'text': arr_text, 'label': arr_label})

            # Throttle
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        # Save progress on Ctrl+C
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n🛑 Interrupted at row {count}/{total}. Progress saved to {OUTPUT_CSV}")
        sys.exit(0)

    # Save when done
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ All done! Translated file saved as {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
