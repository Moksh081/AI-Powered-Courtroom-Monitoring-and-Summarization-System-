#!/usr/bin/env python3
"""
abstractive_diarized_summarizer.py

ABSTRACTIVE DIARIZED LEGAL SUMMARIZER
Generates only the executive (abstractive) summary from a diarized legal transcript, omitting detailed structured analysis.

Usage:
  python abstractive_diarized_summarizer.py --input diarized_proceedings.txt --output summary.txt
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"  # Force transformers to use PyTorch, not TensorFlow

import argparse, logging, sys
from transformers import pipeline, AutoTokenizer, AutoModel

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_legal_content_with_legalbert(text):
    """
    Use LegalBERT to extract legal-relevant sentences or sections from the input text.
    For demonstration, this function will select sentences containing legal keywords.
    In production, you could use NER, classification, or QA with LegalBERT.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
    legalbert_model = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(legalbert_model)
    # For simplicity, use NER pipeline (could be improved for more advanced extraction)
    nlp = hf_pipeline("ner", model=legalbert_model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    # Split text into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    legal_sentences = []
    legal_keywords = ["section", "act", "court", "judge", "evidence", "prosecution", "defense", "accused", "witness", "fir", "ipc", "crpc", "order", "judgment", "petition", "complainant", "respondent", "plaintiff", "defendant"]
    for sent in sentences:
        if any(kw in sent.lower() for kw in legal_keywords):
            legal_sentences.append(sent)
        else:
            # Use NER to check for legal entities
            entities = nlp(sent)
            if entities:
                legal_sentences.append(sent)
    # If nothing found, fallback to original text
    if not legal_sentences:
        return text
    return ' '.join(legal_sentences)

def extract_structured_legal_context(text):
    import re
    context = {}
    # Extract fields
    context['court'] = re.search(r'IN THE COURT OF ([^\n]+)', text, re.IGNORECASE)
    context['judge'] = re.search(r'Coram: ([^\n]+)', text, re.IGNORECASE)
    context['date'] = re.search(r'Date: ([^\n]+)', text, re.IGNORECASE)
    context['case_number'] = re.search(r'(Sessions Case No\.[^\n]+)', text)
    context['case_name'] = re.search(r'State v\. ([^\n]+)', text)
    context['fir'] = re.search(r'FIR No[:\s]*([\w/]+)', text)
    context['sections'] = re.search(r'Sections[:\s]*([\d, ]+IPC)', text)
    context['victim'] = re.search(r'Deceased[:\s]*([A-Za-z ]+)', text)
    context['accused'] = re.findall(r'Accused[:\s]*[\d.]*\s*([A-Za-z ]+)', text)
    context['prosecutor'] = re.search(r'Public Prosecutor[\s:]*([A-Za-z ]+)', text)
    context['adv1'] = re.search(r'Advocate ([A-Za-z ]+) for Accused No. 1', text)
    context['adv2'] = re.search(r'Advocate ([A-Za-z ]+) for Accused No. 2', text)
    context['io'] = re.search(r'Investigating Officer: ([^\n]+)', text)
    context['mo'] = re.search(r'Medical Officer: ([^\n]+)', text)
    context['ps'] = re.search(r'Police Station[:\s]*([A-Za-z ]+)', text)

    lines = []
    if context['court']:
        lines.append(f"Court: {context['court'].group(1).strip()}")
    if context['judge']:
        lines.append(f"Judge: {context['judge'].group(1).strip()}")
    if context['date']:
        lines.append(f"Date: {context['date'].group(1).strip()}")
    if context['case_number']:
        lines.append(f"{context['case_number'].group(1).strip()}")
    if context['case_name']:
        lines.append(f"Case: State v. {context['case_name'].group(1).strip()}")
    if context['fir']:
        lines.append(f"FIR No: {context['fir'].group(1).strip()}")
    if context['sections']:
        lines.append(f"Sections: {context['sections'].group(1).strip()}")
    if context['victim']:
        lines.append(f"Victim: {context['victim'].group(1).strip()}")
    if context['accused']:
        accused_clean = ', '.join([a.strip() for a in context['accused'] if a.strip()])
        if accused_clean:
            lines.append(f"Accused: {accused_clean}")
    if context['prosecutor']:
        lines.append(f"Public Prosecutor: {context['prosecutor'].group(1).strip()}")
    if context['adv1']:
        lines.append(f"Defense (Accused 1): {context['adv1'].group(1).strip()}")
    if context['adv2']:
        lines.append(f"Defense (Accused 2): {context['adv2'].group(1).strip()}")
    if context['io']:
        lines.append(f"Investigating Officer: {context['io'].group(1).strip()}")
    if context['mo']:
        lines.append(f"Medical Officer: {context['mo'].group(1).strip()}")
    if context['ps']:
        lines.append(f"Police Station: {context['ps'].group(1).strip()}")
    if lines:
        return '\n'.join(lines) + "\n\nSummary:"
    else:
        return ''

def generate_abstractive_summary(text, max_length=220):
    legal_content = extract_legal_content_with_legalbert(text)
    structured_context = extract_structured_legal_context(text)
    if structured_context:
        summarization_input = structured_context + "\n" + legal_content
    else:
        summarization_input = legal_content
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
    if len(summarization_input) > 3500:
        chunks = [summarization_input[i:i+3500] for i in range(0, len(summarization_input), 3500)]
        summaries = [summarizer(chunk, max_length=max_length, min_length=60, do_sample=False)[0]['summary_text'] for chunk in chunks]
        return ' '.join(summaries)
    else:
        summary = summarizer(summarization_input, max_length=max_length, min_length=60, do_sample=False)
        return summary[0]['summary_text']

def deduplicate_lines(text):
    seen = set()
    result = []
    for line in text.splitlines():
        line_clean = line.strip().lower()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            result.append(line)
    return '\n'.join(result)

def filter_summary(summary, input_text):
    input_lines = set([line.strip().lower() for line in input_text.splitlines() if line.strip()])
    filtered = []
    contradiction_phrases = [
        'no record of financial dispute',
        'no financial dispute',
        'no records of financial dispute',
        'no financial dispute between'
    ]
    for line in summary.split('. '):
        line_clean = line.strip().lower()
        if (any(line_clean in l for l in input_lines) or
            not any(phrase in line_clean for phrase in contradiction_phrases)):
            filtered.append(line)
    return '. '.join(filtered)

def preprocess_by_role(text):
    # Revert to previous: build a single structured input for the summarizer
    import re
    from collections import defaultdict
    role_statements = defaultdict(list)
    for line in text.splitlines():
        match = re.match(r'<spk:([^>]+)>\s*(.*)', line)
        if match:
            role, statement = match.groups()
            role_statements[role.strip()].append(statement.strip())
    sections = []
    for role, statements in role_statements.items():
        if statements:
            sections.append(f"{role.upper()} STATEMENTS:")
            for s in statements:
                sections.append(f"- {s}")
            sections.append("")
    return '\n'.join(sections)

def main():
    parser = argparse.ArgumentParser("Abstractive Diarized Legal Summarizer")
    parser.add_argument("-i", "--input", required=True, help="Input diarized transcript file")
    parser.add_argument("-o", "--output", required=True, help="Output summary file")
    parser.add_argument("--max-length", type=int, default=220, help="Max summary length")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    try:
        text = load_text(args.input)
        text = deduplicate_lines(text)  # Deduplicate lines before processing
        summarization_input = preprocess_by_role(text)  # Single structured input for summarizer
    except Exception as e:
        logger.error(f"Could not read input file: {e}")
        sys.exit(1)

    logger.info(f"Input text length: {len(text)} characters")
    summary = generate_abstractive_summary(summarization_input, max_length=args.max_length)
    summary = filter_summary(summary, text)  # Filter hallucinated/contradictory lines

    output = f"{'='*100}\nABSTRACTIVE SUMMARY (कार्यकारी सारांश)\n{'='*100}\n{summary}\n{'='*100}"

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    except Exception as e:
        logger.error(f"Could not write output file: {e}")
        sys.exit(1)

    logger.info(f"Abstractive summary written to {args.output}")
    print(output)

if __name__ == "__main__":
    main()
