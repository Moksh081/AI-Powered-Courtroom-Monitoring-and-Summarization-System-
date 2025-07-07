#!/usr/bin/env python3
"""
legal_llm_summarizer.py

LEGAL-SPECIFIC LLM SUMMARIZER
Uses a legal-domain large language model (e.g., Legal-BERT, LegalT5, or GPT-4 via API) to generate an executive summary from a diarized legal transcript.

Usage:
  python legal_llm_summarizer.py --input diarized_proceedings.txt --output summary.txt

Note: This script uses the "nlpaueb/legal-bert-base-uncased" model for demonstration. For best results, use a more advanced legal LLM or an API (e.g., OpenAI GPT-4) if available.
"""

import os
import argparse, logging, sys
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_by_role(text):
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

def generate_legal_summary(text, max_length=220):
    # Use the BART-large-CNN model for summarization (robust, general-purpose)
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")
    # BART can handle up to 1024 tokens, so chunk input at sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 900:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    summaries = [summarizer(chunk, max_length=max_length, min_length=60, do_sample=False)[0]['summary_text'] for chunk in chunks if chunk.strip()]
    return ' '.join(summaries)

def main():
    parser = argparse.ArgumentParser("Legal LLM Summarizer")
    parser.add_argument("-i", "--input", required=True, help="Input diarized transcript file")
    parser.add_argument("-o", "--output", required=True, help="Output summary file")
    parser.add_argument("--max-length", type=int, default=220, help="Max summary length")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    try:
        text = load_text(args.input)
        text = preprocess_by_role(text)
    except Exception as e:
        logger.error(f"Could not read input file: {e}")
        sys.exit(1)

    logger.info(f"Input text length: {len(text)} characters")
    summary = generate_legal_summary(text, max_length=args.max_length)

    output = f"{'='*100}\nLEGAL LLM SUMMARY (कार्यकारी सारांश)\n{'='*100}\n{summary}\n{'='*100}"

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    except Exception as e:
        logger.error(f"Could not write output file: {e}")
        sys.exit(1)

    logger.info(f"Legal LLM summary written to {args.output}")
    print(output)

if __name__ == "__main__":
    main()
