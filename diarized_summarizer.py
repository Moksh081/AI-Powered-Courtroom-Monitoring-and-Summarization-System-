import os
import openai
import pandas as pd

# Set your Groq API key (replace with your secure method in production)
openai.api_key = "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"

# Path to diarized transcript files
diary_folder = os.path.join(os.path.dirname(__file__), '..', 'Speech_to_Text')
transcript_files = [
    f for f in os.listdir(diary_folder)
    if f.endswith('.txt') and 'diarized' in f
]

def read_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Chunking logic to fit within LLM token limits ---
import re
def chunk_text(text, max_words=800):
    # Split by sentences, then group into chunks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    word_count = 0
    for sent in sentences:
        words = sent.split()
        if word_count + len(words) > max_words and current:
            chunks.append(' '.join(current))
            current = []
            word_count = 0
        current.append(sent)
        word_count += len(words)
    if current:
        chunks.append(' '.join(current))
    return chunks

def generate_summary_groq(prompt, model="llama3-8b-8192"):
    client = openai.OpenAI(
        api_key=openai.api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def make_prompts(transcript_text):
    # You can customize this prompt for your diarized format
    rr_prompt = f"Summarize the following legal proceeding with clear sections for Facts, Issues, and Decision.\n\n{{chunk}}"
    full_prompt = f"Summarize this legal proceeding:\n\n{{chunk}}"
    return rr_prompt, full_prompt

def summarize_transcripts():
    results = []
    for fname in transcript_files:
        file_path = os.path.join(diary_folder, fname)
        transcript = read_transcript(file_path)
        rr_prompt_template, full_prompt_template = make_prompts(transcript)
        # Chunk transcript
        chunks = chunk_text(transcript, max_words=1500)
        # Summarize each chunk, then combine
        rr_chunk_summaries = []
        full_chunk_summaries = []
        for chunk in chunks:
            rr_prompt = rr_prompt_template.replace("{chunk}", chunk)
            full_prompt = full_prompt_template.replace("{chunk}", chunk)
            rr_chunk_summaries.append(generate_summary_groq(rr_prompt))
            full_chunk_summaries.append(generate_summary_groq(full_prompt))
        # Optionally, combine chunk summaries into a final summary
        rr_final_prompt = "Combine the following section summaries into a single, well-structured summary with clear Facts, Issues, and Decision sections.\n\n" + "\n\n".join(rr_chunk_summaries)
        full_final_prompt = "Combine the following summaries into a single, concise summary of the legal proceeding.\n\n" + "\n\n".join(full_chunk_summaries)
        rr_summary = generate_summary_groq(rr_final_prompt)
        full_summary = generate_summary_groq(full_final_prompt)
        # Add clear section headers and separators for distinction
        rr_summary_formatted = (
            "==== RR-BASED SUMMARY (Facts, Issues, Decision) ===="
            "\n\n" + rr_summary.strip() + "\n\n==== END RR-BASED SUMMARY ===="
        )
        full_summary_formatted = (
            "==== FULL-TEXT SUMMARY (General/Abstractive) ===="
            "\n\n" + full_summary.strip() + "\n\n==== END FULL-TEXT SUMMARY ===="
        )
        results.append({
            'file': fname,
            'rr_summary': rr_summary_formatted,
            'full_summary': full_summary_formatted
        })
    return pd.DataFrame(results)

def main():
    df = summarize_transcripts()
    output_path = os.path.join(diary_folder, 'summaries_output.csv')
    df.to_csv(output_path, index=False)
    print(f"Summaries saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    main()
