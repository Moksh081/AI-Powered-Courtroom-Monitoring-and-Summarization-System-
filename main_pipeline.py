"""
main_pipeline.py

This script connects the Speech-to-Text, Summarization, and Bias Detection modules.
Input: Audio file path
Output: Diarized transcript, summary, and bias report
"""
import os
import sys
import subprocess

# Paths to module scripts
SPEECH_TO_TEXT_SCRIPT = os.path.join('Speech_to_Text', 'speech_text.py')
SUMMARY_SCRIPT = os.path.join('Summary', 'diarized_summarizer.py')
BIAS_SCRIPT = os.path.join('Bias', 'refined_bias_detector.py')

# Default file names used by modules
DIARIZED_TRANSCRIPT = os.path.join('Speech_to_Text', 'diarized_transcript_Audio3.txt')


def run_speech_to_text(audio_path):
    print(f"[1/3] Running Speech-to-Text on {audio_path}...")
    result = subprocess.run([sys.executable, SPEECH_TO_TEXT_SCRIPT, audio_path], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print("Speech-to-Text failed:", result.stderr)
        sys.exit(1)
    print("Speech-to-Text completed.")
    return DIARIZED_TRANSCRIPT


def run_summarizer(transcript_path):
    print(f"[2/3] Running Summarizer on {transcript_path}...")
    # The diarized_summarizer.py script outputs summaries_output.csv in Speech_to_Text
    print(f"[2/3] Running Summarizer on {transcript_path}...")
    result = subprocess.run([
        sys.executable, SUMMARY_SCRIPT
    ], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print("Summarizer failed:", result.stderr)
        sys.exit(1)
    print("Summarization completed. Output: Speech_to_Text/summaries_output.csv")
    summary_output = os.path.join('Speech_to_Text', 'summaries_output.csv')
    if os.path.exists(summary_output):
        with open(summary_output, 'r', encoding='utf-8', errors='replace') as f:
            print("\n===== SUMMARIES (CSV) =====\n" + f.read() + "\n===========================\n")
    return summary_output


def run_bias_detector(transcript_path):
    print(f"[3/3] Running Bias Detector on {transcript_path}...")
    bias_output = os.path.join('Bias', 'bias_report.txt')
    result = subprocess.run([
        sys.executable, BIAS_SCRIPT, transcript_path, bias_output
    ], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print("Bias Detection failed:", result.stderr)
        sys.exit(1)
    print("Bias Detection completed. Output:", bias_output)
    # Print bias report to terminal
    if os.path.exists(bias_output):
        with open(bias_output, 'r', encoding='utf-8', errors='replace') as f:
            print("\n===== BIAS REPORT =====\n" + f.read() + "\n=======================\n")
    return bias_output


def interactive_speaker_mapping(transcript_path, mapped_path):
    import re
    speakers = set()
    with open(transcript_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            match = re.match(r'<spk:([A-Z])>', line)
            if match:
                speakers.add(match.group(1))
    speakers = sorted(speakers)
    print("\nDetected speakers:")
    mapping = {}
    for spk in speakers:
        role = input(f"Enter name/designation for speaker '{spk}': ")
        mapping[spk] = role.strip() if role.strip() else spk
    # Replace in transcript
    with open(transcript_path, 'r', encoding='utf-8', errors='replace') as fin, \
         open(mapped_path, 'w', encoding='utf-8', errors='replace') as fout:
        for line in fin:
            for spk, role in mapping.items():
                line = re.sub(rf'<spk:{spk}>', f'<spk:{role}>', line)
            fout.write(line)
    print(f"Mapped transcript saved to: {mapped_path}")
    return mapped_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <audio_file_path>")
        sys.exit(1)
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    # Step 1: Speech-to-Text
    transcript_path = run_speech_to_text(audio_path)

    # Step 1.5: Interactive speaker mapping

    # Use a mapped transcript name specific to Audio3 for clarity
    mapped_transcript = os.path.join('Speech_to_Text', 'diarized_transcript_Audio3_mapped.txt')
    mapped_transcript = interactive_speaker_mapping(transcript_path, mapped_transcript)

    # Step 2: Summarization
    summary_path = run_summarizer(mapped_transcript)

    # Step 3: Bias Detection
    run_bias_detector(mapped_transcript)

    print("\nPipeline completed. Check outputs in respective folders.")


if __name__ == "__main__":
    main()
