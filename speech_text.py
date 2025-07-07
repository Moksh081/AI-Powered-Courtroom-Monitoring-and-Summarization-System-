import os
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import assemblyai as aai
import threading
import sys

# === ASSEMBLYAI SETUP ===
API_KEY = "#"
aai.settings.api_key = API_KEY

def list_audio_devices():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"[{i}] {dev['name']} (Inputs: {dev['max_input_channels']} / Outputs: {dev['max_output_channels']})")
    return devices

def record_audio(output_file, duration=30, samplerate=44100, channels=1, device=None):
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    print("Recording will start in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print(f"🎙️ Recording up to {duration}s on device {device or 'default'}...")
    rec = np.zeros((int(duration * samplerate), channels), dtype='int16')
    stop_flag = threading.Event()
    actual_samples = [0]
    start_time = [0]
    end_time = [0]

    def _record():
        nonlocal rec
        start_time[0] = time.time()
        rec = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16', device=device)
        sd.wait()
        end_time[0] = time.time()
        stop_flag.set()

    thread = threading.Thread(target=_record)
    thread.start()
    print("Press Enter to stop recording early...")
    input()
    if not stop_flag.is_set():
        sd.stop()
        end_time[0] = time.time()
        stop_flag.set()
        print("Recording stopped manually.")
    thread.join()
    # Calculate actual duration and samples
    actual_duration = max(0, end_time[0] - start_time[0])
    actual_samples[0] = int(actual_duration * samplerate)
    if actual_samples[0] > 0 and actual_samples[0] < rec.shape[0]:
        rec = rec[:actual_samples[0]]
    wav.write(output_file, samplerate, rec)
    print(f"✅ Saved: {output_file}")
    if np.max(np.abs(rec)) < 100:
        print("⚠️ Warning: very low amplitude – check mic!")
    return output_file

def play_audio(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    rate, data = wav.read(file_path)
    print(f"🔊 Playing back {file_path}...")
    sd.play(data, rate)
    sd.wait()
    print("✅ Playback done.")

# === TRANSCRIPTION + DIARIZATION ===
def diarize_and_transcribe(wav_path):
    print("📝 Sending to AssemblyAI for diarization…")
    config = aai.TranscriptionConfig(
        speech_model   = aai.SpeechModel.best,
        speaker_labels = True
    )
    transcriber = aai.Transcriber(config=config)
    transcript  = transcriber.transcribe(wav_path)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    print("\n=== DIARIZED TRANSCRIPT ===")
    lines = []
    for utt in transcript.utterances:
        line = f"<spk:{utt.speaker}> {utt.text.strip()}"
        print(line)
        lines.append(line)
    return "\n".join(lines)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Usage: python speech_text.py <audio_file_path> [output_transcript_path]
    if len(sys.argv) < 2:
        print("Usage: python speech_text.py <audio_file_path> [output_transcript_path]")
        sys.exit(1)
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)
    output_transcript = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "diarized_transcript.txt")
    transcript_text = diarize_and_transcribe(audio_path)
    with open(output_transcript, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    print(f"\n📄 Diarized transcript saved to: {output_transcript}")
