import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier

# -------------------------------
# Audio utilities
# -------------------------------
def load_audio(path, sr=16000):
    wav, orig_sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav = librosa.resample(y=wav.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)

def compute_logmel(wav, sr=16000, n_mels=64, n_fft=1024, hop_length=256):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-9)
    return torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)

# -------------------------------
# ASV wrapper
# -------------------------------
from speechbrain.inference import EncoderClassifier
import numpy as np

class ASVWrapper:
    def __init__(self):
        # Load ECAPA-TDNN model from HuggingFace
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

    def get_embedding(self, wav_path):
        # load_audio now returns only the signal
        signal = self.classifier.load_audio(wav_path)
        emb = self.classifier.encode_batch(signal).mean(0).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)

    @staticmethod
    def cosine(emb1, emb2):
        return float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
        )



# -------------------------------
# Streamlit App
# -------------------------------
st.title("🔊 Voice Authentication + Liveness Detection")
st.write("Upload an enrollment voice and a test voice to check authentication with spoof detection.")

enroll_file = st.file_uploader("Upload Enrollment Audio (WAV)", type=["wav"])
test_file = st.file_uploader("Upload Test Audio (WAV)", type=["wav"])

if enroll_file and test_file:
    # Save uploaded files
    with open("enroll.wav", "wb") as f:
        f.write(enroll_file.read())
    with open("test.wav", "wb") as f:
        f.write(test_file.read())

    st.audio("enroll.wav", format="audio/wav")
    st.audio("test.wav", format="audio/wav")

    # ASV processing
    asv = ASVWrapper()
    e1 = asv.get_embedding("enroll.wav")
    e2 = asv.get_embedding("test.wav")
    asv_score = (asv.cosine(e1, e2) + 1) / 2  # Normalize -1..1 -> 0..1

    # Simple CM (fake placeholder score: random or dummy)
    wav = load_audio("test.wav")
    logmel = compute_logmel(wav)
    cm_score = np.clip(np.random.uniform(0.4, 0.9), 0, 1)  # <- Replace with trained CM model

    # Fusion
    cm_weight = 0.5
    fused_score = (1 - cm_weight) * asv_score + cm_weight * cm_score

    st.subheader("🔎 Results")
    st.write(f"**ASV Score (speaker similarity):** {asv_score:.3f}")
    st.write(f"**CM Score (genuine probability):** {cm_score:.3f}")
    st.write(f"**Final Fused Score:** {fused_score:.3f}")

    if fused_score > 0.6:
        st.success("✅ Verified as genuine speaker")
    else:
        st.error("❌ Rejected (spoof or wrong speaker)")
