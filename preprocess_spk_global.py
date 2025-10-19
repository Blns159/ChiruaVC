"""
Preprocess speaker embeddings - Global per-speaker approach (Memory Efficient)
Tạo 1 embedding cho mỗi speaker bằng cách average embeddings theo batch
"""
import os
import sys
from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from pathlib import Path
import numpy as np
from tqdm import tqdm
import glob


def compute_global_speaker_embedding_batched(speaker_dir, weights_fpath, batch_size=10):
    """
    Tính global speaker embedding theo batch để tránh OOM
    
    Args:
        speaker_dir: Đường dẫn đến thư mục speaker
        weights_fpath: Path đến speaker encoder checkpoint
        batch_size: Số audio files xử lý mỗi batch (giảm nếu vẫn OOM)
    Returns:
        speaker_embedding: numpy array (256,)
    """
    # Load speaker encoder một lần
    encoder = SpeakerEncoder(weights_fpath)
    
    # Tìm tất cả wav files của speaker
    wav_files = glob.glob(os.path.join(speaker_dir, '*.wav'))
    
    if len(wav_files) == 0:
        print(f"[WARNING] No wav files found in {speaker_dir}")
        return None
    
    print(f"  Found {len(wav_files)} audio files")
    
    # Xử lý theo batch và tích lũy embeddings
    all_embeddings = []
    
    for i in tqdm(range(0, len(wav_files), batch_size), 
                  desc="  Processing batches", leave=False):
        batch_files = wav_files[i:i+batch_size]
        
        # Load và preprocess batch này
        batch_wavs = []
        for wav_path in batch_files:
            try:
                fpath = Path(wav_path)
                wav = preprocess_wav(fpath)
                batch_wavs.append(wav)
            except Exception as e:
                print(f"  [ERROR] Failed to load {wav_path}: {e}")
                continue
        
        if len(batch_wavs) == 0:
            continue
        
        # Compute embeddings cho batch này
        try:
            # Embed từng utterance trong batch
            batch_embeddings = []
            for wav in batch_wavs:
                embed = encoder.embed_utterance(wav)
                batch_embeddings.append(embed)
            
            # Add vào list tổng
            all_embeddings.extend(batch_embeddings)
            
            # Clear memory
            del batch_wavs
            del batch_embeddings
            
        except Exception as e:
            print(f"  [ERROR] Failed to compute embeddings for batch: {e}")
            continue
    
    if len(all_embeddings) == 0:
        print(f"[ERROR] No valid embeddings computed for {speaker_dir}")
        return None
    
    # Average tất cả embeddings
    print(f"  Averaging {len(all_embeddings)} embeddings...")
    speaker_embedding = np.mean(all_embeddings, axis=0)
    
    # Normalize
    speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)
    
    return speaker_embedding


def preprocess_all_speakers(audio_root_dir, weights_fpath, batch_size=10):
    """
    Preprocess tất cả speakers với batch processing
    
    Args:
        audio_root_dir: Root directory chứa các thư mục speaker
        weights_fpath: Path đến speaker encoder checkpoint
        batch_size: Số audio files xử lý mỗi batch
    """
    # Tìm tất cả thư mục SPEAKER_*
    speaker_dirs = glob.glob(os.path.join(audio_root_dir, 'SPEAKER_*'))
    speaker_dirs = [d for d in speaker_dirs if os.path.isdir(d)]
    speaker_dirs = sorted(speaker_dirs)
    
    print(f"Found {len(speaker_dirs)} speakers in {audio_root_dir}")
    print(f"Speaker encoder checkpoint: {weights_fpath}")
    print(f"Batch size: {batch_size} (reduce if OOM occurs)")
    print()
    
    success_count = 0
    failed_speakers = []
    
    for idx, speaker_dir in enumerate(speaker_dirs, 1):
        speaker_id = os.path.basename(speaker_dir)
        print(f"\n[{idx}/{len(speaker_dirs)}] Processing {speaker_id}...")
        
        try:
            # Compute global speaker embedding với batching
            speaker_embedding = compute_global_speaker_embedding_batched(
                speaker_dir, weights_fpath, batch_size=batch_size
            )
            
            if speaker_embedding is None:
                failed_speakers.append(speaker_id)
                continue
            output_dir = "dataset/spk"
            os.makedirs(output_dir, exist_ok=True)
            # Save embedding: SPEAKER_*/SPEAKER_*.npy
            output_path = os.path.join(output_dir, f"{speaker_id}.npy")
            np.save(output_path, speaker_embedding, allow_pickle=False)
            print(f"  ✓ Saved to: {output_path}")
            print(f"  Embedding shape: {speaker_embedding.shape}")
            
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {speaker_id}: {e}")
            failed_speakers.append(speaker_id)
            continue
    
    print("\n" + "="*70)
    print(f"COMPLETED!")
    print(f"Successfully processed: {success_count}/{len(speaker_dirs)} speakers")
    
    if failed_speakers:
        print(f"\nFailed speakers ({len(failed_speakers)}):")
        for spk in failed_speakers:
            print(f"  - {spk}")
    
    return success_count, failed_speakers


def verify_embeddings(audio_root_dir):
    """
    Verify rằng tất cả speakers đều có global embedding
    """
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    speaker_dirs = glob.glob(os.path.join(audio_root_dir, 'SPEAKER_*'))
    speaker_dirs = [d for d in speaker_dirs if os.path.isdir(d)]
    speaker_dirs = sorted(speaker_dirs)
    
    missing_embeddings = []
    
    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)
        embedding_path = os.path.join("dataset/spk", f"{speaker_id}.npy")
        
        if not os.path.exists(embedding_path):
            missing_embeddings.append(speaker_id)
        else:
            # Check embedding shape
            emb = np.load(embedding_path)
            if emb.shape != (256,):
                print(f"[WARNING] {speaker_id}: Invalid embedding shape {emb.shape}")
    
    if missing_embeddings:
        print(f"\nMissing embeddings for {len(missing_embeddings)} speakers:")
        for spk in missing_embeddings:
            print(f"  - {spk}")
    else:
        print(f"\nAll {len(speaker_dirs)} speakers have valid embeddings!")
    
    return len(missing_embeddings) == 0


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    audio_dir = 'dataset/audio_vlsp'
    spk_encoder_ckpt = 'speaker_encoder/ckpt/pretrained_bak_5805000.pt'
    batch_size = 3  # Giảm xuống 5 hoặc 3 nếu vẫn OOM
    verify_only = False
    # =======================================================
    
    print("Configuration:")
    print(f"  Audio directory: {audio_dir}")
    print(f"  Speaker encoder: {spk_encoder_ckpt}")
    print(f"  Batch size: {batch_size}")
    print(f"  Mode: {'Verify only' if verify_only else 'Generate embeddings'}")
    print()
    
    # Verify paths exist
    if not os.path.exists(audio_dir):
        print(f"[ERROR] Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    if not os.path.exists(spk_encoder_ckpt):
        print(f"[ERROR] Speaker encoder checkpoint not found: {spk_encoder_ckpt}")
        sys.exit(1)
    
    if verify_only:
        verify_embeddings(audio_dir)
    else:
        success_count, failed_speakers = preprocess_all_speakers(
            audio_dir, 
            spk_encoder_ckpt,
            batch_size=batch_size
        )
        verify_embeddings(audio_dir)
    
    print("\nDONE!")
    sys.exit(0)