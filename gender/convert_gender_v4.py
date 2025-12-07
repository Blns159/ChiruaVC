# import librosa
# import numpy as np
# import soundfile as sf
# import parselmouth
# from parselmouth.praat import call

# def detect_and_convert(path):
#     y, sr = librosa.load(path, sr=None, mono=True)
#     f0, vf, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
#     v = f0[vf]
#     if len(v) == 0:
#         return "unknown", path

#     g = "male" if np.nanmean(v) < 181 else "female"

#     def convert(pitch_ratio, form_ratio):
#         s = parselmouth.Sound(path)
#         p = s.to_pitch()
#         m = call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

#         if np.isnan(m):
#             return y, sr

#         n = m * pitch_ratio
#         o = call(s, "Change gender", 75, 600, form_ratio, n, 1.0, 1.0)

#         # o.values là mảng shape (1, N) hoặc (N,)
#         y2 = np.asarray(o.values[0], dtype=np.float32)

#         # sr2 là float → cần ép kiểu int
#         sr2 = int(o.sampling_frequency)

#         return y2, sr2

#     if g == "male":
#         y2, sr2 = convert(1.45, 1.12)
#         out = path.replace(".wav", "_female.wav")
#         sf.write(out, y2, sr2)
#         return g, out

#     else:
#         y2, sr2 = convert(0.72, 0.85)
#         out = path.replace(".wav", "_male.wav")
#         sf.write(out, y2, sr2)
#         return g, out


import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import os

def get_voice_info(path):
    """
    Phân tích kỹ hơn: Median Pitch và Max Pitch (để phát hiện giọng Chóe/Hét)
    """
    try:
        s = parselmouth.Sound(path)
        # Tự động điều chỉnh floor dựa trên dự đoán sơ bộ để tránh nhiễu
        # Lấy thử pitch
        p_dummy = s.to_pitch_ac(time_step=None, pitch_floor=50.0, very_accurate=True)
        q50 = call(p_dummy, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        
        # Nếu giọng cao (>160Hz), đặt floor cao hơn (100Hz) để khử tiếng rè trầm (rumble)
        # Nếu giọng thấp, đặt floor 60Hz
        actual_floor = 100.0 if (not np.isnan(q50) and q50 > 160) else 60.0
        
        p = s.to_pitch_ac(time_step=None, pitch_floor=actual_floor, pitch_ceiling=600.0, very_accurate=True)
        
        median = call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        if np.isnan(median): return np.nan, np.nan, "unknown"

        # Phân loại giọng
        if median < 175:
            v_type = "male"
        elif median < 230:
            v_type = "female_warm" # Nữ giọng ấm/trung
        else:
            v_type = "female_high" # Nữ giọng chóe/cao

        return median, p, v_type
    except Exception as e:
        print(f"Error info for {path}: {e}")
        return np.nan, None, "unknown"

def align_source_pitch_to_target(src_path, tgt_path, debug_dir="debug_praat_output"):
    os.makedirs(debug_dir, exist_ok=True)

    src_med, src_pitch_obj, src_type = get_voice_info(src_path)
    tgt_med, tgt_pitch_obj, tgt_type = get_voice_info(tgt_path)

    if np.isnan(src_med) or np.isnan(tgt_med):
        return src_path

    # --- CẤU HÌNH THAM SỐ (Parameters Tuning) ---
    
    # Mặc định
    target_pitch = tgt_med
    pitch_range_factor = 1.0
    formant_shift_ratio = 1.0
    
    # Tính tỷ lệ cơ bản
    pitch_ratio = tgt_med / src_med
    
    # Logic phân loại chuyển đổi
    conversion_mode = "Normal"

    # CASE 1: CROSS GENDER (NAM <-> NỮ) -> Cần mạnh tay
    if ("male" in src_type and "female" in tgt_type) or \
       ("female" in src_type and "male" in tgt_type):
        
        conversion_mode = "Cross-Gender (Strong)"
        
        # Ép 100% theo pitch của target (Không dùng dampening nữa)
        target_pitch = tgt_med 
        
        # Tăng Formant Shift mạnh hơn để đổi chất giọng hoàn toàn
        # Công thức chuẩn là pow(ratio, 0.33). Ở đây tăng lên 0.4 hoặc 0.45 để ép chất giọng.
        # Nếu Nam -> Nữ: Pitch tăng -> Formant phải tăng theo mạnh để giọng thanh
        # Nếu Nữ -> Nam: Pitch giảm -> Formant giảm mạnh để giọng ồm
        formant_shift_ratio = pow(pitch_ratio, 0.45) 
        
        # Giảm rung giọng khi Cross Gender (thường gây rè nếu giữ nguyên range của Nam khi lên Nữ)
        if "male" in src_type: # Nam -> Nữ
            pitch_range_factor = 0.9 # Giảm biên độ một chút cho mượt
        
    # CASE 2: NỮ ẤM <-> NỮ CHÓE (Internal Female Conversion)
    elif "female" in src_type and "female" in tgt_type and src_type != tgt_type:
        conversion_mode = "Female Warm<->High"
        
        target_pitch = tgt_med
        
        # Bí quyết: Chỉnh Formant độc lập với Pitch
        if src_type == "female_warm" and tgt_type == "female_high":
            # Ấm -> Chóe: Cần làm sáng giọng hơn mức bình thường
            # Pitch tăng 1 ít, nhưng Formant cần tăng nhiều hơn để tạo cảm giác "gắt"
            formant_shift_ratio = pow(pitch_ratio, 0.6) # Tăng hệ số lũy thừa
        else:
            # Chóe -> Ấm: Cần làm tối giọng
            formant_shift_ratio = pow(pitch_ratio, 0.5)

    # CASE 3: SAME TYPE (Nam-Nam, Nữ Ấm-Nữ Ấm...) -> Nhẹ nhàng để tự nhiên
    else:
        # Nếu chênh lệch ít thì bỏ qua để giữ chất lượng gốc tốt nhất
        if 0.9 < pitch_ratio < 1.1:
            return src_path
            
        conversion_mode = "Same-Type (Soft)"
        # Soft blend (80% target)
        target_pitch = src_med * (pitch_ratio * 0.8 + 0.2)
        formant_shift_ratio = pow(pitch_ratio, 0.25) # Formant đổi rất ít

    # --- KHỬ RÈ (QUAN TRỌNG) ---
    # Pitch floor quyết định việc Praat có bắt nhầm tiếng thở thành tiếng rè hay không
    # Nếu Source là Nữ, Floor phải cao (100Hz-120Hz). Nam thì 65Hz.
    if "female" in src_type:
        convert_floor = 100.0
    else:
        convert_floor = 65.0 # Nâng nhẹ từ 50 lên 65

    # Giới hạn Formant không quá lố (tránh tiếng Chipmunk hoặc quái vật)
    formant_shift_ratio = np.clip(formant_shift_ratio, 0.75, 1.45)

    print(f"   [Praat V4] {os.path.basename(src_path)} [{src_type}] -> [{tgt_type}]")
    print(f"      Mode: {conversion_mode} | Pitch: {tgt_med:.0f} | FormantRatio: {formant_shift_ratio:.2f}")

    try:
        s = parselmouth.Sound(src_path)
        
        # Change gender arguments:
        # 1. min pitch (dùng convert_floor đã tối ưu)
        # 2. max pitch (600)
        # 3. formant shift
        # 4. new pitch median
        # 5. pitch range factor
        # 6. duration factor
        o = call(s, "Change gender", 
                 convert_floor, 
                 600.0, 
                 formant_shift_ratio, 
                 target_pitch, 
                 pitch_range_factor, 
                 1.0)
        
        y2 = np.asarray(o.values[0], dtype=np.float32)
        sr2 = int(o.sampling_frequency)
        
        new_filename = f"v4_{src_type}2{tgt_type}_{os.path.basename(src_path)}"
        out_path = os.path.join(debug_dir, new_filename)
        sf.write(out_path, y2, sr2)
        
        return out_path
    except Exception as e:
        print(f"   [Praat] Failed: {e}")
        return src_path