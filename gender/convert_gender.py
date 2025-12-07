import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import os

def get_voice_info(path):
    """
    Phân loại chi tiết: Male Deep, Male High, Female Warm, Female High
    """
    try:
        s = parselmouth.Sound(path)
        # Bước 1: Dò tìm sơ bộ để quyết định Floor
        p_dummy = s.to_pitch_ac(time_step=None, pitch_floor=50.0, very_accurate=True)
        q50 = call(p_dummy, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        
        # Quyết định Floor dựa trên độ cao sơ bộ
        # - Nam rất trầm (<110Hz): Cần floor thấp (50Hz) để bắt hết dấu nặng miền Bắc
        # - Nam cao/Nữ (>150Hz): Cần floor cao (90-100Hz) để khử rè
        if np.isnan(q50):
            return np.nan, None, "unknown"
            
        if q50 < 120:
            actual_floor = 50.0  # Quan trọng cho giọng Nam miền Bắc
        elif q50 < 180:
            actual_floor = 75.0
        else:
            actual_floor = 100.0

        p = s.to_pitch_ac(time_step=None, pitch_floor=actual_floor, pitch_ceiling=600.0, very_accurate=True)
        median = call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        
        if np.isnan(median): return np.nan, None, "unknown"

        # Phân loại chi tiết 4 nhóm
        if median < 135:
            v_type = "male_deep"  # Nam trầm/trung niên
        elif median < 175:
            v_type = "male_high"  # Nam cao/già/thanh niên
        elif median < 230:
            v_type = "female_warm"
        else:
            v_type = "female_high"

        return median, actual_floor, v_type
    except Exception as e:
        print(f"Error info for {path}: {e}")
        return np.nan, 50.0, "unknown"

def align_source_pitch_to_target(src_path, tgt_path, debug_dir="debug_praat_output"):
    os.makedirs(debug_dir, exist_ok=True)

    src_med, src_floor, src_type = get_voice_info(src_path)
    tgt_med, _, tgt_type = get_voice_info(tgt_path)

    if np.isnan(src_med) or np.isnan(tgt_med):
        return src_path

    # --- CẤU HÌNH THAM SỐ ---
    target_pitch = tgt_med
    pitch_range_factor = 1.0
    formant_shift_ratio = 1.0
    pitch_ratio = tgt_med / src_med
    
    mode = "Default"

    # KIỂM TRA LOẠI CHUYỂN ĐỔI
    is_cross_gender = ("male" in src_type and "female" in tgt_type) or \
                      ("female" in src_type and "male" in tgt_type)

    # 1. CASE: NAM <-> NAM (Xử lý kỹ Nam thấp vs Nam cao/già)
    if "male" in src_type and "male" in tgt_type:
        mode = "Male-to-Male (Detailed)"
        
        # Nếu Pitch chênh lệch quá ít (< 5%), bình thường sẽ bỏ qua.
        # NHƯNG nếu user muốn test B30 -> B60, ta phải ép đổi Formant để tạo cảm giác già/trẻ.
        
        # Chiến thuật:
        # - Nam Trầm -> Nam Cao (Giả lập già đi/trẻ hóa): Cần tăng Formant để làm mỏng giọng
        # - Nam Cao -> Nam Trầm: Cần giảm Formant để làm dày giọng
        
        # Dùng hệ số mũ mạnh hơn 0.25 (V4) lên 0.35 để thấy rõ sự khác biệt
        formant_shift_ratio = pow(pitch_ratio, 0.35)
        
        # Ép buộc thay đổi Target Pitch 100% (không blend nữa) để giống mẫu hơn
        target_pitch = tgt_med 
        
        # Xử lý Pitch Range:
        # Người già (thường là male_high trong ngữ cảnh này) có range hẹp hơn hoặc thất thường.
        # Nếu đang convert sang Male Deep (thường là trung niên khỏe), tăng range lên chút
        if tgt_type == "male_deep" and src_type == "male_high":
             pitch_range_factor = 1.1 # Tăng độ hào sảng
        elif tgt_type == "male_high" and src_type == "male_deep":
             pitch_range_factor = 0.9 # Giảm độ hào sảng, làm giọng mỏng lại
             
    # 2. CASE: CROSS GENDER (Giữ nguyên V4 vì bạn bảo đã ổn hơn)
    elif is_cross_gender:
        mode = "Cross-Gender"
        target_pitch = tgt_med
        formant_shift_ratio = pow(pitch_ratio, 0.45) 
        if "male" in src_type: 
            pitch_range_factor = 0.9

    # 3. CASE: FEMALE INTERNAL (Giữ nguyên V4)
    elif "female" in src_type and "female" in tgt_type:
        mode = "Female-to-Female"
        target_pitch = tgt_med
        if src_type == "female_warm" and tgt_type == "female_high":
            formant_shift_ratio = pow(pitch_ratio, 0.6) # Ép chóe
        else:
            formant_shift_ratio = pow(pitch_ratio, 0.5)

    # --- KHỬ RÈ VÀ GIỚI HẠN ---
    
    # Sử dụng src_floor đã tính toán động ở hàm get_voice_info
    # (50Hz cho nam trầm, 100Hz cho nữ) -> Đây là chìa khóa khử rè cho Nam Bắc
    convert_floor = src_floor 

    # Giới hạn Formant để không bị méo tiếng
    formant_shift_ratio = np.clip(formant_shift_ratio, 0.7, 1.45)

    # In log để debug
    print(f"   [V5] {os.path.basename(src_path)} ({src_type}) -> ({tgt_type})")
    print(f"        Mode: {mode} | Floor: {convert_floor}Hz | Range: {pitch_range_factor}")

    try:
        s = parselmouth.Sound(src_path)
        o = call(s, "Change gender", 
                 convert_floor, 
                 600.0, 
                 formant_shift_ratio, 
                 target_pitch, 
                 pitch_range_factor, 
                 1.0)
        
        y2 = np.asarray(o.values[0], dtype=np.float32)
        sr2 = int(o.sampling_frequency)
        
        new_filename = f"v5_{src_type}2{tgt_type}_{os.path.basename(src_path)}"
        out_path = os.path.join(debug_dir, new_filename)
        sf.write(out_path, y2, sr2)
        
        return out_path
    except Exception as e:
        print(f"   [Praat] Failed: {e}")
        return src_path