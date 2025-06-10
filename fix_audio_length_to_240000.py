# fix_audio_length_to_240000.py 
# 인자 반영

import os
import sys
import torch
import torchaudio
import argparse
from tqdm import tqdm

# === 인자 parsing: default 값을 None으로 변경하여 인자 전달 여부 확인 ===
parser = argparse.ArgumentParser(description="오디오 파일 길이를 15초(240000 샘플)로 고정합니다.")
parser.add_argument("--mark_version", type=str, default=None, 
                    help="모델 버전 (예: mark2.9.0). 이 버전에 따라 입/출력 폴더가 결정됩니다.")
args = parser.parse_args()

# === 기본 경로 설정 ====
# 현재 스크립트가 있는 폴더가 기준이 됨 ( /content/YourProject/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === 보정 파라미터 ===
TARGET_SAMPLE_RATE = 16000 # sample rate 도 제한을 해서 주파수 해상도를 일정하게 맞추기
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * 15  # 240,000 samples (딱 15초기준)

def fix_wav_length(wav_path, save_path):
    try:
        waveform, sr = torchaudio.load(wav_path)

        # 샘플레이트 맞추기(보정하기)
        if sr != TARGET_SAMPLE_RATE:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resample(waveform)

        # 자르기 or 패딩
        num_samples = waveform.shape[1]
        if num_samples > TARGET_NUM_SAMPLES:
            fixed_waveform = waveform[:, :TARGET_NUM_SAMPLES]
        elif num_samples < TARGET_NUM_SAMPLES:
            pad_len = TARGET_NUM_SAMPLES - num_samples
            pad_tensor = torch.zeros((waveform.shape[0], pad_len))
            fixed_waveform = torch.cat([waveform, pad_tensor], dim=1)
        else:
            fixed_waveform = waveform

        torchaudio.save(save_path, fixed_waveform, TARGET_SAMPLE_RATE)
        # 성공 시 True 반환
        return True
    except Exception as e:
        print(f"\n[ERROR] 파일 처리 중 오류 발생 {os.path.basename(wav_path)}: {e}")
        # 실패 시 False 반환
        return False


def process_all(mark_version):
    # mark_version이 제공되지 않으면 에러 발생
    if mark_version is None:
        print("[CRITICAL ERROR] --mark_version 인자가 반드시 필요합니다. (예: --mark_version mark2.9.0)")
        sys.exit(1) # 오류 코드로 종료
        
    
    input_dir = os.path.join(BASE_DIR, "data")
    output_dir = os.path.join(BASE_DIR, "data_wav")
    
    print(f"[INFO] 오디오 길이 보정 시작: '{input_dir}' -> '{output_dir}'")
    
    if not os.path.isdir(input_dir):
        print(f"[CRITICAL ERROR] 입력 폴더를 찾을 수 없습니다: {input_dir}")
        print("파일 구조가 '.../{mark_version}/data/' 형태인지 확인해주세요.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith((".wav", ".mp3", ".flac"))]
    
    if not file_list:
        print(f"[Warning] 입력 폴더에 처리할 오디오 파일이 없습니다: {input_dir}")
        return

    success_count = 0
    for fname in tqdm(file_list, desc=f"Processing {mark_version}", unit="file"):
        in_path = os.path.join(input_dir, fname)
        # 출력 파일 이름은 항상 .wav로 고정
        out_fname = os.path.splitext(fname)[0] + ".wav"
        out_path = os.path.join(output_dir, out_fname)
        
        if fix_wav_length(in_path, out_path):
            success_count += 1
    
    print(f"\n[DONE] 오디오 보정 완료. 총 {len(file_list)}개 파일 중 {success_count}개 성공.")

if __name__ == "__main__":
    # 스크립트 실행 시 args.mark_version 값을 process_all 함수에 전달
    process_all(mark_version=args.mark_version)