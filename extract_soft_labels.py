# extract_soft_labels.py

import os
import sys
import pickle
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse

# --- 경로 설정 및 모듈 임포트 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead
from vild_parser_teacher import AudioParser
from seed_utils import set_seed


#  Teacher 모델 래퍼 클래스 (예측을 위한 편의 기능)
class TeacherPredictor:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # 1. Teacher 모델 구성 요소 로드
        self.encoder = SimpleAudioEncoder(config).to(device)
        self.classifier = ViLDTextHead(config).to(device)
        
        # 2. Teacher 모델의 학습된 가중치 로드
        encoder_path = os.path.join(BASE_DIR, f"best_teacher_encoder_{config.mark_version}.pth")
        classifier_path = os.path.join(BASE_DIR, f"best_teacher_classifier_{config.mark_version}.pth")
        
        if not os.path.exists(encoder_path) or not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Teacher 모델 가중치 파일을 찾을 수 없습니다: {encoder_path} 또는 {classifier_path}")

        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        
        # 3. Text Embedding 생성 (teacher_train.py와 동일한 로직)
        text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        classes = config.get_classes_for_text_prompts()
        prompt_texts = [f"A sound of {cls.replace('_', ' ')} in the room" for cls in classes]
        self.text_emb = torch.tensor(text_model.encode(prompt_texts), dtype=torch.float).to(device)
        
        # Teacher 모델은 평가 모드로 고정
        self.encoder.eval()
        self.classifier.eval()
        print("[INFO] TeacherPredictor가 성공적으로 초기화되었습니다.")

    @torch.no_grad() # 예측 시에는 그래디언트 계산이 필요 없음
    def predict_soft_labels(self, mel_segments):
        """
        Mel-spectrogram 세그먼트 리스트를 받아 Soft Label(확률 분포) 리스트를 반환합니다.
        
        Args:
            mel_segments (list of torch.Tensor): 각 요소는 [1, 64, 101] 형태의 텐서
            
        Returns:
            list of list of float: 각 세그먼트에 대한 Softmax 확률 분포 리스트
        """
        if not mel_segments:
            return []
            
        # 여러 세그먼트를 하나의 배치로 묶어 효율적으로 처리
        batch_mel = torch.stack(mel_segments, dim=0).to(self.device)
        
        # Teacher 모델을 통해 로짓 계산
        region_emb = self.encoder(batch_mel)
        logits = self.classifier(region_emb, self.text_emb)
        
        # Softmax를 적용하여 Soft Label (확률 분포) 생성
        soft_labels = F.softmax(logits, dim=1)
        
        # 결과를 CPU로 옮기고 리스트 형태로 변환하여 반환
        return soft_labels.cpu().tolist()



#  메인 추출 함수
def extract_soft_labels(mark_version="mark2.9.0"):
    set_seed(42)
    config = AudioViLDConfig(mark_version=mark_version)
    device = torch.device(config.device)

    csv_path = os.path.join(BASE_DIR, f"dataset_index_{mark_version}.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV 파일 없음: {csv_path}")
        return 1

    # Teacher 모델과 오디오 파서 초기화
    try:
        teacher = TeacherPredictor(config, device)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
        
    parser = AudioParser(config, segment_mode=True)
    
    soft_label_entries = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        # CSV 파일의 모든 행을 리스트로 읽어 tqdm 진행률 표시
        reader = list(csv.DictReader(f))
        for row in tqdm(reader, desc=f"Generating soft labels for {mark_version}"):
            path = row['path']

            # Teacher 파서를 사용하여 오디오를 세그먼트로 분할
            segment_list = parser.load_and_segment(path)

            if not segment_list:
                print(f"[Warning] 세그먼트 없음: {path}")
                continue

            # Teacher 모델을 통해 Soft Label 예측
            predicted_soft_labels = teacher.predict_soft_labels(segment_list)

            if not predicted_soft_labels:
                print(f"[Warning] Soft label 예측 실패: {path}")
                continue

            # 파일 경로와 해당 파일의 모든 세그먼트에 대한 soft_labels를 저장
            soft_label_entries.append({
                "path": path,
                "soft_labels": predicted_soft_labels
            })

    # 결과를 .pkl 파일로 저장
    out_path = os.path.join(BASE_DIR, f"soft_labels_{mark_version}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(soft_label_entries, f)

    print(f"\n[완료] soft_labels 저장됨: {out_path}")
    print(f"  - 총 {len(soft_label_entries)}개 파일의 soft labels 추출 완료.")
    
    # 예시 출력
    if soft_label_entries:
        sample_entry = soft_label_entries[0]
        print("\n[추출 예시]")
        print(f"  - 파일 경로: {sample_entry['path']}")
        print(f"  - 세그먼트 수: {len(sample_entry['soft_labels'])}")
        print(f"  - 첫 번째 세그먼트의 Soft Label: {sample_entry['soft_labels'][0]}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract soft labels from a trained Teacher model.")
    parser.add_argument('--mark_version', type=str, default="mark2.9.0", help="The model version to use (e.g., 'mark2.9.0').")
    args = parser.parse_args()
    
    exit(extract_soft_labels(mark_version=args.mark_version))
    
    
    