# student_train_distillation.py  

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 학습률 스케줄러를 위해 추가
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import argparse

import functools
print = functools.partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder
from vild_head import ViLDHead
from vild_parser_student import AudioParser
from seed_utils import set_seed


# 얼리 스타핑(Early Stopping)을 위한 클래스 정의
class EarlyStopping:
    """검증 손실이 개선되지 않으면 학습을 조기 종료합니다."""
    def __init__(self, patience=5, verbose=False, delta=0, path_encoder='encoder.pth', path_head='head.pth'):
        """
        Args:
            patience (int): 검증 손실이 개선되지 않아도 참을 에폭 수.
            verbose (bool): 메시지 출력 여부.
            delta (float): 개선으로 간주하기 위한 최소 변화량.
            path_encoder (str): 최고의 인코더 모델을 저장할 경로.
            path_head (str): 최고의 헤드 모델을 저장할 경로.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path_encoder = path_encoder
        self.path_head = path_head

    def __call__(self, val_loss, encoder, head):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, encoder, head)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[EarlyStopping] counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, encoder, head)
            self.counter = 0

    def save_checkpoint(self, val_loss, encoder, head):
        """모델의 가중치를 저장합니다."""
        if self.verbose:
            print(f'[EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(encoder.state_dict(), self.path_encoder)
        torch.save(head.state_dict(), self.path_head)
        self.val_loss_min = val_loss
        

# STEP 1: 지식 증류를 위한 새로운 Loss 함수 정의 (기존 코드와 동일)
class DistillationLoss(nn.Module):
    def __init__(self, T, alpha, ignore_index=-1):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.hard_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.soft_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, soft_labels, hard_labels):
        valid_indices = hard_labels != -1
        if not valid_indices.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        loss_hard = self.hard_loss_fn(student_logits[valid_indices], hard_labels[valid_indices])
        soft_student_logits = F.log_softmax(student_logits[valid_indices] / self.T, dim=1)
        soft_teacher_logits = F.softmax(soft_labels[valid_indices] / self.T, dim=1)
        loss_soft = self.soft_loss_fn(soft_student_logits, soft_teacher_logits) * (self.T ** 2)
        
        total_loss = self.alpha * loss_soft + (1 - self.alpha) * loss_hard
        return total_loss


# STEP 2: Hard/Soft Label을 모두 로드하는 함수 (기존 코드와 동일)
def load_labels(config, mark_version):
    hard_label_path = os.path.join(BASE_DIR, f"hard_labels_{mark_version}.pkl")
    soft_label_path = os.path.join(BASE_DIR, f"soft_labels_{mark_version}.pkl")

    with open(hard_label_path, "rb") as f:
        hard_data = pickle.load(f)
    with open(soft_label_path, "rb") as f:
        soft_data = pickle.load(f)
        
    soft_map = {entry['path']: entry['soft_labels'] for entry in soft_data}
    samples = []
    for hard_entry in hard_data:
        path = hard_entry["path"]
        if path in soft_map:
            hard_labels = torch.tensor(hard_entry["hard_labels"], dtype=torch.long)
            soft_labels = torch.tensor(soft_map[path], dtype=torch.float)
            
            if len(hard_labels) != len(soft_labels):
                print(f"[Warning] 레이블 수 불일치: {path}. Hard: {len(hard_labels)}, Soft: {len(soft_labels)}. 건너뜁니다.")
                continue
            
            samples.append((path, hard_labels, soft_labels))
    
    return samples


# STEP 3: Soft Label 처리를 포함하도록 collate_fn 수정 (기존 코드와 동일)
def collate_fn_distillation(batch):
    mel_list, hard_label_list, soft_label_list = [], [], []

    for path, hard_labels, soft_labels in batch:
        segments = parser.load_and_segment(path)
        if not segments: continue
        
        num_segs = min(len(segments), len(hard_labels))
        if num_segs == 0: continue
        
        mel_tensor = torch.stack(segments[:num_segs])
        mel_list.append(mel_tensor)
        hard_label_list.append(hard_labels[:num_segs])
        soft_label_list.append(soft_labels[:num_segs])
        
    if not mel_list:
        return torch.empty(0), torch.empty(0), torch.empty(0)
        
    max_k = max(mel.shape[0] for mel in mel_list)
    num_classes = soft_label_list[0].shape[1] if soft_label_list else 0
        
    for i in range(len(mel_list)):
        cur_k = mel_list[i].shape[0]
        if cur_k < max_k:
            pad_mel = torch.zeros((max_k - cur_k, 1, 64, 101))
            mel_list[i] = torch.cat([mel_list[i], pad_mel], dim=0)
            
            pad_hard = torch.full((max_k - cur_k,), -1, dtype=torch.long)
            hard_label_list[i] = torch.cat([hard_label_list[i], pad_hard], dim=0)
            
            pad_soft = torch.zeros((max_k - cur_k, num_classes), dtype=torch.float)
            soft_label_list[i] = torch.cat([soft_label_list[i], pad_soft], dim=0)
        
    return torch.stack(mel_list), torch.stack(hard_label_list), torch.stack(soft_label_list)


# STEP 4: 메인 학습 함수 수정
# 함수 전체적으로 수정
def train_student_with_distillation(seed_value=42, mark_version="mark2.9.0"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    global parser
    parser = AudioParser(config)
    device = torch.device(config.device)

    try:
        samples = load_labels(config, mark_version)
    except FileNotFoundError:
        print(f"[ERROR] '{mark_version}'에 대한 hard 또는 soft label 파일이 없습니다. extract 스크립트를 먼저 실행하세요.")
        return

    paths, hard_labels, soft_labels = zip(*samples)
    train_indices, val_indices = train_test_split(range(len(paths)), test_size=0.1, random_state=seed_value)

    train_data = [(paths[i], hard_labels[i], soft_labels[i]) for i in train_indices]
    val_data = [(paths[i], hard_labels[i], soft_labels[i]) for i in val_indices]
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_distillation)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_distillation)

    encoder = SimpleAudioEncoder(config).to(device)
    head = ViLDHead(config.embedding_dim, len(config.classes)).to(device)
    student_model = nn.Sequential(encoder, nn.Flatten(start_dim=1), head).to(device)

    T = 4.0      # Temperature
    alpha = 0.7  # Soft loss 가중치
    criterion = DistillationLoss(T=T, alpha=alpha, ignore_index=-1)
    optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)

    # 학습률 스케줄러와 얼리 스타핑 설정
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopping_patience = 10 # 10 에폭 동안 개선 없으면 중지
    
    # 모델 저장 경로 설정
    encoder_path = f"distilled_student_encoder_{config.mark_version}.pth"
    head_path = f"distilled_student_head_{config.mark_version}.pth"
    early_stopper = EarlyStopping(patience=early_stopping_patience, verbose=True, path_encoder=encoder_path, path_head=head_path)

    train_loss_history, val_loss_history = [], []

    print(f"[INFO] Student training with Distillation for {mark_version} on {device}")
    print(f"[INFO] Hyperparameters: T={T}, alpha={alpha}, LR={config.learning_rate}")
    print(f"[INFO] Early Stopping Patience: {early_stopping_patience}, LR Scheduler Patience: 3")
    
    for epoch in range(config.num_epochs):
        student_model.train()
        total_loss = 0.0

        for mel_batch, hard_label_batch, soft_label_batch in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
            if mel_batch.numel() == 0: continue

            B, K, C, H, W = mel_batch.shape
            mel_batch_flat = mel_batch.view(B * K, C, H, W).to(device)
            hard_label_flat = hard_label_batch.view(B * K).to(device)
            soft_label_flat = soft_label_batch.view(B * K, -1).to(device)

            student_logits = student_model(mel_batch_flat)
            loss = criterion(student_logits, soft_label_flat, hard_label_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel_batch, hard_label_batch, soft_label_batch in val_loader:
                if mel_batch.numel() == 0: continue

                B, K, C, H, W = mel_batch.shape
                mel_batch_flat = mel_batch.view(B * K, C, H, W).to(device)
                hard_label_flat = hard_label_batch.view(B * K).to(device)
                soft_label_flat = soft_label_batch.view(B * K, -1).to(device)

                student_logits = student_model(mel_batch_flat)
                loss = criterion(student_logits, soft_label_flat, hard_label_flat)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 기존 모델 저장 로직을 EarlyStopping 클래스로 대체
        # 얼리 스토핑 체크포인트. 이 클래스 내부에서 최고의 모델을 저장.
        early_stopper(avg_val_loss, encoder, head)
        if early_stopper.early_stop:
            print("[INFO] Early stopping triggered.")
            break

        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)

    # Loss 그래프 저장
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6)) # 그래프 크기 조정
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title(f'Distilled Student Loss Curve ({mark_version})')
    plt.xlabel('Epoch') # x축 라벨 추가
    plt.ylabel('Loss')  # y축 라벨 추가
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/loss_curve_distilled_student_{mark_version}.png")
    print(f"\n[INFO] Distilled student loss curve saved to plots/loss_curve_distilled_student_{mark_version}.png")
    print(f"[INFO] Best model saved to {encoder_path} and {head_path} with validation loss: {early_stopper.val_loss_min:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Student model with Knowledge Distillation.")
    parser.add_argument('--mark_version', type=str, default="mark2.9.0", help="The model version to use.")
    args = parser.parse_args()
    train_student_with_distillation(mark_version=args.mark_version)  