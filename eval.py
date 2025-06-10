# eval.py  


import os
import sys
import csv
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    precision_recall_fscore_support, accuracy_score, roc_auc_score,
    roc_curve, auc
)
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder
from vild_head import ViLDHead
from vild_parser_student import AudioParser
from seed_utils import set_seed


def evaluate(audio_label_list, seed_value=42, mark_version="mark2.9.0"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    parser = AudioParser(config)
    device = config.device

    class_names = config.classes
    num_classes = len(class_names)
    idx_to_label = {i: label for i, label in enumerate(class_names)}
    label_to_idx = {label: i for i, label in enumerate(class_names)}

    encoder = SimpleAudioEncoder(config).to(device)
    head = ViLDHead(config.embedding_dim, num_classes).to(device)

    # --- 모델 경로 수정: 지식 증류로 학습된 모델을 로드 ---
    encoder_path = f"distilled_student_encoder_{mark_version}.pth"
    head_path = f"distilled_student_head_{mark_version}.pth"

    if not os.path.exists(encoder_path) or not os.path.exists(head_path):
        print(f"[ERROR] 모델 파일 없음: {encoder_path} / {head_path}. 일반 Student 모델로 재시도합니다.")
        encoder_path = f"best_student_encoder_{mark_version}.pth"
        head_path = f"best_student_head_{mark_version}.pth"
        if not os.path.exists(encoder_path) or not os.path.exists(head_path):
            print(f"[ERROR] 어떤 모델 파일도 찾을 수 없습니다: {encoder_path} / {head_path}")
            return
    # --- 경로 수정 끝 ---

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    head.load_state_dict(torch.load(head_path, map_location=device))
    encoder.eval()
    head.eval()

    y_true, y_pred, y_prob, paths = [], [], [], []

    for path, true_label in audio_label_list:
        if true_label not in label_to_idx:
            continue
        true_idx = label_to_idx[true_label]
        segments = parser.load_and_segment(path)
        if not segments:
            # 어떤 파일이 평가에서 제외되었는지 출력  
            print(f"[INFO] Skipping file due to no valid segments: {os.path.basename(path)}")
            continue

        total_score = torch.zeros(num_classes, device=device)
        valid_segments = 0

        with torch.no_grad():
            for seg in segments:
                if seg is None or seg.ndim not in (3, 4):
                    continue
                if seg.ndim == 3:
                    seg = seg.unsqueeze(0)

                seg = seg.to(device)
                feat = encoder(seg)
                logits = head(feat.flatten(start_dim=1))
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                total_score += probs
                valid_segments += 1

        if valid_segments == 0:
            continue

        avg_score = total_score / valid_segments
        pred_idx = torch.argmax(avg_score).item()

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_prob.append(avg_score.cpu().numpy()) # ROC AUC를 위한 확률값 저장
        paths.append(path)

    if not y_true:
        print("[WARN] 평가 가능한 예측 없음.")
        return
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    plot_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Confusion Matrix (기존과 동일)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({mark_version})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{mark_version}.png"))
    plt.close()
    print("[INFO] Confusion matrix 저장 완료.")

    # 2. 성능 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    
    # ROC AUC Score (Multi-class)
    # OvR(One-vs-Rest) 방식으로 각 클래스에 대한 ROC AUC 계산
    if num_classes == 2:
        # 이진 분류의 경우 positive class에 대한 확률만 사용
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        # 다중 클래스의 경우 macro 평균 사용
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    
    print("\n" + "="*30)
    print(f"      성능 평가 결과 ({mark_version})")
    print("="*30)
    print(f"  - 전체 정확도 (Accuracy): {accuracy:.4f}")
    if isinstance(roc_auc, float):
        print(f"  - ROC AUC Score: {roc_auc:.4f}")
    print("\n클래스별 성능:")
    for i in range(num_classes):
        print(f"  - 클래스: {class_names[i]}")
        print(f"    - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1[i]:.4f}")
    print("="*30 + "\n")


    # 3. 성능 지표 히트맵 스타일 표로 시각화 (신규 기능)
    metrics_data = {
        'Precision': list(precision) + [None],
        'Recall': list(recall) + [None],
        'F1-Score': list(f1) + [None]
    }
    df_metrics = pd.DataFrame(metrics_data, index=class_names + ['Overall'])
    df_metrics.loc['Overall', 'Accuracy'] = accuracy
    df_metrics.loc['Overall', 'ROC AUC'] = roc_auc if isinstance(roc_auc, float) else None
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_metrics, annot=True, fmt=".4f", cmap="viridis", cbar=False, linewidths=.5)
    plt.title(f'Performance Metrics ({mark_version})', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'performance_metrics_table_{mark_version}.png'))
    plt.close()
    print("[INFO] 성능 지표 테이블 이미지 저장 완료.")

    # 4. ROC Curve 그리기 (신규 기능)
    plt.figure(figsize=(7, 6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    else: # 다중 클래스
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            roc_auc_class = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve of class {class_names[i]} (area = {roc_auc_class:.4f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {mark_version}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'roc_curve_{mark_version}.png'))
    plt.close()
    print("[INFO] ROC 커브 이미지 저장 완료.")


    # 5. 성능 지표 CSV 저장 (신규 기능)
    metrics_summary = {
        'Metric': ['Accuracy', 'ROC AUC (Macro)' if num_classes > 2 else 'ROC AUC'],
        'Score': [accuracy, roc_auc if isinstance(roc_auc, float) else 'N/A']
    }
    df_summary = pd.DataFrame(metrics_summary)
    
    df_classwise = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    csv_path = os.path.join(plot_dir, f'performance_summary_{mark_version}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write(f"# Performance Summary for {mark_version}\n\n")
        df_summary.to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        df_classwise.to_csv(f, index=False)
    print(f"[INFO] 성능 요약 CSV 저장 완료: {csv_path}")

    # 예측 결과 CSV 저장 (기존 기능 유지)
    pred_results_path = os.path.join(plot_dir, f'prediction_details_{mark_version}.csv')
    with open(pred_results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Filename', 'True Label', 'Predicted Label'] + [f'Prob_{name}' for name in class_names]
        writer.writerow(header)
        for i in range(len(paths)):
            row = [os.path.basename(paths[i]), idx_to_label[y_true[i]], idx_to_label[y_pred[i]]] + list(y_prob[i])
            writer.writerow(row)
    print(f"[INFO] 상세 예측 결과 CSV 저장 완료: {pred_results_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, default="mark2.9.0")
    args = parser.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    csv_path = os.path.join(BASE_DIR, f"dataset_index_{args.mark_version}.csv")

    # 유효성 검사를 위한 파서 인스턴스를 미리 생성
    pre_parser = AudioParser(config)

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = [(row['path'], row['label']) for row in reader if row['label'] in config.classes]

        # 1. 기존과 동일하게 데이터 샘플링
        sampled_files = []
        per_class_max = 30
        class_counter = defaultdict(int)
        for path, label in data:
            if class_counter[label] < per_class_max:
                sampled_files.append((path, label))
                class_counter[label] += 1
        
        print(f"[INFO] 총 {len(sampled_files)}개의 파일을 샘플링했습니다. 이제 유효성을 검사합니다...")

        # 2. 샘플링된 파일 목록의 유효성을 미리 검사
        valid_samples_for_eval = []
        for path, label in sampled_files:
            # 미리 파일을 로드하고 세그먼트 생성이 가능한지 확인
            segments = pre_parser.load_and_segment(path)
            if segments: # 세그먼트가 하나라도 생성되면 유효한 파일로 간주
                valid_samples_for_eval.append((path, label))
            else:
                # 어떤 파일이 손상되었거나 너무 짧아서 제외되는지 출력
                print(f"[WARN] 유효하지 않은 파일입니다. 평가에서 제외합니다: {os.path.basename(path)}")

        print(f"[INFO] 유효성 검사 완료. 총 {len(valid_samples_for_eval)}개의 유효한 파일로 평가를 시작합니다.")

        # 3. 유효성이 확인된 파일 목록만 evaluate 함수에 전달
        if not valid_samples_for_eval:
            print("[ERROR] 평가할 유효한 샘플이 없습니다.")
        else:
            evaluate(valid_samples_for_eval, seed_value=42, mark_version=args.mark_version)

    except Exception as e:
        print(f"[ERROR] 평가 중 예외 발생: {e}", file=sys.stderr)