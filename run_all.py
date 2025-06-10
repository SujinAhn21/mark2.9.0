# run_all.py (지식 증류 파이프라인 버전)

import os
import subprocess
import logging
from datetime import datetime
import argparse

# ===== 파라미터 설정 =====
parser = argparse.ArgumentParser(description="소음 분류 전체 학습 파이프라인 (지식 증류 포함)")
parser.add_argument("--mark_version", type=str, default="mark2.9.0", help="실행할 모델 버전")
args = parser.parse_args()
mark_version = args.mark_version

# ===== 경로 및 로그 설정 =====
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "logFiles")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"run_pipeline_distillation_{mark_version}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ===== 데코레이터 =====
def timed_step(func):
    def wrapper(*args, **kwargs):
        # 함수 이름에서 단계 이름과 설명을 추출
        step_name = func.__name__.replace("run_", "").replace("_", " ").title()
        logging.info(f"\n[실행 시작] --> {step_name}")
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = (end - start).total_seconds()
        logging.info(f"[완료] --> {step_name} (소요시간: {duration:.2f}초)")
        return result
    return wrapper

# ===== 서브프로세스 실행 함수 =====
def run_subprocess(command_list):
    try:
        logging.info(f"[CMD] {' '.join(command_list)}")
        # shell=True는 보안 위험이 있으므로 사용하지 않고, command_list를 직접 전달합니다.
        result = subprocess.run(
            command_list,
            capture_output=True, # stdout, stderr를 캡처
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.stdout:
            logging.info("[STDOUT]\n" + result.stdout)
        if result.stderr:
            # 에러가 아니더라도 경고 메시지 등이 stderr로 나올 수 있으므로 info 레벨로 로깅
            logging.info("[STDERR]\n" + result.stderr)

        return result.returncode

    except Exception as e:
        logging.error(f"[ERROR] Subprocess 실행 중 예외 발생: {e}")
        return 1

# ===== 단계별 실행 함수 정의 (지식 증류 파이프라인) =====
@timed_step
def run_step0_preprocess_audio():
    """오디오 파일을 고정된 길이로 전처리합니다."""
    return run_subprocess(["python", "fix_audio_length_to_240000.py", "--mark_version", mark_version])

@timed_step
def run_step1_generate_dataset_index():
    """데이터셋 인덱스 CSV 파일을 생성합니다."""
    return run_subprocess(["python", "generate_dataset_index.py", "--mark_version", mark_version])

@timed_step
def run_step2_teacher_model_train():
    """Teacher 모델을 학습시킵니다."""
    return run_subprocess(["python", "teacher_train.py", "--mark_version", mark_version])

@timed_step
def run_step3_extract_hard_labels():
    """학습 데이터로부터 Hard Label을 추출합니다."""
    return run_subprocess(["python", "extract_hard_labels.py", "--mark_version", mark_version])

@timed_step
def run_step4_extract_soft_labels():
    """학습된 Teacher 모델로부터 Soft Label을 추출합니다. (지식 증류 핵심)"""
    return run_subprocess(["python", "extract_soft_labels.py", "--mark_version", mark_version])

@timed_step
def run_step5_student_distillation_train():
    """Hard Label과 Soft Label을 함께 사용하여 Student 모델을 학습시킵니다."""
    return run_subprocess(["python", "student_train_distillation.py", "--mark_version", mark_version])

@timed_step
def run_step6_evaluate_model():
    """학습된 Student 모델의 성능을 평가합니다."""
    return run_subprocess(["python", "eval.py", "--mark_version", mark_version])

@timed_step
def run_step7_plot_results():
    """결과를 시각화합니다."""
    return run_subprocess(["python", "plot_audio.py", "--mark_version", mark_version])


# ===== 메인 실행 =====
if __name__ == "__main__":
    logging.info("="*50)
    logging.info("  소음 분류 전체 학습 파이프라인 (지식 증류 Ver.) 시작  ")
    logging.info("="*50)
    logging.info(f"모델 버전: {mark_version}")
    logging.info("현재 모델은 Teacher의 Soft Label과 실제 Hard Label을 함께 사용하는\n"
                 "지식 증류(Knowledge Distillation) 방식으로 학습됩니다.")

    steps = [
        run_step0_preprocess_audio,
        run_step1_generate_dataset_index,
        run_step2_teacher_model_train,
        run_step3_extract_hard_labels,
        run_step4_extract_soft_labels,       # 새로 추가된 단계
        run_step5_student_distillation_train,
        run_step6_evaluate_model,
        run_step7_plot_results
    ]

    for step in steps:
        return_code = step()
        if return_code != 0:
            logging.error(f"\n[CRITICAL ERROR] 파이프라인 실패: '{step.__name__}' 단계에서 오류 발생 (종료 코드: {return_code}).")
            logging.error("이후 단계를 생략하고 파이프라인을 중단합니다.")
            break
    
    logging.info("="*50)
    logging.info(f"[종료] 전체 파이프라인 완료. 로그 파일: {log_file_path}")
    logging.info("="*50)  