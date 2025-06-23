import torch
import torch.nn as nn
from diffwave.model_SEW_SResnet import sew_resnet18_1d
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

# 이 스크립트는 learner.py의 복잡한 로직 없이
# 모델의 순전파 -> 손실 계산 -> 역전파 과정이 독립적으로 잘 되는지만 확인합니다.

def run_debug():
    print("디버깅 스크립트를 시작합니다...")

    # 1. 모델 초기화
    # connect_f='ADD'로 설정하여 가장 간단한 덧셈 잔차 연결을 사용
    try:
        model = sew_resnet18_1d(connect_f='ADD').cuda()
        model.train()
        print("모델 초기화 성공.")
    except Exception as e:
        print(f"모델 초기화 중 에러 발생: {e}")
        return

    # 2. 가상의 더미 데이터 생성
    batch_size = 4
    audio_length = 16384  # learner.py의 audio_segment_length와 유사한 값

    # [B, L] 형태의 입력 데이터
    dummy_input = torch.randn(batch_size, audio_length).cuda()
    # [B, L] 형태의 정답 데이터 (노이즈)
    dummy_target = torch.randn(batch_size, audio_length).cuda()
    # [B] 형태의 가상 timestep
    dummy_t = torch.randint(0, 100, [batch_size]).cuda()

    print(f"더미 데이터 생성 완료. 입력 형태: {dummy_input.shape}")

    # 3. 옵티마이저 및 손실 함수 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    print("옵티마이저 및 손실 함수 정의 완료.")

    try:
        # 4. 순전파 (Forward pass)
        print("순전파를 시작합니다...")
        predicted = model(dummy_input, dummy_t)
        print(f"순전파 완료. 예측 결과 형태: {predicted.shape}")

        # 출력 형태를 [B, L]로 맞춤
        predicted = predicted.squeeze(1)

        # 5. 손실 계산
        loss = loss_fn(predicted, dummy_target)
        print(f"손실 계산 완료. Loss: {loss.item()}")

        # 6. 역전파 (Backward pass)
        print("역전파를 시작합니다...")
        optimizer.zero_grad()
        loss.backward()
        print("역전파 완료.")

        # 7. 옵티마이저 스텝
        optimizer.step()
        print("옵티마이저 스텝 완료.")

        print("\n*** 디버깅 성공: 모델의 순전파 및 역전파가 독립적으로 실행되었습니다. ***")
        print("이것은 모델 아키텍처 자체에는 문제가 없으며, learner.py의 학습 루프 내 다른 요소가 문제의 원인일 가능성이 매우 높다는 것을 의미합니다.")

    except Exception as e:
        print(f"\n*** 디버깅 실패: 최소 실행 환경에서 에러가 발생했습니다! ***")
        print("에러 타입:", type(e).__name__)
        print("에러 메시지:", e)
        print("\n이 경우, torch와 spikingjelly 라이브러리 버전 간의 충돌 등 환경적인 문제일 수 있습니다.")


if __name__ == '__main__':
    run_debug()