import torch
import numpy as np
import argparse
import torchaudio
from tqdm import tqdm

# 학습에 사용된 모델 클래스를 임포트합니다.
from diffwave.model_KAFE_SResnet import SResnet1D

from diffwave.params import params as diffparams

def generate_waveform(model, params, device, audio_length):
    """
  확산 모델의 역방향 프로세스를 실행하여 오디오 파형을 생성합니다.
  """
    # 1. 확산 스케줄(beta, alpha 등)을 params로부터 계산합니다.
    beta = np.array(params.noise_schedule)
    alpha = 1.0 - beta
    alpha_cumprod = np.cumprod(alpha)

    # 텐서로 변환하여 GPU로 이동
    beta = torch.tensor(beta.astype(np.float32)).to(device)
    alpha = torch.tensor(alpha.astype(np.float32)).to(device)
    alpha_cumprod = torch.tensor(alpha_cumprod.astype(np.float32)).to(device)

    # 2. 순수한 가우시안 노이즈에서 샘플링을 시작합니다. (x_T)
    audio = torch.randn(1, audio_length, device=device)
    print(f"오디오 길이 {audio_length}의 노이즈에서 생성을 시작합니다...")

    # 3. 타임스텝 T에서 0까지 역방향으로 루프를 실행합니다.
    for t in tqdm(range(len(params.noise_schedule) - 1, -1, -1), desc="Generating waveform"):
        # 현재 타임스텝 `t`를 텐서로 만듭니다. (모델 입력용)
        current_t = torch.full((1,), t, device=device, dtype=torch.long)

        # 현재 스텝의 alpha 값들을 가져옵니다.
        alpha_t = alpha[t]
        alpha_cumprod_t = alpha_cumprod[t]

        # 4. 모델을 사용해 현재 오디오(x_t)에 포함된 노이즈(ε)를 예측합니다.
        predicted_noise = model(audio, current_t)

        # 5. DDPM 샘플링 공식을 사용하여 x_{t-1}을 계산합니다.
        # 계수 계산
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        # 평균(mean) 계산
        audio_mean = coef1 * (audio - coef2 * predicted_noise)

        # 6. 마지막 스텝(t=0)이 아니면, 새로운 노이즈(z)를 추가합니다.
        if t > 0:
            noise = torch.randn_like(audio)
            sigma_t = torch.sqrt(beta[t])  # 분산(variance)의 제곱근
            audio = audio_mean + sigma_t * noise
        else:
            audio = audio_mean

    print("생성이 완료되었습니다.")
    return audio


def main(args):
    # 1. 체크포인트 로드
    print(f"체크포인트 로드 중: {args.model_checkpoint}")
    checkpoint = torch.load(args.model_checkpoint)
    params = checkpoint['params']

    # 2. 모델 초기화 및 학습된 가중치 로드
    # 체크포인트 저장 시 사용된 모델과 동일한 아키텍처를 사용해야 합니다.
    model = SResnet1D(leak_mem=0.99).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 추론 모드로 설정
    print("모델 로드 완료.")

    # 3. 파형 생성
    with torch.no_grad():  # 그래디언트 계산 비활성화
      for idx in range (0, 16):
        generated_audio = generate_waveform(model, diffparams, 'cuda', args.audio_length)

        # 최종 오디오의 범위를 [-1, 1]로 클리핑 또는 tanh 적용
        generated_audio = torch.clamp(generated_audio.squeeze(), -1.0, 1.0)

        # 4. 생성된 오디오를 .wav 파일로 저장
        np.save(args.output_path + 'output_' + str(idx).zfill(2) + '.npy', generated_audio.cpu())
        print(f"생성된 '{str(idx)}'번째 파형이 '{args.output_path}'에 저장되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate audio waveform from a trained DiffWave model.')
    parser.add_argument('model_checkpoint', type=str, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('output_path', type=str, help='Path to save the generated .wav file.')
    parser.add_argument('--audio_length', type=int, default=48000, help='Length of the audio to generate.')
    args = parser.parse_args()

    main(args)