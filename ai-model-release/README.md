# ICCAS AI 모델 배포 패키지

GPS, IMU 데이터를 사용해 LSTM 기반으로 실시간 배회감지와 낙상감지를 수행하는 AI 모델 패키지입니다. 팀원이 Git에서 받을 때 원본 엑셀 데이터나 실행 결과 파일 없이, AI 코드와 학습 완료 모델만 사용할 수 있도록 구성했습니다.

## Git에 올릴 파일

이 폴더에서 아래 파일만 Git에 올리면 됩니다.

```text
ai-model-release/
  README.md
  MODEL_CARD.md
  requirements.txt
  .gitignore
  scripts/realtime_sensor_lstm.py
  models/iccas_sensor_lstm_fall.pt
  models/iccas_sensor_lstm_fall.json
```

Git에 올리지 않는 파일은 원본 학습 데이터, 테스트 결과, 캐시 파일입니다.

```text
ICCAS_total_data.xlsx
ICCAS_total_data_with_fall.xlsx
data/
outputs/
__pycache__/
._*
.DS_Store
```

## 맥 개발 환경 준비

```bash
cd ai-model-release
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

가상환경에서 나가려면 아래 명령어를 사용합니다.

```bash
deactivate
```

## 엑셀/CSV 데이터 형식

학습 또는 시뮬레이션 입력 파일은 `.xlsx` 또는 `.csv`를 사용할 수 있습니다. 최소 필요 컬럼은 아래와 같습니다.

```text
server_time, device, roll, pitch, yaw, ax, ay, az, wx, wy, wz
```

GPS 기반 배회감지와 지도 표시까지 하려면 아래 컬럼도 같이 있어야 합니다.

```text
lat, lng
```

`latitude`, `longitude`, `timestamp` 컬럼명은 코드에서 각각 `lat`, `lng`, `server_time`으로 자동 변환합니다. `label` 컬럼은 있으면 학습/검증 참고용으로 사용하고, 없어도 실행은 가능합니다.

## 데이터 확인

원본 데이터가 준비되어 있을 때 구조를 먼저 확인합니다. 원본 데이터는 Git에 올리지 말고 팀 내부 공유 드라이브나 로컬 경로에 둡니다.

```bash
python scripts/realtime_sensor_lstm.py inspect \
  --source ../ICCAS_total_data_with_fall.xlsx \
  --sheet data
```

## 이미 학습된 모델로 시뮬레이션

백엔드 없이 로컬에서 추론 결과 CSV만 만들려면 아래처럼 실행합니다.

```bash
python scripts/realtime_sensor_lstm.py replay \
  --source ../ICCAS_total_data_with_fall.xlsx \
  --sheet data \
  --model models/iccas_sensor_lstm_fall.pt \
  --output outputs/replay_results.csv \
  --device cpu \
  --print-events
```

Mac에서 Apple Silicon GPU를 쓰고 싶으면 `--device mps`를 사용할 수 있습니다. 호환 문제가 있으면 `--device cpu`가 가장 안정적입니다.

## 백엔드로 결과 전송

백엔드 서버가 먼저 실행되어 있어야 합니다.

```bash
cd ../backend-server/backend
PYTHONPATH=. python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

다른 터미널에서 AI 추론을 실행하면 각 센서 포인트의 추론 결과가 백엔드의 `/api/sensor-result`로 전송됩니다.

```bash
cd ../ai-model-release
source .venv/bin/activate
python scripts/realtime_sensor_lstm.py replay \
  --source ../ICCAS_total_data_with_fall.xlsx \
  --sheet data \
  --model models/iccas_sensor_lstm_fall.pt \
  --backend-url http://127.0.0.1:8000/api/sensor-result \
  --output outputs/backend_replay_results.csv \
  --sleep 0.05 \
  --device cpu \
  --print-events
```

백엔드에서 확인하는 주소는 아래와 같습니다.

```text
http://127.0.0.1:8000/api/sensor-result
http://127.0.0.1:8000/api/sensor-result/latest
http://127.0.0.1:8000/api/sensor-result/history?limit=5000
```

`/api/sensor-result`와 `/latest`는 최신 1건만 보여줍니다. 낙상 이벤트가 지나간 뒤 최신 데이터가 정상이라면 `fall_detected=false`로 보일 수 있습니다. 전체 이벤트 확인은 `/history?limit=5000`에서 봅니다.

## 실시간 JSON 입력 추론

실제 장비나 다른 프로세스에서 센서 JSON을 한 줄씩 넘길 때는 `live` 모드를 사용합니다.

```bash
python scripts/realtime_sensor_lstm.py live \
  --model models/iccas_sensor_lstm_fall.pt \
  --backend-url http://127.0.0.1:8000/api/sensor-result \
  --device cpu
```

입력 JSON 예시는 아래와 같습니다.

```json
{"server_time":"2026-07-01 11:35:56.764000","device":"esp32-1","lat":36.621853,"lng":127.426337,"roll":-12.1,"pitch":9.7,"yaw":8.8,"ax":3.2,"ay":-2.6,"az":4.1,"wx":320.0,"wy":-280.0,"wz":220.0}
```

## 재학습 방법

새로운 학습 데이터가 생기면 원본 데이터는 Git에 올리지 않고 로컬 경로에서만 사용합니다. 아래 명령어는 기존 모델 파일을 새로 학습한 결과로 덮어씁니다.

```bash
python scripts/realtime_sensor_lstm.py train \
  --source ../ICCAS_total_data_with_fall.xlsx \
  --sheet data \
  --model models/iccas_sensor_lstm_fall.pt \
  --epochs 80 \
  --batch-size 32 \
  --sequence-length 8 \
  --threshold-quantile 0.98 \
  --route-threshold-m 30.0 \
  --device cpu
```

학습이 끝나면 아래 두 파일이 갱신됩니다.

```text
models/iccas_sensor_lstm_fall.pt
models/iccas_sensor_lstm_fall.json
```

그 다음 이 두 파일과 필요한 코드 변경만 Git에 커밋합니다.

```bash
git add README.md MODEL_CARD.md requirements.txt scripts/realtime_sensor_lstm.py models/iccas_sensor_lstm_fall.pt models/iccas_sensor_lstm_fall.json
git commit -m "Add ICCAS LSTM fall and wandering model"
git push
```

## 프론트 화면 흐름

AI 스크립트가 백엔드로 결과를 보내면 프론트는 백엔드의 `/api/sensor-result/latest`와 `/api/sensor-result/history?limit=5000`을 주기적으로 조회합니다. 지도 화면에서는 `lat`, `lng`, `fall_detected`, `wandering_detected`, `risk_level`, `detection_type` 값을 사용해 실시간 위치와 알림을 표시합니다.
