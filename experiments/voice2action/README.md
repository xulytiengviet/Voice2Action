# Voice2Action — mô phỏng thực nghiệm (13/11/2026)

Trạng thái: nghiên cứu/robot ảo, không phải chương trình điều khiển motor. Repository này vốn chứa XPolicyLab; toàn bộ thay đổi mới nằm trong thư mục độc lập, không thay đổi các robot policy hiện có.

## Pipeline và phân biệt hai mô phỏng

- Python CNN, experiments/voice2action/v2a.py: WAV PCM16 → STFT(512, hop256) → 32-bin log-mel → CNN 2 tầng → action ID → lexicon → Safety Gate → robot ảo.
- Browser, docs/voice2action/index.html: microphone → STFT/log-mel → prototype cosine similarity → action ID → lexicon → Safety Gate → robot ảo. **Browser không dùng CNN.**

Không chạy ASR hay LLM trong đường suy luận chính. Nhãn tiếng Việt được dùng cho huấn luyện và kiểm toán, không phải văn bản được nhận dạng từ âm thanh.

### Giới hạn CVNSS4.0

lexicon.json giữ 4 lệnh được duyệt: stop, forward, left, right. Cột CVNSS4.0 đang null vì chưa nhập codec kèm version-pinned golden vectors; tuyệt đối không được bịa mã CVNSS4.0. Action-ID lookup hạn chế lệnh ngoài danh sách nhưng KHÔNG thể chứng minh đã nghe đúng câu nói. 7 mẫu rẽ phải bị dự đoán thành rẽ trái vẫn qua Gate vì cả hai lệnh đều hợp lệ. [VF1 của vifirmware](https://github.com/xulytiengviet/vifirmware/blob/main/spec/VF1.md) không chứng minh CVNSS4.0 có phép giải mã đơn trị.

## Tái lập (Python 3.10+, eSpeak với voices vi, vi-hue, vi-sgn)

~~~sh
cd experiments/voice2action
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
python v2a.py generate --variants 12
python v2a.py train --epochs 25
python v2a.py eval
python v2a.py infer data/wavs/vi-sgn_left_0.wav
~~~

Trên Windows, thay dòng activate bằng .venv\Scripts\activate. Cần cài eSpeak trước khi generate.

Generator dùng seed 20261113 tạo 144 mẫu giọng eSpeak; 72 train + 24 val từ giọng vi và vi-hue, 48 test từ profile vi-sgn chưa xuất hiện trong train/val. Đây là giọng TTS chứ không phải phát ngôn người thật hay thí nghiệm trên các vùng phương ngữ.

### Kết quả đo tham khảo (máy CPU của phiên thử nghiệm)

Validation 24/24. Held-out test 41/48 = 85,42%. Nhầm lẫn 7/12 lệnh right thành left. 48/48 lệnh vẫn được chấp thuận bởi Gate, nên chấp thuận không bằng đúng nghĩa. p50 tính toán feature+CNN+Gate khoảng 1,85 ms; không bao gồm thu âm, VAD, mạng, ROS2 hay thời gian motor. Chạy lại và xem output/report.json để lấy số đo mới; không cam kết tái lập đúng milliseconds trên thiết bị khác.

### An toàn và nghiên cứu tiếp

Gate độc lập với CNN: giới hạn action, confidence, margin, e-stop, biên ±1m và vùng cấm (0.6,0.5,r=0.2). Không có giao tiếp tới thiết bị thật. Bước tiếp theo: thu giọng người thật, chia theo speaker, kiểm thử nhiễu/giọng chưa gặp/phủ định/OOD, đo macro-F1/false accept/p50-p95 từ lời nói tới hành động, baseline ASR→Action và kiểm tra safety độc lập. Không đủ bằng chứng cho tuyên bố chống ảo giác hoàn toàn hay tốc độ 3×.

### Nguồn nghiên cứu

- [AST](https://github.com/YuanGongND/ast) — spectrogram transformer.
- [SepTr](https://github.com/ristea/septr) — separable attention time/frequency.
- [VLAS](https://github.com/whichwhichgone/VLAS) — robot vision/speech/action.
- [RoboOmni](https://github.com/OpenMOSS/RoboOmni) — omni-modal robot.
- [ELLSA](https://github.com/bytedance/SALMONN/tree/ELLSA) — full duplex multimodal.
- [Speech-π0.5 paper](https://arxiv.org/abs/2610.33866) — direct speech action; không đồng nhất với LeRobot π0.5 hiện có.

Giấy phép kế thừa Apache-2.0 của repository gốc.
