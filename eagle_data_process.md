### Eagle2 TRL SFT 데이터 처리/튜닝 인수인계

이 문서는 `nvidia/Eagle2-2B`를 TRL의 SFT 파이프라인으로 튜닝할 때 요구되는 데이터 스키마, 전처리 로직, 콜레이터 기대값, 학습 구성 및 실행 절차를 정리한 인수인계 자료입니다. 소스 근거는 다음 파일들입니다:

- `eagle2-2b-finetuning/eagle2_trl_sft_trainer.py`
- `eagle2-2b-finetuning/eagle2_data_collator.py`
- `eagle2-2b-finetuning/eagle2_trl_sft_multiview.py`


### 핵심 개요
- **모델/프로세서**: `AutoModel/AutoProcessor.from_pretrained("nvidia/Eagle2-2B", trust_remote_code=True)`
- **학습 방법**: TRL `SFTTrainer` + LoRA (`use_dora=True`), BF16, 왼쪽 패딩
- **입력 구조**: 멀티뷰 이미지를 포함하는 chat-style `messages` 구조
- **전처리**: 텍스트/이미지 정규화 → Processor로 토큰화/비전 피처 생성 → 레이블 마스킹
- **콜레이터 기대값**: `input_ids`, `attention_mask`, `labels` + 선택적으로 `pixel_values` `[N,C,H,W]`, `image_flags` `[N]`


### 원천 데이터 → 모델 입력 스키마
원천 JSONL의 각 샘플은 대략 아래와 같은 필드를 포함합니다.
- `prompt_blocks`: 텍스트/이미지 블록 리스트
  - 각 블록은 `{ "type": "text"|"image_url", "text"?: str, "image_url"?: {"url": str} }`
- `ground_truth_answer`: 정답 텍스트 (예: 멀티클래스에서 선택지 번호 등)

SFT용으로는 다음과 같은 `messages` 포맷으로 변환합니다. 변환은 `eagle2_trl_sft_multiview.py`의 `ealge_format_multiview_data`에서 수행합니다.
- `image_url` → `type: "image"`로 치환
- `messages` 예시:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "...설명..."},
      {"type": "image", "image_url": {"url": "/path/to/img1.jpg"}},
      {"type": "text", "text": "...설명..."},
      {"type": "image", "image_url": {"url": "/path/to/img2.jpg"}}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "정답텍스트"}
    ]}
  ]
}
```

주의:
- `content` 리스트 안의 각 아이템에서 `None` 값인 키는 제거됩니다.
- 이미지 소스는 세 가지 모두 지원됩니다: PIL.Image 객체, `{bytes: ...}` 바이트, `{url: "http|https|localpath"}`. 로컬 경로/웹 URL 모두 허용됩니다.


### 전처리 파이프라인 상세 (`Eagle2TRLSFTTrainer._prepare_dataset`)
파일: `eagle2_trl_sft_trainer.py`

1) 샘플 정규화
- `messages[].content[]`에서 `None` 값 키 제거
- `type == "image"` 아이템들의 실제 이미지를 로딩하여 `content_item["image"]`에 `PIL.Image`로 주입

2) Processor 호출
- `processing_class.process_vision_info(messages)`로 멀티뷰 이미지 입력을 생성
- `apply_chat_template`로 전체 텍스트(`full_text`)와 프롬프트 전용 텍스트(`prompt_text`) 생성

3) 토큰화/피처 생성
- 이미지가 있으면 `processing_class(text=[full_text], images=image_inputs, return_tensors="pt")`
- 텍스트만이면 `images=None`로 호출. 필요한 경우 dummy `pixel_values`와 `image_flags` 추가
- 출력 텐서 모양 정규화:
  - `input_ids`, `attention_mask`, `labels`: `[1,L]` → `[L]`
  - `pixel_values`: `[1,N,C,H,W]` or `[C,H,W]` → `[N,C,H,W]` (N은 뷰 수)
  - `image_sizes`: `[1,N,2]` or `[2]` → `[N,2]`
  - `image_flags`: `[1,N]` or `[]` → `[N]` (이미지 존재 여부/뷰 수 표기)

4) 레이블 마스킹
- `labels = input_ids.clone()` 후, 프롬프트 프리픽스 길이(`prompt_text` 토크나이즈 길이)까지 `-100`으로 마스킹
- 이미지 토큰 마스킹: 모델 전용 `<IMG_CONTEXT>` 토큰 ID(없으면 151667)와 `<image>`, `<image-1..8>` 같은 플레이스홀더 토큰들을 `-100`으로 마스킹

5) 최종 컬럼 형식
- `set_format(type='torch', columns=['input_ids','attention_mask','pixel_values','image_sizes','labels','image_flags'])`


### 데이터 콜레이터 기대값 (`Eagle2DataCollator`)
파일: `eagle2_data_collator.py`

- 입력 features는 최소 다음 키를 가짐:
  - `input_ids: LongTensor[L]`
  - `attention_mask: LongTensor[L]`
  - `labels: LongTensor[L]`
  - 선택: `pixel_values: FloatTensor[N,C,H,W]`, `image_flags: BoolTensor[N]`

동작 요약:
- 텍스트 부분은 `tokenizer.pad(..., padding='longest')`로 배치 정렬
- `labels`는 좌측 패딩 길이에 맞춰 앞쪽에 `-100`으로 패딩
- 배치 내 최대 뷰 수(`max_num_views`)를 기준으로 per-sample `pixel_values`를 `[n,C,H,W]`로 정규화/제로패딩 후 `[B,N,C,H,W]` → `[B*N,C,H,W]`로 리쉐입
- 동일하게 `image_flags`도 `[B,N]` → `[B*N]`로 리쉐입
- 안전장치로 `input_ids` 내 이미지 토큰(`<IMG_CONTEXT>` 또는 기본값 151667)에 대응하는 `labels` 위치를 `-100`으로 마스킹


### 학습 구성 (SFT)
파일: `eagle2_trl_sft_multiview.py`

- 모델/프로세서
  - `AutoModel.from_pretrained("nvidia/Eagle2-2B", trust_remote_code=True, torch_dtype=torch.bfloat16)`
  - `AutoProcessor.from_pretrained("nvidia/Eagle2-2B", trust_remote_code=True)`
  - `processor.tokenizer.padding_side = "left"`
  - `processor.tokenizer.pad_token = "<|endoftext|>"`, `pad_token_id = 151643`

- LoRA 설정
  - `r=32, lora_alpha=8, lora_dropout=0.1, use_dora=True`
  - `target_modules=["down_proj","o_proj","k_proj","q_proj","gate_proj","up_proj","v_proj"]`

- SFTConfig 주요 옵션
  - `num_train_epochs=5`, `per_device_train_batch_size=1`, `per_device_eval_batch_size=1`
  - `gradient_accumulation_steps=64`, `bf16=True`, `remove_unused_columns=False`
  - `optim="adamw_torch_fused"`, `learning_rate=2e-4`, `lr_scheduler_type="cosine"`
  - `eval_strategy="steps"`, `eval_steps=50`, `save_strategy="steps"`, `save_steps=100`
  - `label_names=["labels"]`, `use_legacy_prediction_loop=True`, `report_to="wandb"`

- W&B
  - `entity="schaeck-dongguk-university"`, `project="eagle2-2b-finetuning"`

- 캐시 최적화 노트
  - HF 캐시의 `config.json`, `preprocessor_config.json`에서 `"max_dynamic_tiles": 1`로 조정 권장 (VRAM 절약)


### 데이터 준비 절차
1) 원천 JSONL 준비
- 경로는 스크립트의 `dataset_path`를 사용하며 환경에 맞게 수정 필요
- 각 샘플은 `prompt_blocks`와 `ground_truth_answer`를 포함

2) SFT용 리스트/데이터셋 변환
- `eagle2_trl_sft_multiview.py`에서:
  - `dataset = load_dataset("json", data_files=dataset_path)`
  - `train/test` 분리 후, `ealge_format_multiview_data(sample)`로 `messages` 생성
  - `Dataset.from_list(...)`로 허깅페이스 데이터셋 구성 및 `save_to_disk`로 캐싱 가능

3) 전처리/토큰화/비전 피처 생성
- `Eagle2TRLSFTTrainer._prepare_dataset`가 전체 데이터셋에 `map`으로 적용되어 자동 처리


### 실행 방법
아래는 단일 GPU 예시입니다. 환경에 맞춰 `dataset_path`, W&B 설정 등을 조정하세요.

```bash
python /workspace/verl/eagle2-2b-finetuning/eagle2_trl_sft_multiview.py
```

학습 산출물은 `SFTConfig.output_dir`에 저장됩니다 (예: `eagle2-2b-trl-sft-Multitask`).


### 자주 발생하는 이슈와 해결
- 에러: "can only concatenate str (not 'list') to str" (chat template 렌더링)
  - 원인: `messages[].content[]` 내부 구조가 템플릿 기대와 불일치하거나 `text`가 리스트형으로 들어간 경우
  - 조치: `ealge_format_multiview_data` 포맷을 준수하고, `content` 아이템의 `text`는 문자열이어야 함. `None` 키는 제거됨

- 경고: `trust_remote_code=True` 필요
  - 조치: `AutoModel/AutoProcessor` 생성 시 이미 설정되어 있음

- 이미지 경로/URL 로딩 실패
  - 조치: 로컬 경로 존재 확인 또는 HTTP 응답 코드 확인. 실패 시 해당 이미지 블록은 무시됨

- 라벨 마스킹 확인
  - 프롬프트 프리픽스, `<IMG_CONTEXT>`(기본 151667), `<image>`/`<image-i>` 플레이스홀더는 모두 `-100`으로 마스킹되어 로스에 기여하지 않음


### 스키마 요약 체크리스트
- 입력 샘플: `messages: [{role: 'user'|'assistant', content: [{type: 'text'|'image', text?: str, image_url?: {url: str}, image?: PIL.Image}]}]`
- 전처리 산출: `input_ids [L]`, `attention_mask [L]`, `labels [L] (마스킹 포함)`, 선택 `pixel_values [N,C,H,W]`, `image_flags [N]`, 필요 시 `image_sizes [N,2]`
- 콜레이터 산출(배치):
  - 텍스트: longest pad
  - 레이블: 좌측 패딩분 `-100`
  - 비전: `[B,N,C,H,W]→[B*N,C,H,W]`, `image_flags [B*N]`


### 참고/경로
- 데이터 변환: `eagle2_trl_sft_multiview.py`
- 전처리(토크나이즈/비전/마스킹): `eagle2_trl_sft_trainer.py`
- 배치 콜레이트: `eagle2_data_collator.py`


