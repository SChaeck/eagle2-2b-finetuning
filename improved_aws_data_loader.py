import json
import os
import tempfile
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
import cv2
import subprocess
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSVideoFrameExtractor:
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region_name='ap-northeast-2'):
        """
        AWS S3 클라이언트 초기화
        
        Args:
            aws_access_key_id: AWS Access Key ID (None이면 환경변수나 IAM 역할 사용)
            aws_secret_access_key: AWS Secret Access Key 
            region_name: AWS 리전 (기본값: ap-northeast-2, 서울)
        """
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # 환경변수나 IAM 역할에서 자격증명 사용
                self.s3_client = boto3.client('s3', region_name=region_name)
                
            logger.info("AWS S3 클라이언트가 성공적으로 초기화되었습니다.")
        except NoCredentialsError:
            logger.error("AWS 자격증명을 찾을 수 없습니다.")
            raise
    
    def parse_filename(self, filename):
        """
        파일명에서 정보 추출
        새로운 형식: "fix_center_episode_000024_20250831_135639_0feef1d5_0.25_5.0s.jpg"
        기존 형식: "fix_left-0.75-24.1s.jpg"
        """
        # 새로운 형식 파싱 (더 복잡한 형식)
        # 패턴: view_episode_숫자_날짜_시간_해시_ratio_timestamp.jpg
        # view는 fix_center, ego_left, fix_right_1 등 다양한 형태 가능
        new_pattern = r'([^_]+(?:_[^_]+)*?)_episode_\d+_\d+_\d+_[a-f0-9]+_([0-9.]+)_([0-9.]+)s\.jpg'
        match = re.match(new_pattern, filename)
        
        if match:
            view = match.group(1)
            ratio = float(match.group(2))
            timestamp = float(match.group(3))
            return view, ratio, timestamp
        
        # 기존 형식 파싱 (fallback)
        old_pattern = r'([^-]+)-([0-9.]+)-([0-9.]+)s\.jpg'
        match = re.match(old_pattern, filename)
        
        if match:
            view = match.group(1)
            ratio = float(match.group(2))
            timestamp = float(match.group(3))
            return view, ratio, timestamp
        
        raise ValueError(f"파일명 형식이 올바르지 않습니다: {filename}")
    
    def map_camera_name(self, view):
        """
        카메라 명 매핑 함수
        ego_leftside -> ego_left, ego_rightside -> ego_right 등의 변환
        
        Args:
            view: 원본 카메라 뷰 이름
            
        Returns:
            str: 매핑된 카메라 뷰 이름
        """
        # 카메라 명 매핑 규칙
        camera_mapping = {
            'ego_leftside': 'ego_left',
            'ego_rightside': 'ego_right',
            # 필요에 따라 추가 매핑 규칙을 여기에 추가
            # 'old_name': 'new_name',
        }
        
        # 매핑된 이름이 있으면 반환, 없으면 원본 반환
        mapped_view = camera_mapping.get(view, view)
        
        if mapped_view != view:
            logger.debug(f"카메라 명 매핑: {view} -> {mapped_view}")
        
        return mapped_view
    
    def download_video(self, bucket_name, s3_key, local_path):
        """
        S3에서 비디오 다운로드
        
        Args:
            bucket_name: S3 버킷 이름
            s3_key: S3 객체 키 (MP4 파일)
            local_path: 로컬 저장 경로
            
        Returns:
            bool: 다운로드 성공 여부
        """
        try:
            # s3_key가 's3://bucket/key' 형식이면 bucket과 key를 파싱해서 사용
            if s3_key.startswith("s3://"):
                # 예: s3://configint/external_dataset/v1.0/robo_set/episode_000050/fix_center.mp4
                s3_key_no_prefix = s3_key[5:]  # 's3://'
                parts = s3_key_no_prefix.split('/', 1)
                if len(parts) == 2:
                    bucket_name, s3_key = parts
                else:
                    raise ValueError(f"s3_key 형식이 올바르지 않습니다: {s3_key}")
            
            # 로컬 디렉토리 생성
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # S3에서 비디오 파일 다운로드
            self.s3_client.download_file(bucket_name, s3_key, local_path)
            logger.info(f"비디오 다운로드 완료: {s3_key} -> {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"비디오 다운로드 실패: {s3_key} - {str(e)}")
            return False
        except Exception as e:
            logger.error(f"예상치 못한 오류: {str(e)}")
            return False
    
    def get_video_duration(self, video_path):
        """
        비디오 길이 구하기 (ffprobe 사용)
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info['format']['duration'])
                return duration
            else:
                logger.error(f"ffprobe 실패: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"비디오 길이 구하기 실패: {str(e)}")
            return None
    
    def extract_frame_ffmpeg(self, video_path, timestamp, output_path):
        """
        ffmpeg를 사용해서 특정 시간의 프레임 추출
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cmd = [
                'ffmpeg', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"프레임 추출 완료: {output_path} (시간: {timestamp}초)")
                return True
            else:
                logger.error(f"ffmpeg 프레임 추출 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"프레임 추출 중 오류: {str(e)}")
            return False
    
    def extract_frame_opencv(self, video_path, timestamp, output_path):
        """
        OpenCV를 사용해서 특정 시간의 프레임 추출 (fallback)
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"비디오 파일을 열 수 없습니다: {video_path}")
                return False
            
            # FPS와 총 프레임 수 구하기
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            # 특정 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(output_path, frame)
                logger.info(f"프레임 추출 완료: {output_path} (시간: {timestamp}초)")
                cap.release()
                return True
            else:
                logger.error(f"프레임을 읽을 수 없습니다: 프레임 {frame_number}")
                cap.release()
                return False
                
        except Exception as e:
            logger.error(f"OpenCV 프레임 추출 중 오류: {str(e)}")
            return False
    
    def process_jsonl_file(self, jsonl_file_path, use_ffmpeg=True, temp_video_dir=None, clear_existing=False):
        """
        JSONL 파일을 읽어서 비디오를 다운로드하고 프레임을 추출 (개선된 버전)
        Task2의 2개 시나리오를 지원합니다.
        
        Args:
            jsonl_file_path: JSONL 파일 경로
            use_ffmpeg: ffmpeg 사용 여부 (False면 OpenCV 사용)
            temp_video_dir: 비디오 임시 저장 디렉토리 (None이면 자동 생성)
            clear_existing: 기존 이미지 파일들을 삭제하고 시작할지 여부
            
        Returns:
            list: 처리된 데이터 리스트
        """
        if temp_video_dir is None:
            temp_video_dir = tempfile.mkdtemp(prefix="temp_videos_")
        
        logger.info(f"비디오 임시 디렉토리: {temp_video_dir}")
        
        # 기존 파일 삭제 옵션
        if clear_existing:
            self.clear_existing_images(jsonl_file_path)
        
        processed_data = []
        downloaded_videos = {}  # 중복 다운로드 방지를 위한 캐시
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    data = json.loads(line.strip())
                    logger.info(f"라인 {line_num} 처리 중... (Task: {data.get('task', 'unknown')})")
                    
                    # S3 정보 추출
                    s3_info = data.get('s3_info', {})
                    bucket_name = s3_info.get('bucket')
                    
                    # Task2의 경우 시나리오별로 views를 처리
                    task = data.get('task', '')
                    views_dict = {}
                    
                    if task == 'task2':
                        # Task2: 2개 시나리오 지원
                        scenario1_views = s3_info.get('scenario1_views', {})
                        scenario2_views = s3_info.get('scenario2_views', {})
                        views_dict['scenario1'] = scenario1_views
                        views_dict['scenario2'] = scenario2_views
                        
                        logger.info(f"Task2 감지 - 시나리오1: {len(scenario1_views)}개 뷰, 시나리오2: {len(scenario2_views)}개 뷰")
                        
                        if not bucket_name or (not scenario1_views and not scenario2_views):
                            logger.warning(f"라인 {line_num}: Task2 S3 정보가 불완전합니다.")
                            continue
                    else:
                        # Task1, Task3: 기존 방식
                        views = s3_info.get('views', {})
                        views_dict['default'] = views
                        
                        logger.info(f"{task} 감지 - {len(views)}개 뷰")
                        
                        if not bucket_name or not views:
                            logger.warning(f"라인 {line_num}: S3 정보가 불완전합니다.")
                            continue
                    
                    # 이미지 정보 처리
                    updated_image_info = []
                    
                    for img_info in data.get('image_info', []):
                        original_path = img_info.get('path')
                        view = img_info.get('view')
                        scenario = img_info.get('scenario', 'default')  # Task2에서 사용
                        
                        if not original_path or not view:
                            continue
                        
                        try:
                            # 파일명에서 정보 추출
                            filename = Path(original_path).name
                            parsed_view, ratio, timestamp = self.parse_filename(filename)
                            
                            # view가 일치하는지 확인
                            if parsed_view != view:
                                logger.warning(f"뷰 불일치: {parsed_view} != {view}")
                                continue
                            
                            # 시나리오별 views 선택
                            current_views = views_dict.get(scenario, {})
                            if not current_views:
                                logger.warning(f"시나리오 '{scenario}'의 views를 찾을 수 없습니다.")
                                continue
                            
                            # 카메라 명 매핑 (ego_leftside -> ego_left, ego_rightside -> ego_right)
                            mapped_view = self.map_camera_name(view)
                            
                            # S3에서 해당 뷰의 비디오 파일 경로
                            if mapped_view not in current_views:
                                # 원본 view명으로도 시도
                                if view not in current_views:
                                    logger.warning(f"시나리오 '{scenario}'에서 뷰 '{view}' (매핑: {mapped_view})를 찾을 수 없습니다.")
                                    logger.debug(f"사용 가능한 뷰: {list(current_views.keys())}")
                                    continue
                                else:
                                    video_s3_key = current_views[view]
                                    logger.info(f"원본 뷰명 사용: {view}")
                            else:
                                video_s3_key = current_views[mapped_view]
                                logger.info(f"매핑된 뷰명 사용: {view} -> {mapped_view}")
                            
                            # 비디오 다운로드 (캐시 확인) - 시나리오별로 구분
                            video_cache_key = f"{scenario}_{view}_{video_s3_key}"
                            
                            # 시나리오별로 비디오 파일명 생성 (에피소드 정보 포함)
                            sequence = img_info.get('sequence', '')
                            if sequence:
                                episode_info = sequence.split('_')[-1] if '_' in sequence else 'unknown'
                                video_local_path = os.path.join(temp_video_dir, f"{scenario}_{view}_{episode_info}.mp4")
                            else:
                                video_local_path = os.path.join(temp_video_dir, f"{scenario}_{view}.mp4")
                            
                            if video_cache_key not in downloaded_videos:
                                if self.download_video(bucket_name, video_s3_key, video_local_path):
                                    downloaded_videos[video_cache_key] = video_local_path
                                    logger.info(f"비디오 다운로드 완료: {scenario}_{view}")
                                else:
                                    logger.error(f"비디오 다운로드 실패: {video_s3_key}")
                                    continue
                            else:
                                video_local_path = downloaded_videos[video_cache_key]
                            
                            # JSON에 지정된 경로 그대로 사용
                            image_output_path = original_path
                            
                            # 이미 파일이 존재하는지 확인
                            if os.path.exists(image_output_path):
                                logger.info(f"이미지가 이미 존재함, 건너뜀: {image_output_path}")
                                success = True
                            else:
                                # 프레임 추출
                                success = False
                                if use_ffmpeg:
                                    success = self.extract_frame_ffmpeg(video_local_path, timestamp, image_output_path)
                                
                                if not success:
                                    # ffmpeg 실패시 OpenCV로 fallback
                                    success = self.extract_frame_opencv(video_local_path, timestamp, image_output_path)
                            
                            if success:
                                img_info_copy = img_info.copy()
                                img_info_copy['source_video'] = video_s3_key
                                img_info_copy['extracted_timestamp'] = timestamp
                                img_info_copy['video_ratio'] = ratio
                                img_info_copy['scenario'] = scenario  # 시나리오 정보 보존
                                updated_image_info.append(img_info_copy)
                                logger.info(f"프레임 추출 성공: {scenario} - {view} - {timestamp}초")
                            else:
                                logger.error(f"프레임 추출 실패: {filename}")
                                
                        except Exception as e:
                            logger.error(f"이미지 처리 중 오류: {str(e)}")
                            continue
                    
                    # 업데이트된 데이터 저장
                    data_copy = data.copy()
                    data_copy['image_info'] = updated_image_info
                    data_copy['temp_video_dir'] = temp_video_dir
                    processed_data.append(data_copy)
                    
                    logger.info(f"라인 {line_num} 처리 완료: {len(updated_image_info)}개 이미지 처리됨")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"라인 {line_num}: JSON 파싱 오류 - {str(e)}")
                except Exception as e:
                    logger.error(f"라인 {line_num}: 처리 오류 - {str(e)}")
        
        logger.info(f"처리 완료. 총 {len(processed_data)}개 항목 처리됨")
        
        # 임시 비디오 파일들 정리 (선택사항)
        # self.cleanup_temp_videos(temp_video_dir)
        
        return processed_data
    
    def clear_existing_images(self, jsonl_file_path):
        """
        JSONL 파일에 명시된 모든 이미지 파일들을 삭제
        """
        logger.info("기존 이미지 파일들을 삭제하는 중...")
        deleted_count = 0
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    for img_info in data.get('image_info', []):
                        image_path = img_info.get('path')
                        if image_path and os.path.exists(image_path):
                            try:
                                os.remove(image_path)
                                deleted_count += 1
                                logger.debug(f"삭제됨: {image_path}")
                            except Exception as e:
                                logger.warning(f"파일 삭제 실패: {image_path} - {str(e)}")
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"기존 이미지 {deleted_count}개 파일 삭제 완료")
    
    def cleanup_temp_videos(self, temp_video_dir):
        """
        임시 비디오 파일들 정리
        """
        try:
            import shutil
            shutil.rmtree(temp_video_dir)
            logger.info(f"임시 비디오 디렉토리 정리 완료: {temp_video_dir}")
        except Exception as e:
            logger.warning(f"임시 파일 정리 실패: {str(e)}")

    def analyze_dataset_structure(self, jsonl_file_path, max_lines=10):
        """
        데이터셋 구조를 분석하여 Task별 특성을 파악
        """
        logger.info(f"데이터셋 구조 분석 중: {jsonl_file_path}")
        
        task_counts = {}
        sample_data = {}
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line_num > max_lines:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    task = data.get('task', 'unknown')
                    
                    # Task별 카운트
                    task_counts[task] = task_counts.get(task, 0) + 1
                    
                    # 샘플 데이터 저장
                    if task not in sample_data:
                        sample_data[task] = {
                            'sample_id': data.get('sample_id', ''),
                            's3_info_keys': list(data.get('s3_info', {}).keys()),
                            'image_count': len(data.get('image_info', [])),
                            'has_scenarios': any('scenario' in img for img in data.get('image_info', []))
                        }
                
                except json.JSONDecodeError:
                    continue
        
        logger.info("=== 데이터셋 구조 분석 결과 ===")
        for task, count in task_counts.items():
            info = sample_data.get(task, {})
            logger.info(f"Task: {task}")
            logger.info(f"  - 샘플 수: {count}")
            logger.info(f"  - S3 정보 키: {info.get('s3_info_keys', [])}")
            logger.info(f"  - 이미지 수: {info.get('image_count', 0)}")
            logger.info(f"  - 시나리오 정보: {info.get('has_scenarios', False)}")
            logger.info("")
        
        return task_counts, sample_data
