import os
from pathlib import Path
import logging
from typing import List

logger = logging.getLogger(__name__)


def save_processed_text(file_path: str, output_path: str, processed_text: str) -> str:
        """
        처리된 텍스트를 파일로 저장합니다.
        
        Args:
            file_path: 원본 파일 경로
            output_path: 저장할 파일 경로
            processed_text: 처리된 텍스트
        Returns:
            저장된 파일 경로
        """
        try:                        
            # 출력 파일 경로
            filename = Path(file_path).stem
            output_path = os.path.join(output_path, f"{filename}.txt")
            
            # 파일 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
                
            logger.info(f"처리된 텍스트 저장 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"텍스트 저장 중 오류 발생: {str(e)}")
            return ""

def get_files(raw_data_dir: Path, supported_extensions: List[str]) -> List[Path]:
    """
    raw_data_dir에서 처리할 모든 문서 파일들을 가져옵니다.

    Returns:
        처리할 문서 파일 경로 리스트
    """
    document_files = []

    # raw 폴더의 모든 파일 처리
    for ext in supported_extensions:
        for file in raw_data_dir.glob(f"*{ext}"):
            # 숨김 파일이나 시스템 파일은 제외
            if not file.name.startswith('.'):
                document_files.append(file)

    return document_files
