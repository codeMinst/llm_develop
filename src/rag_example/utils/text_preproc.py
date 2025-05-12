"""
텍스트 처리 유틸리티 모듈입니다.
"""
import re
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# Ollama 클라이언트 관련 변수
_ollama_client = None


def get_ollama_client():
    """
    Ollama 클라이언트를 가져오거나 생성합니다.
    """
    global _ollama_client
    
    if _ollama_client is None:
        try:
            from langchain_community.llms import Ollama
            
            logger.info("Ollama 클라이언트 생성 중...")
            _ollama_client = Ollama(model="llama3.2", temperature=0)
            logger.info("Ollama 클라이언트 생성 완료")
        except Exception as e:
            logger.error(f"Ollama 클라이언트 생성 중 오류 발생: {str(e)}")
            return None
    
    return _ollama_client


def ollama_spacing(text: str) -> str:
    """
    Ollama를 사용하여 한국어 띄어쓰기 교정을 수행합니다.
    
    Args:
        text: 교정할 텍스트
        
    Returns:
        교정된 텍스트
    """
    try:
        # Ollama 클라이언트 가져오기
        client = get_ollama_client()
        if client is None:
            logger.warning("Ollama 클라이언트 생성 실패, 원본 텍스트 반환")
            return text
        
        # 띄어쓰기 교정 프롬프트
        prompt = f"""너는 지금부터 띄어쓰기 교정기입니다. 다음 한국어 텍스트의 띄어쓰기만 교정해주세요. 
        특수문자나 다른 언어는 무시해주세요. 오직 원본 텍스트의 띄어쓰기만 수정하여 반환해주세요.
        원본 : {text}
        """
        
        # Ollama로 띄어쓰기 교정 요청
        response = client.invoke(prompt)
        
        # 디버그 로깅
        if text != response:
            logger.debug(f"원본: {text[:100]}...")
            logger.debug(f"교정: {response[:100]}...")
        
        return response
    
    except Exception as e:
        logger.error(f"Ollama 띄어쓰기 교정 중 오류 발생: {str(e)}")
        return text  # 오류 발생 시 원본 텍스트 반환

def improve_text(text: str) -> str:
    """
    간단한 텍스트 개선을 수행합니다. 줄바꿈은 유지합니다.
    
    Args:
        text: 개선할 텍스트
        
    Returns:
        개선된 텍스트
    """
    # 페이지 태그 패턴 정의
    page_tag_pattern = re.compile(r'-- \d+ page (start|end) --')
    try:
        # 줄 단위로 처리하여 줄바꿈 유지
        lines = text.split('\n')
        improved_lines = []
                
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                # 빈 줄은 그대로 유지
                improved_lines.append(line)
                continue
                
            # 한글과 영문/숫자/특수문자 사이에 공백 추가 (정규식 통합)
            # 개선된 정규식 (괄호류, 백슬래시 제외)
            line = re.sub(r'([가-힣])([a-zA-Z0-9.,;:?!])', r'\1 \2', line)
            line = re.sub(r'([a-zA-Z0-9.,;:?!])([가-힣])', r'\1 \2', line)
            
            # 연속된 공백 제거
            line = re.sub(r'\s+', ' ', line)
            
            # 문장 부호 앞의 공백 제거
            line = re.sub(r' (?=[\.,;:!?])', '', line)
            
            improved_lines.append(line)
        
        # 개선된 줄들을 다시 합치기
        result = '\n'.join(improved_lines)
        
        # 페이지 태그가 사라졌는지 확인
        result_lines = result.split('\n')
        
        # 원본 텍스트에서 페이지 태그와 위치 추출
        original_lines = text.split('\n')
        page_tags = []
        
        for i, line in enumerate(original_lines):
            if page_tag_pattern.match(line.strip()):
                page_tags.append((i, line))
        
        # 페이지 태그 다시 삽입
        for idx, tag in page_tags:
            # 원래 위치가 결과 범위를 벗어나지 않는지 확인
            if idx < len(result_lines):
                # 해당 위치에 페이지 태그가 없으면 삽입
                if not page_tag_pattern.match(result_lines[idx].strip()):
                    result_lines[idx] = tag
            else:
                # 원래 위치가 결과 범위를 벗어나면 마지막에 추가
                result_lines.append(tag)
        
        # 페이지 태그 삽입 후 결과 반환
        return '\n'.join(result_lines)
    
    except Exception as e:
        logger.error(f"텍스트 개선 중 오류 발생: {str(e)}")
        return text  # 오류 발생 시 원본 텍스트 반환
