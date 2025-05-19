# 🚀 LLMOps 개발을 위한 RAG 시스템 예제

이 프로젝트는 LangChain 0.3.x와 LangGraph를 활용한 구조적인 RAG 시스템 구현에 중점을 둔 모듈형 파이프라인으로, LLMOps 환경에서 운영될 수 있는 실용적인 예제입니다. 

이 시스템은 다음과 같은 특징을 가지고 있습니다:

1. **체계적인 컴포넌트 구성**: 각 기능을 담당하는 컴포넌트가 명확히 분리되어 있어 유지보수성이 뛰어남
2. **확장 가능한 아키텍처**: 새로운 문서 형식, 임베딩 모델, LLM 등을 쉽게 추가할 수 있는 구조
3. **파이프라인 기반 처리**: 문서 수집부터 질의응답까지 일관된 흐름으로 처리
4. **웹 및 CLI 인터페이스**: 다양한 환경에서 시스템을 활용할 수 있는 인터페이스 제공

이 시스템은 다양한 문서 형식을 처리하고, 벡터 임베딩 및 LLM 기반 질의응답 흐름을 효율적으로 관리하는 방법을 보여줍니다.

## 📋 주요 기능

- **문서 수집 및 처리**
  - PDF 및 텍스트 문서에서 텍스트 추출 및 청크 분할
  - 페이지 태그 관리 및 텍스트 정규화

- **벡터 임베딩 및 저장**
  - HuggingFace 기반 임베딩 모델 활용
  - Chroma 벡터 데이터베이스를 통한 효율적인 저장 및 검색
  - 벡터 저장소 초기화 및 관리 기능

- **LangGraph 활용 워크플로우**
  - GraphRAGChainBuilder를 통한 노드기반 워크플로우 구성
  - 질의응답 프로세스의 단계별 관리 및 제어
  - 확장 가능한 노드 기반 아키텍처

- **RAG 체인 구성 및 질의응답**
  - LangChain 0.3.x 기반 최신 LCEL(LangChain Expression Language) 구현
  - RunnableWithMessageHistory를 활용한 세션별 메시지 히스토리 관리
  - 사용자 정의 프롬프트 템플릿을 통한 질의응답 품질 최적화
  - 다양한 LLM 지원 (Ollama, Claude 등 확장가능)
  - 상태 관리 및 대화 초기화 기능 
  - 문서 검색 및 관련도 점수 기반 정렬 알고리즘 적용
  - 요약 정보 운영 및 메모리 관리 기능

- **웹 인터페이스**
  - Gradio 기반 직관적인 웹 UI 제공
  - 세션 관리 및 대화 기록 유지
  - 문서 정보 조회 및 벡터 저장소 관리 기능

## 프로젝트 구조

```
src/rag_example/
├── adapters/           # 다양한 문서 포맷을 처리하는 어댑터 계층
│   ├── base/          # 기본 어댑터 인터페이스 및 타입 정의
│   ├── document/      # PDF, 텍스트 등 문서별 어댑터 구현
│   └── doc_factory.py # 문서 타입에 따른 어댑터 생성 팩토리
├── config/            # 설정 관리
│   └── settings.py    # 경로, 모델, 청크 크기 등 주요 설정
├── pipeline/          # 파이프라인 핵심 흐름을 단계별로 분리
│   ├── ingestion/     # 문서 로딩, 추출, 청크 분할 처리
│   ├── indexing/      # 임베딩 및 벡터 저장소 생성
│   ├── querying/      # RAG 체인 구성 및 프롬프트 정의
│   └── rag_pipeline.py # 전체 파이프라인 오케스트레이션
├── utils/              # 상수, 텍스트 전처리, 파일 입출력 유틸리티
│   ├── constants.py   # 시스템 상수 정의
│   ├── text_preproc.py # 텍스트 전처리 및 개선 기능
│   └── file_io.py     # 파일 입출력 관련 유틸리티
├── ui.py              # Gradio 기반 웹 인터페이스
└── main.py            # CLI 기반 실행 엔트리포인트
```

## 🧩 아키텍처 및 설계 원칙

이 프로젝트는 다음과 같은 설계 원칙을 준수하여 개발되었습니다:

- **파이프라인 패턴**: 데이터 흐름을 명확한 단계로 구분하여 처리
  - 문서 로딩 및 청크 분할 → 벡터 저장소 생성 → RAG 체인 구성 → 질의응답
  - 각 단계는 독립적으로 개발하고 테스트 가능

- **관심사 분리**: 각 컴포넌트는 하나의 명확한 책임만 가짐
  - DocumentLoader: 문서 로딩과 청크 분할만 담당
  - VectorStoreBuilder: 임베딩과 벡터 저장소 관리만 담당
  - RAGChainBuilder: 질의응답 처리만 담당

- **유연한 구현**: 각 컴포넌트는 내부 구현 세부사항을 캡슐화
  - 인터페이스와 구현을 분리하여 구현 변경이 외부에 영향을 미치지 않도록 설계
  - 패토리 패턴, 전략 패턴 등 적절한 디자인 패턴 적용

- **확장성**: 새로운 기능 추가가 용이한 구조
  - 새로운 문서 포맷: 어댑터 패턴을 통해 쉽게 추가 가능
  - 새로운 LLM: 팩토리 패턴을 통해 새로운 모델 쉽게 통합

- **세션 관리**: 웹 환경에서 동시 사용자 지원
  - 고유한 세션 ID를 통한 대화 기록 독립적 관리
  - 세션별 메시지 히스토리 관리로 고립된 사용자 경험 제공

## 💻 사용 방법

### CLI 모드

```bash
# 기존 벡터 저장소 유지하며 실행
conda run -n test python -m src.rag_example.main

# 기존 벡터 저장소 삭제 후 새로 생성
conda run -n test python -m src.rag_example.main --clean-db
```

### 웹 UI 모드

```bash
conda run -n test python -m src.rag_example.ui
```

## 🔮 향후 계획

이 프로젝트는 다음과 같은 방향으로 확장될 예정입니다:

- **LLMOps 시스템 구축**
  - LLM 애플리케이션 전체 생애주기 관리
  - 개발, 배포, 모니터링, 평가를 위한 통합 플랫폼 구축
  - 자동화된 성능 측정 및 리포팅 시스템 구축

- **LangGraph 기반 워크플로우**
  - MapReduce 패턴을 이용한 다단계 요약 기능 구현
  - 긴 문서를 분할 요약 후 통합하는 계층적 처리
  - 노드 기반 워크플로우를 활용한 복잡한 처리 과정 시각화

- **MCP(Model Context Protocol) 활용**
  - 필요에 따라 다양한 MCP 서버를 유연하게 활용할 수 있는 개방형 구조 구현
  - 모듈화된 컨텍스트 관리 시스템 개발

- **멀티모달 지원**
  - 이미지, 오디오 등 다양한 형태의 데이터 처리 지원 확장
  - 이미지 분석 및 이해를 통한 멀티모달 RAG 시스템 구축

--- 

# 프로젝트 상세

## 주요 컴포넌트 설명 (Key Components)

### **main.py** – CLI 인터페이스 진입점

메인 모듈로, 커맨드라인에서 시스템을 실행할 때 사용됩니다. `main()` 함수에서 **RAGPipeline**을 초기화하고, 사용자의 입력을 반복해서 받는 대화 루프를 시작합니다. 주요 기능:

* `--clean-rag` 인자에 따라 벡터스토어를 초기화(clean)할지 결정합니다.
* **RAGPipeline**을 생성하고 `.setup_chain()`을 호출하여 질의응답 **체인 객체**를 준비합니다.
* 무한 루프를 돌며 사용자 질문을 입력 받고, `"exit"`나 `"quit"` 등의 종료 명령 입력 시 루프를 탈출합니다.
* `"reset"` 또는 `"clear"` 입력 시 현재 대화 세션의 메모리를 초기화하고, `"reset all"` 입력 시 모든 세션의 메모리를 초기화합니다 (체인의 `reset_memory` 메서드 이용).
* 일반 질문은 준비된 **rag\_chain** 객체의 `run(query)`을 호출하여 답변을 생성하고 출력합니다. 사용자의 질문과 답변은 연속적인 대화로 처리되며, SummarizingMemory를 통해 대화 히스토리가 관리됩니다.

### **ui.py** – Gradio 웹 UI 인터페이스

웹 기반 데모 인터페이스를 제공하는 모듈입니다. **Gradio** 라이브러리를 사용하여 손쉽게 대화형 웹 UI를 구성합니다. 주요 내용:

* `RAGUI` 클래스: UI 초기화와 이벤트 핸들링을 담당합니다. 내부에서 `RAGPipeline`과 RAG 체인(`rag_chain`)을 관리합니다.

  * 초기화 시 `self.session_id`를 UUID로 생성하여 사용자 세션을 구분하고, `WELCOME_MESSAGES` (초기 안내 메시지 리스트)를 채팅 히스토리에 표시합니다【17†】.
  * `initialize_pipeline(clean_vectorstore=True)` 메서드: RAGPipeline을 생성 (`document_dir`, `LLM_TYPE` 등 설정)하고 체인을 빌드합니다. `clean_vectorstore` 인자에 따라 기존 벡터스토어를 새로 만들지 결정합니다.
* `create_ui()` 메서드: Gradio Blocks 레이아웃을 정의하여 대화창, 입력창, 버튼 등을 배치합니다.

  * 좌측 컬럼: 채팅 대화 (`gr.Chatbot`)와 질문 입력창(`gr.Textbox`), 제출 버튼, 대화 초기화 및 로그아웃 버튼.
  * 우측 컬럼: **시스템 관리 패널**로, 문서 정보 표시(`doc_info`), 벡터스토어 재빌드 버튼, 시스템 메시지 표시 등이 포함됩니다.
* 이벤트 연결:

  * 질문 입력 또는 “질문하기” 버튼 클릭 시 `process_query` 메서드가 호출되어 질문을 처리하고 답변을 채팅창에 추가합니다.
  * “chat 초기화” 클릭 시 `reset_conversation`으로 현재 세션 대화를 초기화하고 환영 메시지로 되돌립니다.
  * “벡터 저장소 초기화” 클릭 시 `rebuild_vectorstore`로 문서 임베딩 인덱스를 재생성하고,
  * “새로고침” 버튼으로 최신 문서 정보를 다시 불러옵니다.
* `process_query(query, history)` 메서드: 웹 UI에서 사용자의 질의를 처리하는 핵심 로직입니다.

  * IP별 요청 제한, 빈 질문 무시 등의 검증을 거쳐, 앞서 언급한 **rag\_chain**의 `run(query, session_id)`을 호출합니다. 여기서 `session_id`를 전달하여 세션별 대화 메모리를 유지합니다.
  * 실행 시간 등을 로깅하며, 생성된 답변을 대화 히스토리에 추가하고 반환합니다.
  * 특수 명령 (`reset`, `clear` 등)도 처리하여 대화를 초기화할 수 있습니다.
* 이러한 UI를 통해 사용자들은 웹 브라우저 상에서 질문을 입력하고, 백엔드에서 RAG 체인이 답변을 생성해 실시간으로 응답받을 수 있습니다.

### **config/settings.py** – 환경 설정 모듈

시스템 전역 설정 값들을 정의합니다. 예를 들어:

* **데이터 경로**: `BASE_DIR` 및 `DATA_DIR`, 원본 문서 위치인 `RAW_DATA_DIR` 등이 정의되어 있습니다.
* **데이터 전처리**: 문서를 쪼갤 때 사용하는 `CHUNK_SIZE` (청크 크기, 예: 1000자)와 `CHUNK_OVERLAP` (청크 간 겹치는 부분, 예: 150자)【25†】【26†】, `CHUNK_SEPARATORS` (청크 분할 구분자 리스트)【27†】를 지정합니다. 이 값들을 통해 문서를 일정 길이로 겹치며 분할합니다.
* **LLM 및 검색 설정**: 기본 사용할 LLM 타입 (`LLM_TYPE`, 예: "ollama"), 모델 이름 (`MODEL_NAME`), 검색 시 가져올 문서 수 (`SEARCH_K`), 대화 메모리로 유지할 최근 턴 수 (`MAX_RECENT_TURNS`, 예: 3) 등이 설정됩니다. 이러한 설정은 체인 빌더와 메모리 모듈 등에서 활용됩니다.

### **pipeline/rag\_pipeline.py** – RAG 파이프라인 클래스 (전체 오케스트레이션)

**RAGPipeline** 클래스는 RAG 시스템의 전반적인 **파이프라인 구성과 실행**을 관리합니다. **파이프라인 패턴**을 적용하여 질의응답에 필요한 단계를 순차적으로 처리합니다. 주요 내용:

* **초기화** (`RAGPipeline.__init__`): 문서 디렉토리, 청크 크기 및 겹침, 벡터스토어 초기화 여부, LLM 종류 등을 파라미터로 받아 설정합니다. 이때 기본값은 `settings.py`에 정의된 상수를 사용합니다.

  ```python
  pipeline = RAGPipeline(document_dir=RAW_DATA_DIR, is_clean_vectorstore=False, llm_type="ollama")
  ```

  초기화 시 내부에서 DocumentLoader, VectorStoreBuilder, (Graph)RAGChainBuilder 인스턴스를 생성할 준비를 합니다.
* **`setup_chain()` 메서드**: 전체 RAG 파이프라인을 실행하여 질의응답 **체인 객체**를 준비합니다. 이 과정에서 다음 단계를 순차적으로 수행합니다:

  1. **문서 로딩** – `DocumentLoader`를 통해 지정된 폴더의 문서를 모두 불러옵니다. 로드된 문서는 LangChain의 `Document` 객체 목록으로 반환됩니다.
  2. **문서 분할** – 큰 문서들은 지정된 `CHUNK_SIZE`와 `CHUNK_OVERLAP`에 따라 적절한 크기로 **청크 분할**됩니다. (예: 1000자씩 150자 겹치도록).
  3. **벡터 임베딩 & 색인** – `VectorStoreBuilder`를 이용해 각 청크에 대해 임베딩 벡터를 생성하고, **벡터 저장소**(예: Chroma 데이터베이스)를 구축합니다. 기존에 생성된 벡터스토어가 있고 `is_clean_vectorstore=False`라면, 재생성을 생략하고 로드할 수도 있습니다.
  4. **RAG 체인 구성** – 마지막 단계로, 질의응답을 수행할 체인을 빌드합니다. 현재 구현에서는 **GraphRAGChainBuilder** (LangGraph 기반 체인 빌더)를 생성하여 사용하며, 필요 시 일반 **RAGChainBuilder**로 대체 가능합니다. 이 체인 빌더는 LLM, 프롬프트, 메모리, 검색기 등을 조합하여 최종적인 QA 체인을 구성합니다.
* 모든 준비 과정을 마치면 최종적으로 체인 빌더 객체 (`self.chain_builder`)를 반환하며, 이 객체의 `.run(query)` 메서드를 통해 이후 질의응답을 수행합니다.
* **설계 의도:** RAGPipeline은 “문서 -> 임베딩 -> 체인”의 흐름을 한 곳에서 관리하여, 상위 모듈(main.py, ui.py 등)이 손쉽게 전체 시스템을 초기화할 수 있도록 합니다. 각 단계의 구현 세부사항은 개별 컴포넌트(DocumentLoader, VectorStoreBuilder 등)에 캡슐화되어 있으며, 파이프라인은 이들을 순서대로 호출만 합니다.

### **pipeline/ingestion/document\_loader.py** – 문서 로더 및 전처리

**DocumentLoader** 클래스는 지정한 디렉터리에서 문서를 읽고 전처리하는 역할을 합니다. 특징:

* **다양한 문서 형식 지원:** `rag_example/adapters` 패키지의 어댑터들을 활용하여 PDF, 텍스트 등 여러 형식의 파일을 처리합니다. 내부적으로 **어댑터 패턴**이 적용되어, 파일 확장자에 따라 적절한 DocumentAdapter를 생성합니다. 예를 들어 `.pdf` 파일은 PDFAdapter를, `.txt` 파일은 TextAdapter를 사용합니다.
* **파일 로드:** `load_documents()` 메서드는 `os.walk`를 통해 폴더 내의 모든 파일 경로를 찾고, 각 파일에 대해 `doc_factory.get_document_proc(ext, path)`를 호출합니다. 이 팩토리는 파일 확장자에 맞는 어댑터 (예: PDFExtractor/Adapter, TextExtractor/Adapter)를 생성하고 실행하여, 파일의 **텍스트 추출 및 정제**를 수행합니다.
* **문서 반환:** 각 어댑터는 LangChain의 `Document` 객체 리스트를 반환하며, DocumentLoader는 이들을 모아 전체 문서 리스트를 생성합니다. Document 객체에는 `page_content` (추출된 텍스트)와 `metadata` (파일명, 경로 등 메타정보)가 포함됩니다.
* **문서 청크 분할:** 로드된 문서 리스트를 입력받아 `split_documents(documents)` 메서드로 **내용 분할**을 수행합니다. LangChain의 텍스트 분할 유틸리티(예: RecursiveCharacterTextSplitter 등)를 활용하여, 설정된 `CHUNK_SIZE` 단위로 청크를 나누고 겹쳐 이어주는 방식입니다. 이렇게 분할된 작은 Document 조각들의 리스트를 반환하여 이후 벡터 임베딩에 사용합니다.

### **pipeline/indexing/vectorstore\_builder.py** – 벡터스토어 빌더 (임베딩 & 인덱싱)

**VectorStoreBuilder** 클래스는 문서 청크로부터 임베딩 벡터를 생성하고 벡터 검색 인덱스를 구축하는 역할을 합니다. 주요 기능:

* **임베딩 생성:** 지정한 임베딩 모델을 사용하여 각 문서 청크를 벡터화합니다. (예: OpenAI의 임베딩 API, HuggingFace 임베딩 모델 등). `langchain`에서 추상화한 임베딩 모델 클래스를 활용할 수 있습니다.
* **벡터 저장소 구성:** 기본 구현으로 **Chroma** 벡터 DB를 사용합니다. LangChain의 `Chroma` wrapper를 통해 쉽게 벡터스토어 인스턴스를 생성하고, 앞서 얻은 임베딩과 문서 청크들을 삽입하여 **유사도 검색 인덱스**를 만듭니다.
* **검색기 구성:** 구축된 벡터스토어로부터 `vectorestore.as_retriever(search_type="mmr", k=SEARCH_K)` 등을 호출하여 \*\*문서 검색기(retriever)\*\*를 준비합니다. MMR(Maximal Marginal Relevance) 등의 기법을 사용해 다양하고 관련도 높은 결과를 얻도록 설정합니다.
* **벡터스토어 캐시/로드:** `is_clean_vectorstore` 옵션이 꺼져 있는 경우, 이미 저장된 벡터 DB가 존재하면 불러오고 재사용할 수 있습니다. 설정에 따라 벡터스토어를 갱신(rebuild)할 수도 있습니다. 이 기능은 대용량 문서 집합에 대해 초기 로딩 비용을 줄이고자 설계되었습니다.

### **pipeline/querying/rag\_chain\_builder.py** – RAG 체인 빌더 (기본 QA 체인 구성)

**RAGChainBuilder** 클래스는 LangChain의 표준 체인을 활용하여 **질의응답 체인**을 생성하는 빌더입니다. 주요 특징과 메서드:

* **설계 개요:** 빌더 패턴을 적용하여 복잡한 체인 초기화 과정을 캡슐화하였습니다. LangChain에서 제공하는 **ConversationalRetrievalChain**의 구조를 바탕으로, LLM, 프롬프트, 메모리, 검색기를 조합해 사용자 질문에 답변하는 체인을 구성합니다【23†】.
* **LLM 초기화:** `_create_llm()` 내부에서 **LLMFactory**를 사용하여 `self.llm`을 생성합니다. 현재 `"ollama"`(로컬 Llama2 등 모델)와 `"claude"`(Anthropic Claude API)를 지원하며, `LLM_TYPE`과 `MODEL_NAME` 설정에 따라 해당 LLM wrapper를 가져옵니다. 이 함수는 팩토리 패턴을 통해 LLM 생성 방식을 추상화하여, 필요한 경우 쉽게 다른 모델을 추가할 수 있게 합니다【15†】.
* **메모리 구성:** 체인은 대화 메모리로 **SummarizingMemory**를 사용합니다. `_get_session_history(session_id)` 메서드는 주어진 세션 ID에 대한 SummarizingMemory 인스턴스를 생성하거나 기존 것을 반환합니다. 세션별로 별도 `session_histories` 딕셔너리에 저장하여 여러 사용자의 대화를 구분합니다. SummarizingMemory는 LangChain의 `BaseChatMessageHistory`를 상속하여 구현되었으며, 메모리에 메시지를 추가할 때 자동으로 **과거 메시지를 요약**해주는 특별한 메모리입니다 (아래 SummarizingMemory 섹션 참조).
* **프롬프트 템플릿 구성:** `_create_prompt_templates()`에서 대화 체인에 필요한 **Prompt**들을 설정합니다. 예를 들어:

  * `get_condense_prompt()`: 대화의 히스토리와 사용자 질문을 받아 **질문을 재구성**하기 위한 프롬프트. (대화 내용을 고려하여 follow-up 질문을 독립적인 질문으로 바꿀 때 사용 가능)
  * `get_qa_prompt()`: 실제 답변 생성시 사용하는 **QA 시스템 메시지** 템플릿. 사용자 프로필에 특화된 시스템 역할 지시와 함께, `{chat_history}` (요약+최근 대화)와 `{context}` (검색된 문서 컨텍스트)를 채워 넣어 답변하도록 설계됩니다.
* **체인 빌드/실행:** 구성된 LLM, 프롬프트, 메모리를 활용해 LangChain의 QA 체인을 초기화합니다. 간단한 구현으로는 `ConversationalRetrievalChain.from_llm`에 `llm`, `retriever`, `memory`를 넣어 생성할 수 있습니다. 현재 코드에서는 GraphRAGChainBuilder를 도입 중이므로, RAGChainBuilder의 `run(question, session_id)`는 주로 **GraphRAGChainBuilder** 쪽으로 대체되어가고 있습니다. 하지만 기본적으로:

  * `run()` 호출 시 주어진 `session_id`로부터 SummarizingMemory를 가져와 (필요시 생성), LLM을 통해 질의를 처리하고 답변을 반환합니다.
  * 메서드 내부에서 검색기(retriever)를 사용해 관련 문맥을 찾고, ChatPromptTemplate에 메모리의 요약+최근 대화와 검색 결과를 채워 LLM에게 전달함으로써 최종 답변을 생성합니다.
* **메모리 관리:** `reset_memory(session_id)` 메서드를 제공하여 특정 세션 또는 모든 세션의 대화 메모리를 비울 수 있습니다. `"all"`을 인자로 받으면 내부의 `session_histories` 딕셔너리를 초기화하여 **모든** SummarizingMemory를 리셋하고, 특정 세션 ID가 주어지면 해당 세션의 SummarizingMemory를 새로 교체합니다.

### **pipeline/querying/summarizing\_memory.py** – SummarizingMemory (대화 요약 메모리)

**SummarizingMemory** 클래스는 사용자와 AI의 대화 기록을 관리하며, 오래된 내용을 자동으로 요약해주는 메모리입니다. 이것은 LangChain의 `BaseChatMessageHistory`를 상속하고 Pydantic `BaseModel`로 정의되었습니다. 주요 동작:

* **메시지 저장:** `.add_message` 또는 `.add_messages` 메서드로 새 사용자/AI 메시지를 추가하면 내부 리스트에 쌓입니다. 추가 후 `_maybe_summarize()`를 호출하여 요약 여부를 판단합니다.
* **자동 요약 트리거:** `max_recent_turns` (최근 유지할 턴의 개수, 예: 3) 속성을 기준으로, **저장된 메시지 수가 `max_recent_turns * 2` (user+AI 쌍 기준)보다 많아지면** 오래된 대화를 요약하게 됩니다.

  * 요약 대상인 이전 메시지 (`to_summarize`)와 최근 메시지 (`recent`)를 분리하고, `_format_history(to_summarize)`를 통해 과거 대화를 하나의 텍스트로 포맷팅합니다.
  * `prompts.get_summary_prompt()`로 미리 정의된 요약 지시 프롬프트를 가져와, 해당 `chat_history` 자리에 이 포맷된 과거 대화를 채워넣습니다.
  * `self.llm.invoke(prompt_messages)`를 통해 현재 LLM으로 **요약문 생성**을 시도합니다. (SummarizingMemory는 체인 빌더에서 생성 시 LLM을 주입받아 `self.llm`으로 보관합니다.)
  * 성공하면 `self.summary` 속성에 요약문을 저장하고, 로그를 남깁니다. 이후 `self.messages` 리스트를 `recent` (최근 몇 턴만 남긴 리스트)로 교체하여 메시지 수를 줄입니다. 요약된 내용은 `self.summary`에 남아 필요시 참조할 수 있습니다.
* **요약 활용:** SummarizingMemory는 요약된 과거를 별도로 저장함으로써, **컨텍스트가 길어지면 이전 대화를 압축 저장**하고, **최신 대화 위주로 LLM에게 전달**될 수 있게 합니다. 체인이 프롬프트를 만들 때 이 메모리의 요약과 최근 메시지를 결합하여 `{chat_history}`로 사용할 수 있습니다. 예를 들어 `get_qa_prompt` 템플릿에서는 대화 이력을 넣는 부분에 SummarizingMemory가 관리하는 (요약 + 최근 대화) 텍스트가 들어가도록 의도되어 있습니다. 이를 통해 대화 맥락은 유지하면서도 불필요하게 토큰을 소모하지 않도록 최적화했습니다.
* **요약 실패 처리:** LLM 호출이 실패하거나 예외가 발생해도 요약 과정의 오류는 catch되어 경고 로그를 남기고, 요약이 안 된 상태로 계속 진행합니다 (다음 턴에 재시도 가능).

### **pipeline/querying/prompts.py** – 프롬프트 템플릿 모음

이 모듈은 체인에서 사용하는 여러 **프롬프트 템플릿**을 정의합니다:

* `get_summary_check_prompt()`: 사용자 질문이 요약된 정보를 요구하는지 (`YES/NO`) 판단하는 지시 프롬프트【7†】. 미리 정의된 예시 (“이력 요약...”, “프로젝트 알려줘...” 등)이 제시되어 있으며, 모델에게 질문이 이러한 요약 요청에 해당하는지 판단하도록 합니다.
* `get_summary_type_prompt()`: 요약 질문으로 판별된 경우, 어떤 **요약 유형**인지 분류하는 프롬프트【9†】. 가능한 카테고리로 *resume* (이력서 요약), *projects* (주요 프로젝트), *workstyle* (업무 스타일), *all* (전체 요약), *none* (해당없음)을 정의하고, 모델이 사용자 질문을 이 중 하나로 분류하게 합니다.
* `get_condense_prompt()`: 대화 맥락과 follow-up 질문을 하나의 자체 포함 질문으로 만들어주는 프롬프트【5†】. (예: “그는 어떤 일을 했나요?” 같은 추가 질문을 이전 대화와 합쳐 “나현석님의 경력에서 어떤 일을 했는지 알려주세요.”로 재구성).
* `get_qa_prompt()`: 최종 답변 생성을 위한 프롬프트 템플릿【11†】. 시스템 메시지로서 AI 비서의 역할과 답변 조건 (한국어로 답변하되, 전문용어는 영어 허용 등)을 명시하고, 자리표시자 `{chat_history}`와 `{context}`를 포함합니다. 체인 실행 시 `{chat_history}`에는 SummarizingMemory로부터 온 대화 히스토리(필요하면 요약 포함)가, `{context}`에는 벡터스토어에서 검색된 관련 문서 내용이 채워져 LLM에게 전달됩니다.

### **pipeline/querying/llm\_factory.py** – LLM 팩토리 클래스

LLMFactory는 다양한 **대형언어모델(LLM)** 인스턴스를 생성하는 정적 메서드를 제공합니다. 구현 세부사항:

* 지원하는 LLM 타입: `"ollama"`와 `"claude"`를 기본 지원합니다.

  * `"ollama"`: 로컬에서 구동되는 Llama 계열 모델 (예: Ollama 서버를 통해 Llama2 등 실행)
  * `"claude"`: Anthropic Claude API를 통한 언어모델
* `create_llm(llm_type, model_name, **kwargs) -> BaseLLM`: 주어진 타입 문자열에 따라 해당 LLM의 LangChain 호환 래퍼 객체를 생성합니다. 예를 들어 ollama의 경우 `langchain_community.llms.Ollama` 클래스를 호출하고, claude의 경우 `langchain_anthropic.Chat` 클래스를 사용합니다【13†】【16†】. 지원하지 않는 타입이면 `ValueError`를 발생시킵니다.
* 내부적으로 `_create_ollama`, `_create_claude` 같은 헬퍼를 둘 수 있으며, 필요한 경우 새로운 모델 타입 (예: OpenAI GPT)도 쉽게 추가하도록 설계됐습니다.
* 이렇게 팩토리로 LLM을 추상화함으로써, 체인 빌더 코드가 특정 LLM 클래스에 종속되지 않고 `LLM_TYPE` 설정만 바꾸면 다른 모델로 교체할 수 있습니다.

### **pipeline/querying/graph\_builder.py** – GraphRAGChainBuilder (LangGraph 기반 체인 빌더)

이 클래스는 **LangGraph**를 활용한 **그래프 기반 RAG 체인 빌더**로, 향후 시스템의 질의 처리 흐름을 확장하기 위한 구조를 담고 있습니다. 현재는 실험적/확장적 요소이며, RAGChainBuilder와 유사한 역할을 하지만 **질문 분기 논리를 명시적인 상태 그래프**으로 표현합니다. 주요 특징:

* **LangGraph 통합:** LangGraph는 대화 흐름을 **State Graph**로 표현하는 프레임워크입니다. GraphRAGChainBuilder는 `langgraph.graph.StateGraph`와 `RunnableLambda` 등을 사용하여 질의응답 과정을 Node(상태)들의 그래프로 정의합니다.
* **질문 처리 흐름:** 클래스의 주석에 명시된 바와 같이, LangGraph를 통해 다음과 같은 **분기 로직**을 수행합니다:

  1. **요약 질문 여부 판별:** 첫 번째 노드에서 `get_summary_check_prompt`를 LLM에 적용하여 입력 질문이 요약을 요구하는지 판단합니다 (“YES” 또는 “NO”).
  2. **요약 타입 분기:** 만약 요약 질문이라면, 다음 단계에서 `get_summary_type_prompt`로 질문의 구체적 요약 유형(resume, projects, workstyle, all)을 얻습니다. 이를 통해 이후 검색에 어떤 **카테고리**의 정보를 사용할지 결정합니다.
  3. **문서 검색:** 요약 유형이 결정되면 해당 유형에 맞는 문서만 추출하거나, 또는 일반 질문의 경우에는 전체 문서 대상의 **MMR 기반 유사도 검색**을 수행합니다. 이 때 VectorStore (예: Chroma)에서 `search_k`개 만큼 연관도가 높은 청크를 가져옵니다. 요약 질문인 경우 `metadata['summary_type']` 같은 필터를 적용해 특정 종류의 정보만 검색하도록 설계되었습니다 (예: 이력서 요약은 이력 관련 청크만).
  4. **답변 생성:** 마지막으로 QA 프롬프트 (`get_qa_prompt`)에 대화 히스토리와 검색된 문서 컨텍스트를 넣어 LLM으로 답변 생성 노드를 실행합니다. 생성된 답변은 최종 출력으로 사용자에게 전달됩니다. 대화 메모리는 SummarizingMemory를 통해 업데이트되고, LangGraph 상에서 다음 질문을 받을 준비가 됩니다.
* **상태 그래프 구성:** `GraphRAGChainBuilder.__init__`에서 LangGraph로 질의 처리용 StateGraph를 초기화하고, 위의 단계들을 노드로 등록하여 `self.runnable` 그래프를 빌드합니다. 각 노드는 프롬프트 수행이나 검색 등을 람다 형태로 감싸 Runnable로 등록되며, 간선 조건에 따라 분기가 일어납니다. 예를 들어, 요약 판단 노드의 출력이 "YES"일 때만 요약 타입 노드로 이어지고, "NO"일 경우 바로 일반 검색 노드로 건너뜁니다.
* **실행 (`run`) 메서드:** `GraphRAGChainBuilder.run(question, session_id)`를 호출하면 내부적으로 `self.runnable.invoke({"question": ..., "session_id": ...})`를 수행하여, 정의된 상태 그래프에 따라 일련의 처리가 자동 진행됩니다. 로그에는 각 단계의 결과(요약 여부, 타입, 검색 수행 등)를 남기며, **LangGraph**를 통해 보다 구조화된 멀티스텝 체인이 구현됩니다.
* **현재 상태:** 이 그래프 기반 체인은 향후 **LangChain Hub**의 LangGraph 통합이나 복잡한 대화 흐름 지원을 위해 계획된 것입니다. 아직 완전히 메인 흐름에 적용되지 않았지만, `rag_pipeline`에서 `GraphRAGChainBuilder`를 우선적으로 생성하도록 해 두어, 점진적으로 이 방식으로 전환할 수 있게 해 놓았습니다. LangGraph 도입으로 코드의 **가독성과 확장성**이 높아지며, 새로운 분기나 추가 처리 로직을 쉽게 그래프 노드로 표현할 수 있을 것으로 기대됩니다.

### **adapters/** – 문서 어댑터 패키지

다양한 형식의 문서를 일관된 인터페이스로 불러오기 위해 **Adapter 패턴**이 적용되었습니다:

* `adapters/base/`: DocumentAdapter 기반 클래스와, 문서 내 **피쳐 추출**이나 **타입 정의**를 위한 클래스들이 정의되어 있습니다. `DocumentAdapter`는 모든 문서 어댑터의 추상 기반으로, `load()` 메서드 등을 갖습니다.
* `adapters/document/`: 실제 문서 형식별 어댑터 구현이 위치합니다.

  * **pdf.py**: PyMuPDF (`fitz`)를 사용하여 PDF 파일을 읽고 텍스트를 추출하는 `PDFAdapter`와 `PDFExtractor`가 구현되어 있습니다. `PDFExtractor`는 페이지별로 텍스트를 추출하고, `DocumentFeatureProcessor`를 상속받아 추가 전처리를 할 수 있게 합니다【30†】.
  * **text.py**: 일반 텍스트 파일을 처리하는 `TextAdapter`와 `TextExtractor`가 정의되어 있습니다. 단순히 파일 전체를 읽고 필요하면 정제합니다.
* `doc_factory.py`: 주어진 파일 경로의 확장자를 보고 적절한 Adapter 객체를 만들어주는 **팩토리 함수** (`get_document_proc`)가 있습니다. 예를 들어 `.pdf`인 경우 `PDFAdapter`를 생성하며, 이 때 내부적으로 `PDFExtractor`를 Runner로 감싸 비동기/병렬 처리를 지원하도록 구성합니다【34†】. 이 팩토리를 통해 DocumentLoader는 구체적인 파일 처리 로직을 몰라도 되며, 새로운 문서 타입을 추가하려면 어댑터만 작성하면 됩니다.

### **utils/** – 유틸리티 모듈

* **constants.py**: 시스템에서 사용하는 상수들을 정의합니다. 예: 초기 안내 대화 `WELCOME_MESSAGES` (UI에서 처음 채팅창에 나오는 튜플 리스트), `SEARCH_K`, `MAX_RECENT_TURNS` 등이 들어있습니다. (`MAX_RECENT_TURNS` 등 몇몇 상수는 settings.py에서 정의되었습니다.)
* **file\_io.py**: 파일 열기/저장 등의 I/O 도우미 함수.
* **text\_preproc.py**: 불필요한 공백 제거, 맞춤법 교정, Ollama 모델 대응을 위한 특수 후처리 등 **텍스트 전처리 함수**들이 있습니다. (예: `improve_text`, `ollama_spacing`)
* **runner.py**: 멀티스레딩 등으로 문서 처리를 가속하기 위한 Runner 클래스가 있습니다. Adapter에서 `Runner.wrap()`을 사용하여 무거운 작업을 비동기로 돌리는 데 활용됩니다.
* 이외에 특정 기능에 사용되는 유틸 함수들이 이 패키지에 모여 있습니다.

## 시스템 구성 및 동작 흐름 (Architecture & Workflow)

이 섹션에서는 전체 시스템의 아키텍처와 질의응답 처리 흐름을 요약합니다.

* **데이터 준비 단계:** 우선 원본 문서들이 `data/raw/` 폴더 등에 저장되어 있다고 가정합니다. 이는 사용자의 이력서, 프로젝트 소개서, 업무 스타일 문서 등이며, PDF나 TXT 형식일 수 있습니다.

* **파이프라인 초기화:** 애플리케이션이 시작되면 (CLI 또는 UI를 통해) **RAGPipeline**을 생성하고 `setup_chain()`을 호출합니다. 이 과정에서:

  1. **문서 Ingestion:** DocumentLoader가 모든 문서를 읽어 들여 텍스트를 추출하고, Document 리스트를 만듭니다.
  2. **텍스트 Chunk 분할:** 긴 문서는 설정된 크기로 분할하여 정보 검색 단위를 세분화합니다.
  3. **벡터 임베딩 & 저장:** VectorStoreBuilder가 각 청크별 임베딩을 생성하고 벡터스토어(DB)에 저장합니다. 이후 유사도 검색을 빠르게 수행할 수 있게 됩니다.
  4. **질문응답 체인 구성:** GraphRAGChainBuilder (또는 RAGChainBuilder)가 선택된 LLM 및 설정을 바탕으로 QA 체인을 초기화합니다. SummarizingMemory를 세션별로 준비하고, 필요한 프롬프트 템플릿을 로딩합니다.

  * 🔧 *LangGraph 활성화:* 현재 `GraphRAGChainBuilder`를 사용하도록 되어 있어, LangGraph 기반으로 질의 처리 흐름을 준비합니다 (요약 질문 분기 로직 포함).

* **사용자 질의 처리:** 사용자가 질문을 입력하면, \*\*체인 빌더의 `run(query, session_id)`\*\*가 호출되어 다음과 같은 **분기 흐름**으로 진행됩니다:

  1. **요약 질문 여부 판정:** 우선 시스템이 **질문 유형을 분류**합니다. 내부적으로 미리 정의된 프롬프트를 통해 "이 질문이 요약된 형태의 정보를 요구하는가?"를 판단합니다.

     * *예:* “이력 요약을 알려주세요”, “핵심 프로젝트가 뭐가 있나요?” 등의 질문은 **요약형 질문**으로 분류됩니다.
     * 그런 의도가 아닌 경우 일반 질문으로 간주됩니다.

  2. **요약형 질문 처리:**

     * 요약형으로 판별되면, 추가로 **어떤 요약을 원하는지**를 분류합니다 (이력서, 프로젝트, 업무스타일, 전체 등). 이 결과에 따라 해당 카테고리에 맞는 정보를 준비합니다.
     * **문맥 검색:** 시스템은 벡터스토어에서 **해당 요약 카테고리에 해당하는 문서 조각들만** 검색합니다. 예를 들어 “이력 요약” 질문이라면, 이력서 관련 청크들만 필터링하여 검색 결과로 가져옵니다. 이러한 메타데이터 필터링을 통해 불필요한 정보 없이 정확한 요약 컨텍스트를 수집합니다.
     * 최종적으로 선택된 요약 정보 조각들은 답변 생성 단계의 컨텍스트로 활용됩니다.

  3. **일반 질문 처리:**

     * 요약형이 아니라 일반 정보 질의인 경우, **전체 문서**를 대상으로 유사도 검색을 수행합니다. 벡터스토어로부터 질문과 가장 관련 높은 `k`개의 청크를 MMR 방식으로 가져와 컨텍스트로 삼습니다.
     * 이렇게 하면 사용자의 구체적인 질문에 대해 문서 전체에서 필요한 내용만 뽑아와 답변에 활용할 수 있습니다.

  4. **답변 생성 (LLM 호출):**

     * 검색 단계에서 얻은 컨텍스트를 바탕으로, **최종 프롬프트**가 구성됩니다. 여기에는 시스템 역할 지시 + 대화 히스토리(요약 포함) + 검색 문맥 + 사용자 질문 등이 포함됩니다.
     * 준비된 LLM(예: Ollama의 Llama2 모델 혹은 Claude API)이 이 프롬프트를 받아 **답변을 생성**합니다.
     * 생성된 답변 텍스트는 최종 결과로 사용자에게 반환됩니다.

  5. **대화 메모리 업데이트:**

     * 반환된 답변은 SummarizingMemory를 통해 해당 세션의 히스토리에 추가됩니다.
     * SummarizingMemory는 직전 단계에서 사용자 질문과 AI 답변을 저장하면서, 설정된 최근 턴 수를 초과한 이전 대화가 있다면 즉시 **요약 작업을 수행**합니다. 오래된 대화 내용을 간결한 요약문으로 바꾸고, 최신 대화와 요약문을 함께 보존함으로써 이후 질의에서도 맥락을 유지합니다.
     * 이 과정은 사용자에게 투명하게 일어나며, 다음 질문 처리 시 자동으로 반영됩니다. (예: 대화가 길어지면 SummarizingMemory가 “... 이전 대화 요약: \[요약문] ...”를 내부적으로 갖고, `{chat_history}`에 사용)

* **반복 처리:** 이후 사용자 질의가 연속해서 들어오면 위 과정을 반복합니다. SummarizingMemory로 요약된 과거 히스토리 + 최신 대화를 함께 고려하여 질의를 이해하고, 항상 관련 문서를 찾아 답을 구성합니다. 사용자는 긴 대화 흐름 속에서도 시스템이 중요한 맥락을 기억하고 응답한다고 느끼게 됩니다.

* **(재)초기화 기능:** 필요 시 벡터스토어를 재구축하거나 (예: 새로운 문서를 추가한 경우 “벡터 저장소 초기화” 버튼), 대화 메모리를 리셋하여 새 대화를 시작할 수 있습니다. 이러한 관리 기능은 UI와 CLI를 통해 제공되며, 내부적으로 RAGPipeline이나 체인 빌더의 초기화 메서드를 다시 호출합니다.

## LangChain 활용 및 LangGraph 확장 계획

이 프로젝트 전반에 **LangChain** 프레임워크의 여러 추상화가 활용되었습니다:

* **Document**와 **VectorStore**: 문서 표현과 벡터 DB 검색에 LangChain의 표준 인터페이스를 사용하여 구현했습니다. DocumentLoader 및 VectorStoreBuilder는 LangChain의 문서 처리 및 벡터 저장소 API를 적극 활용하여, 최소한의 코드로 강력한 RAG 기반을 구축합니다.
* **LLM 및 Chains**: LangChain을 통해 다양한 LLM을 손쉽게 교체할 수 있도록 구성했습니다. ConversationalRetrievalChain 개념을 변형하여 SummarizingMemory와 커스텀 프롬프트를 넣는 등, LangChain의 체인 구성 요소들을 조합해 QA 체인을 만들었습니다. 또한, LangChain이 제공하는 `ChatPromptTemplate`, `BaseChatMessageHistory` 등을 사용하여 **프롬프트 엔지니어링**과 **메모리 관리**를 구현했습니다.
* **Memory**: SummarizingMemory는 LangChain의 대화 메모리 인터페이스를 구현한 커스텀 클래스입니다. LangChain의 기본 ConversationBufferMemory와 SummaryMemory 아이디어를 결합하여, 실제 프로젝트 요구에 맞게 튜닝한 것입니다. 이를 통해 긴 문맥도 안정적으로 유지하면서 성능을 유지합니다.
* **LangGraph 통합**: 앞으로의 계획으로, 현재 도입 중인 **LangGraph**를 완전히 활용할 예정입니다. LangGraph는 LangChain과 상호보완적으로 동작하며, 체인 로직을 **그래프 형태로 시각화하고 관리**할 수 있게 해줍니다. GraphRAGChainBuilder에 일부 적용된 것처럼, 요약 질문 분기나 멀티스텝 질의응답 흐름을 명시적으로 표현하고 관리하기 쉽습니다.

  * 예를 들어, 추가로 **질문 재구성 단계**(condense prompt 사용)나, **후처리 단계**(응답 검열 또는 포맷 조정)를 넣고 싶다면, 그래프 노드를 추가하여 흐름을 확장하면 됩니다. LangGraph를 쓰면 이러한 변경이 코드 구조에 잘 드러나고 재사용도 용이해집니다.
  * 현재 GraphRAGChainBuilder는 LangGraph 기반의 프로토타입으로 존재하며, 내부적으로 `StateGraph`와 `Runnable`을 사용하고 있습니다. 향후 안정화되면 RAGPipeline에서 기존 RAGChainBuilder를 완전히 대체하여, **복잡한 질의응답 시나리오** (예: 질의 유형이 더 다양해지거나, 외부 API 호출과 결합 등)도 유연하게 다룰 수 있을 것으로 예상합니다.

## 간략한 시스템 아키텍처 도식 (Text-based Architecture Diagram)

```plaintext
사용자 질문 입력 
   │
   ▼
[ UI/CLI 인터페이스 ]  ←─―─―─┐ 
(Gradio 웹 또는 CLI)        │ 대화 초기화/재설정 등 사용자 명령 처리
   │                        │ 
   ▼                        │ 
[ RAGPipeline ] (파이프라인 초기화 및 구성)
   ├─ DocumentLoader: 문서 읽기 & 텍스트 추출
   ├─ 문서 청크 분할 (Chunk Size, Overlap 적용)
   ├─ VectorStoreBuilder: 임베딩 계산 & 벡터DB 색인
   └─ RAGChainBuilder / GraphRAGChainBuilder: QA 체인 준비 (LLM, Memory, Prompts 결합)
           │
           │ (질문 처리 실행 `run(query)`)
           ▼
           ┌────────────────────────────────────────┐
           │      **질문 처리 흐름 (요약 분기)**      │
           │ 1. SummarizingMemory에서 대화 히스토리 확보   │
           │ 2. 요약 질문인지 판별 (프롬프트 분류)         │
           │ 3. ┌─ YES: 요약 유형 파악 (resume/projects 등)│
           │ 3. │        관련 문서만 벡터 DB에서 검색       │
           │ 3. └─ NO: 일반 질문으로 전체 문서 검색        │
           │ 4. 검색된 문맥 + (요약된)대화히스토리로 LLM 프롬프트 생성  │
           │ 5. LLM (예: Llama2 또는 Claude) 호출 → 답변 생성 │
           └────────────────────────────────────────┘
           │
   ▼       ▼
SummarizingMemory 갱신      생성된 답변 반환
(오래된 대화는 요약 저장)    (사용자에게 출력)
```

위 도식은 시스템의 주요 구성 요소와 데이터 흐름을 텍스트로 나타낸 것입니다. 사용자 질문이 들어오면, UI/CLI를 통해 RAGPipeline의 체인이 실행되고, 내부에서 요약 질문 여부에 따른 분기, 문서 검색, 답변 생성, 메모리 업데이트가 이루어집니다. 이러한 일련의 과정을 통해 **요약 + RAG 기반의 대화형 QA 시스템**이 완성됩니다.


이 README에서는 제공된 코드베이스에 기반하여, **요약 분기형 RAG 질의응답 시스템**의 전체 구조와 구성 요소를 설명했습니다. 각 파일의 역할과 클래스/함수의 기능을 살펴보았고, 시스템이 동작하는 방식을 단계별로 기술했습니다. 기술 스택으로 **LangChain**을 활용해 문서 임베딩, LLM 체인, 메모리 관리 등을 구현하였고, 향후 **LangGraph**를 통해 더욱 복잡한 대화 흐름도 체계적으로 관리할 계획임을 명시했습니다. 

