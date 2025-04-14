from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import json
import sys
import tiktoken
from typing import AsyncIterator, Dict, Any, List
from datetime import datetime
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import asyncio

# .env 파일 로드
load_dotenv()

# FastAPI 애플리케이션 생성 및 static, templates 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 로그 파일 설정
os.makedirs("logs", exist_ok=True)
log_file_path = os.path.join("logs", "token.log")

# 로그 파일 핸들러 생성
def get_log_file():
    return open(log_file_path, 'a', encoding='utf-8')

# 콘솔 로깅 핸들러 생성
console_handler = StreamingStdOutCallbackHandler()

# OpenAI API 키 설정 (환경 변수에서 가져옴)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Vector DB 경로 지정
os.makedirs("vectorstore", exist_ok=True)

# 마크다운 파일 불러오기 및 처리
loader = TextLoader("Job_data.md", encoding="utf-8")
documents = loader.load()

# 마크다운 직업 기준 분할
headers_to_split_on = [
    ("#", "직업")
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)
# 텍스트 추출 및 분할
docs = []
for doc in documents:
    splits = markdown_splitter.split_text(doc.page_content)
    for split in splits:
        docs.append(split)

# 임베딩 모델 생성 및 벡터 DB 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = FAISS.from_documents(docs, embedding_model)

# 임베딩 저장
vectorstore.save_local("vectorstore/faiss_index")

# 불러올 때
if os.path.exists("vectorstore/faiss_index"):
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    # 임베딩 생성 코드
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    vectorstore.save_local("vectorstore/faiss_index")

retriever = vectorstore.as_retriever()

# LLM 모델 설정
llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4o")

# 토큰 비용 설정
GPT4_INPUT_COST_PER_1M_TOKENS = 2.5  # GPT 4o 토큰 비용, 100만 토큰 당 2.5달러
GPT4_OUTPUT_COST_PER_1M_TOKENS = 10.0  # GPT 4o 토큰 비용, 100만 토큰 당 10달러

class TokenUsageLogger(BaseCallbackHandler):
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.input_tokens = 0
        self.output_tokens = 0
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        # 입력 토큰 수 초기화
        self.output_tokens = 0
        # 입력 토큰 수 계산
        self.input_tokens = sum(len(self.encoding.encode(p)) for p in prompts)
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.output_tokens += len(self.encoding.encode(token))
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLM 응답이 완료되면 토큰 사용량과 비용을 로깅"""
        try:
            # 비용 계산
            input_cost = (self.input_tokens / 1_000_000) * GPT4_INPUT_COST_PER_1M_TOKENS
            output_cost = (self.output_tokens / 1_000_000) * GPT4_OUTPUT_COST_PER_1M_TOKENS
            total_cost = input_cost + output_cost
            
            # 로그 작성
            log_entry = (
                f"\n{'='*50}\n"
                f"시간: {datetime.now().isoformat()}\n"
                f"토큰:\n"
                f"  - 입력 토큰: {self.input_tokens}\n"
                f"  - 답변 토큰: {self.output_tokens}\n"
                f"  - 토큰 사용량 : {self.input_tokens + self.output_tokens}\n"
                f"비용:\n"
                f"  - 입력 토큰 비용 : ${input_cost:.6f}\n"
                f"  - 출력 토큰 비용 : ${output_cost:.6f}\n"
                f"  - 총 토큰 비용 : ${total_cost:.6f}\n"
                f"{'='*50}\n"
            )
            
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"로깅 중 오류 발생: {str(e)}")

# RIASEC 유형 설명
RIASEC_TYPE_DESCRIPTIONS = {
    "R": """
    현실형(R)
                        성향 특징
                            - 사물, 도구, 기계를 다루는 실제적인 활동을 선호함
                            - 현장에서의 신체 활동과 작업을 즐김
                            - 계획적이고 안정적인 환경에서 일하는 것을 좋아함
                            - 독립적으로 일하는 것을 선호하며, 협업보다는 개인 작업을 선호함
                            - 자연, 동식물, 도구, 기계 등과 직접 상호작용하는 것을 좋아함
                        주요 성격
                            - 악착같고 실용적인 성향
                            - 계획적이며 체계적인 사고방식
                            - 독립적이고 신체활동에 강함
                            - 감성보다는 사실과 실행 위주의 사고
                            - 반복적이고 구체적인 작업을 잘 수행함
    """,
    "I": """
    탐구형(I)
성향 특징
                        - 아이디어와 데이터를 다루는 지적 활동을 선호함
                        - 과학적, 기술적 문제 해결에 흥미가 많음
                        - 실험, 분석, 연구 등 추상적이고 논리적인 사고를 즐김
                        - 리더십보다는 독립적으로 문제를 해결하는 것에 강함
                        - 설득보다는 탐구와 이해 중심의 사고 성향

                        주요 성격
                        - 호기심이 많고 지적이며 내성적임
                        - 분석적이고 정교하며 논리적인 사고에 강함
                        - 학문적 성향, 도전정신 강함
                        - 복잡한 문제를 해결하거나 새로운 아이디어를 탐구하는 것을 즐김
    """,
    "A": """
    예술형(A)
                        성향 특징
                            - 사람, 아이디어, 사물과 함께 창의적으로 일하는 것을 선호함
                            - 상상력과 독창성을 발휘하는 활동을 좋아함
                            - 예상 가능한 환경 속에서 융통성 있게 일하는 것을 선호함
                            - 자기표현, 예술적 감각, 감정의 표현에 민감함
                        주요 성격
                            - 개방적이고 상상력이 풍부함
                            - 직관적이고 정서적이며 독립적인 성향
                            - 충동적이고 예술적 감각이 뛰어남
                            - 표현욕이 강하고 감각적으로 섬세함
    """,
    "S": """
    사회형(S)
                        성향 특징
                        - 사람들과 함께 일하고, 돕고, 가르치는 활동을 선호함
                        - 치료, 조언, 교육과 관련된 직무에 흥미가 많음
                        - 타인을 직접적으로 만나서 상호작용하는 것을 좋아함
                        - 이해심 많고, 참을성 있으며, 관대한 성격
                        주요 성격
                        - 친절하고 정력적이며 책임감이 강함
                        - 협동적이고 설득력이 있으며, 통찰력 있는 성향
                        - 감정적으로 안정되고 다른 사람의 입장을 잘 이해함
                        - 집단 내에서 잘 협력하고, 리더십을 발휘함
    """,
    "E": """
    기업형(E)
                        성향 특징
                        - 사람들과 함께 일하며 설득하거나 주도하는 활동을 선호함
                        - 자신감 있고 목표 지향적인 업무에 흥미를 가짐
                        - 사업적 모험과 리더십을 발휘할 수 있는 환경을 선호함
                        - 판매, 정치, 비즈니스, 리더십 활동에 흥미가 많음
                        주요 성격
                        - 자신감 있고 사교적이며 활동적
                        - 충동적이면서도 자발적인 성향
                        - 언변이 뛰어나고 리더십이 강함
                        - 타인을 설득하고 조직을 주도하는 능력이 있음
    """,
    "C": """
    관습형(C)
                        성향 특징
                        - 자료, 사물, 사람과 함께 일하는 것을 선호함
                        - 기록, 계산, 문서 작업 등 구조화된 활동을 좋아함
                        - 숫자, 컴퓨터, 기계 등을 다루는 실무형 작업에 강함
                        - 예측 가능하고 체계적인 업무 환경을 선호함

                        주요 성격
                        - 조직적이고 효율적이며 정확성이 뛰어남
                        - 집중력과 책임감이 강하고 체계적인 사고방식 보유
                        - 구조화된 규칙, 절차에 잘 따르고 순응적임
                        - 세부사항을 꼼꼼하게 챙기며 실수를 최소화함
    """
}

# RIASEC 결과 해석 함수
async def interpret_survey_results(survey_results, websocket: WebSocket):
    try:
        # JSON 문자열 파싱
        survey_data = json.loads(survey_results)
        scores = survey_data.get("scores", {})
        
        # RIASEC 점수 추출
        r_score = scores.get("R", 0)
        i_score = scores.get("I", 0)
        a_score = scores.get("A", 0)
        s_score = scores.get("S", 0)
        e_score = scores.get("E", 0)
        c_score = scores.get("C", 0)
        
        # 스트리밍과 로깅을 위한 콜백 핸들러 설정
        stream_handler = AsyncIteratorCallbackHandler()
        token_logger = TokenUsageLogger(log_file_path)
        
        survey_interpreter = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4o",
            streaming=True,
            callbacks=[stream_handler, token_logger]
        )
        
        messages = [
                    SystemMessage(content="""
                    너는 고등학생인 나의 성격과 흥미를 분석해주는 전문가야.  
                    내가 입력한 흥미 유형 점수를 바탕으로, 나의 성격과 행동 성향을 친구처럼 편하게, 짧고 따뜻하게 설명해줘.

                    각 유형은 아래 설명을 참고해서 해석에 반영해 줘:

                    {RIASEC_TYPE_DESCRIPTIONS['R']}
                    {RIASEC_TYPE_DESCRIPTIONS['I']}
                    {RIASEC_TYPE_DESCRIPTIONS['A']}
                    {RIASEC_TYPE_DESCRIPTIONS['S']}
                    {RIASEC_TYPE_DESCRIPTIONS['E']}
                    {RIASEC_TYPE_DESCRIPTIONS['C']}

                    점수 분포를 확인한 뒤, 다음 원칙을 따라 분석해줘:
                    1. 점수가 두드러지게 높은 1~2개가 있다면, 그 유형 중심으로 해석해줘.
                    2. 점수 차이가 크지 않으면, 다양한 성향이 골고루 섞인 균형잡힌 스타일로 해석해줘.
                    3. 단정적이지 않게, 말투는 부드럽고 친구처럼 친근하게 해줘.
                    4. 어려운 단어나 이론 용어는 쓰지 말고, 최대한 자연스럽게 설명해줘.
                    5. 출력은 200자 이내로 하고, 줄바꿈을 위해 `\\n`을 꼭 넣어줘.
                    6. 마지막 문장은 이 성향이 어떤 방향에서 강점이 될 수 있는지 긍정적으로 한줄로 마무리해줘.
                    7. 특징은 완전한 문장보단 키워드 위주로 정리해줘.

                    출력 형식은 반드시 아래처럼 맞춰줘:

                    📝 검사 결과 분석

                        너는 이런 성향을 가진 사람이야:

                        - ...
                        - ...
                        - ...

                    """),
                    HumanMessage(content=f"""
                    다음은 나의 검사 결과야:

                    현실형(R): {r_score}  
                    탐구형(I): {i_score}  
                    예술형(A): {a_score}  
                    사회형(S): {s_score}  
                    기업형(E): {e_score}  
                    관습형(C): {c_score}

                    이걸 바탕으로 내 성향을 분석해줘!
                    """)
        ]
        
        # LLM에 질문 전송 (비동기)
        task = asyncio.create_task(survey_interpreter.ainvoke(messages))
        
        # 응답 스트리밍
        full_response = ""
        async for token in stream_handler.aiter():
            full_response += token
            
        # 전체 응답을 한 번에 전송
        await websocket.send_text(full_response)
        
        # 응답 완료 신호 전송
        await websocket.send_text("<END>")
        
        # 응답 완료 대기
        await task
        
        return full_response
    except Exception as e:
        error_message = f"설문 결과 해석에 실패하였습니다 {str(e)}"
        await websocket.send_text(error_message)
        return error_message

# 문맥 정보 가져오기 함수
async def get_contexts(query: str) -> dict:
    # 직업 관련 문서 검색
    job_docs = vectorstore.similarity_search(query, k=3)
    job_context = "\n\n".join([doc.page_content for doc in job_docs])
    
    # 일반 대화 문서 검색 (직업 관련이 아닌 경우)
    general_docs = vectorstore.similarity_search(query, k=3, filter={"type": "general"})
    general_context = "\n\n".join([doc.page_content for doc in general_docs])
    
    return {
        "job_context": job_context,
        "general_context": general_context
    }

# WebSocket 연결 관리자
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.memories = {} 
        self.survey_interpretations = {}

    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # 클라이언트별 메모리 초기화
        if client_id not in self.memories:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            # 메모리 초기화
            self.memories[client_id] = memory
            # 설문 해석 결과 초기화
            self.survey_interpretations[client_id] = ""

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/test")
async def test():
    return {"message": "hello FastAPI!"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket, client_id)
    try:
        # 첫 연결 시 RIASEC 검사 결과 수신 및 해석
        survey_results = await websocket.receive_text()
        interpretation = await interpret_survey_results(survey_results, websocket)
        manager.survey_interpretations[client_id] = interpretation
        await manager.send_personal_message("\n", websocket)

        while True:
            query = await websocket.receive_text()
            memory = manager.memories[client_id]
            chat_history = memory.chat_memory.messages

            # 대화 이력 포맷팅
            formatted_history = ""
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    formatted_history += f"사용자: {message.content}\n"
                elif isinstance(message, AIMessage):
                    formatted_history += f"어시스턴트: {message.content}\n"

            # 문맥: GPT가 판단할 수 있도록 직업 관련과 일반 문서 둘 다 준비
            contexts = await get_contexts(query)
            job_context = contexts["job_context"]
            general_context = contexts["general_context"]

            # 설문 해석 결과
            survey_interpretation = manager.survey_interpretations[client_id]

            # 콜백 핸들러 설정
            stream_handler = AsyncIteratorCallbackHandler()
            token_logger = TokenUsageLogger(log_file_path)

            streaming_llm = ChatOpenAI(
                temperature=0,
                streaming=True,
                callbacks=[stream_handler, token_logger],
                model_name="gpt-4o"
            )

            # GPT가 질문 의도를 판단해서 문맥 선택하게 유도
            messages = [
                SystemMessage(content=f"""
                    너는 고등학생의 진로를 도와주는 상담 전문가야.

                    학생의 검사 결과는 다음과 같아:
                    {survey_interpretation}

                    이전 대화 내용:
                    {formatted_history}

                    이제 나의 질문이 들어올 건데, 먼저 아래 중 어떤 유형인지 판단해줘:

                    1. 직업/진로 추천 관련 질문
                    2. 일반적인 성격 상담이나 대화 진로탐색 질문
                    3. 진로탐색과 관련이 없는 대화

                    만약 직업 추천 관련 질문이면 `직업 관련 문서`를 중심으로 답하고,  
                    그 외 2번 질문이면 `일반 대화 문서`를 중심으로 대답해.
                    그리고 나머지 진로탐색과 관련없는 대화에 대해서는 답변은 하되, 진로탐색에 대해 얘기하도록 유도하는 문장을 마지막에 포함해줘.

                    내가 직업 추천을 해달라고 할 경우, 나의 검사 결과를 바탕으로 직업 1~2개를 추천하며 해당 직업에 대해 간략하게만 요약해서 최대한 가독성 좋게 보내줘.
                    학교 추천해달라고 했을때는 알려주지 마. 대신, 나에게 알려줄 수 없다고 친절하게 안내해줘.


                    직업 관련 문서:
                    {job_context}

                    일반 대화 문서:
                    {general_context}

                    반드시 한국어로 답하고, 말투는 친근하게 친구한테 하는 반말로 해줘.  
                    마크다운, 번호 매기기(1., 2.), 별표(*) 등 어떤 형식도 절대 사용하지 마.  
                    그냥 자연스럽게 문장만 써줘.  
                    분량은 200자가 넘지 않도록.
                    무조건 모든 문장은 줄바꿈해주고, 줄바꿈을 위해 `\\n`을 꼭 넣어줘.  
                    마지막 문장은 매번 다양한 이모지로 기분 좋게 마무리해줘.

                    """),
                HumanMessage(content=query)
            ]

            # 응답 생성 및 스트리밍
            async def generate_response() -> AsyncIterator[str]:
                response_tokens = []
                async for token in stream_handler.aiter():
                    response_tokens.append(token)
                    yield token

            task = asyncio.create_task(streaming_llm.ainvoke(messages))

            full_response = ""
            async for token in generate_response():
                await manager.send_personal_message(token, websocket)
                full_response += token

            await manager.send_personal_message("<END>", websocket)

            # 대화 이력 저장
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(full_response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
