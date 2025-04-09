## 📥 설치 방법
### 1. 저장소 클론 및 디렉터리 진입

```
git clone -b fastapi --single-branch https://github.com/alpaca-drun/pm_project.git
cd fastapi
```

### 2. 가상 환경 생성 및 활성화 (옵션이지만 추천)

```
python -m venv venv

venv\Scripts\activate
```


### 3. 의존성 라이브러리 설치

```
pip install -r requirements.txt
```

## ⚙️ 환경 변수 설정
- 프로젝트 루트 디렉터리에 .env 파일을 생성하고, 다음과 같이 OpenAI API 키를 입력합니다.
```
OPENAI_API_KEY=OpenAI_API_키
```

## ▶️ 서버 실행하기
```
uvicorn main:app --reload
```
## 📂 프로젝트 구조

```
pm_project/
├── main.py              # FastAPI 엔트리 포인트
├── static               # CSS, JavaScript, 이미지 등 정적 파일
├── templates            # HTML 템플릿 파일
├── requirements.txt     # 의존성 목록
└── .env                 # 환경변수 파일 (직접 생성)
```
