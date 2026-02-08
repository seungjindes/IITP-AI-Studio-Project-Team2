# IITP-AI-Studio-Project-Team2 - Gemini Chatbot Setup Guide

## 🤖 Gemini AI 챗봇 통합 완료!

이 프로젝트에 **Google Gemini AI**를 사용한 자연어 항공편 챗봇이 통합되었습니다.

---

## 📁 새로 추가된 파일

```
app/
├── ChatModels.swift        # 챗봇 데이터 모델
├── ChatBotClient.swift     # Gemini API 클라이언트
└── ContentView.swift       # 수정됨 - AI 챗봇 연동
```

---

## 🔑 1단계: Gemini API Key 발급

1. **Google AI Studio** 방문: https://ai.google.dev/
2. **Get API Key** 클릭
3. API Key 복사

---

## ⚙️ 2단계: API Key 설정

### 방법 1: 코드에 직접 입력 (개발용)

`app/ChatBotClient.swift` 파일을 열고 12번째 줄을 수정:

```swift
private let apiKey = "YOUR_ACTUAL_API_KEY_HERE"
```

**⚠️ 주의**: Git에 커밋하기 전에 API Key를 제거하세요!

### 방법 2: 환경변수 사용 (프로덕션 추천)

추후 구현 가능:
- Info.plist에 저장
- Keychain 사용
- Configuration 파일 사용

---

## 🚀 3단계: 앱 실행

1. Xcode에서 프로젝트 열기
2. 빌드 및 실행
3. 항공편 검색 (예: AA0021 또는 hex 코드)
4. **Ask** 버튼 클릭하여 챗봇 시작

---

## 💬 4단계: 챗봇 사용하기

### 질문 예시

**기본 질문:**
- "나 지금 어디야?"
- "언제 도착해?"
- "남은 시간 알려줘"
- "비행 상태는?"

**자유로운 질문:**
- "고도가 얼마나 돼?"
- "목적지까지 얼마나 남았어?"
- "출발지가 어디야?"
- "현재 속도는?"
- "어느 방향으로 가고 있어?"

**자연스러운 대화:**
- "지금 뭐해?"
- "잘 가고 있어?"
- "곧 도착해?"

---

## 🔧 커스터마이징

### 모델 변경

`app/ChatBotClient.swift`의 15번째 줄:

```swift
// 무료 모델
private let modelName = "gemini-2.0-flash-exp"     // 빠르고 가벼움
// private let modelName = "gemini-1.5-flash"      // 안정적
// private let modelName = "gemini-1.5-pro"        // 더 똑똑함 (유료)
```

### 프롬프트 수정

`app/ChatBotClient.swift`의 `buildSystemPrompt()` 함수에서 챗봇 성격 변경 가능:

```swift
private func buildSystemPrompt(context: FlightContext?) -> String {
    var prompt = """
    당신은 친근하고 도움이 되는 항공편 추적 AI 어시스턴트입니다.
    // ✏️ 여기를 수정하여 챗봇 성격 변경
    """
    ...
}
```

### 응답 길이 조절

`app/ChatBotClient.swift`의 `callGeminiAPI()` 함수:

```swift
maxOutputTokens: 1024  // 더 길게: 2048, 짧게: 512
```

---

## 📊 Gemini API 무료 할당량

- **gemini-2.0-flash-exp**: 분당 10 요청
- **gemini-1.5-flash**: 분당 15 요청, 일일 1500 요청
- **gemini-1.5-pro**: 분당 2 요청

자세한 내용: https://ai.google.dev/pricing

---

## 🐛 문제 해결

### "Gemini API Key가 설정되지 않았습니다"
→ `ChatBotClient.swift`의 `apiKey` 변수를 확인하세요.

### "네트워크 오류"
→ 인터넷 연결을 확인하세요.

### "HTTP 429: Too Many Requests"
→ API 할당량 초과. 잠시 후 다시 시도하세요.

### "HTTP 403: Forbidden"
→ API Key가 잘못되었거나 만료되었습니다.

---

## 🔒 보안 주의사항

1. **절대 API Key를 Git에 커밋하지 마세요!**
2. `.gitignore`에 설정 파일 추가
3. 프로덕션에서는 환경변수 사용
4. API Key 정기적으로 교체

---

## 📝 아키텍처 설명

```
사용자 질문
    ↓
ContentView.sendChat()
    ↓
ChatBotClient.sendMessage()
    ↓
- FlightContext 생성 (현재 항공편 정보)
- System Prompt 구성
- Gemini API 호출
    ↓
응답 받음
    ↓
후처리 (길이 제한, 포맷팅)
    ↓
UI에 표시
```

---

## 🎨 UI 개선 사항

- ✅ "생각 중..." 로딩 표시
- ✅ 에러 메시지 친절하게 표시
- ✅ 제안 칩 4개로 확장
- ✅ 초기 인사말 개선

---

## 🚧 향후 개선 사항

- [ ] 스트리밍 응답 (실시간으로 글자 표시)
- [ ] 대화 히스토리 유지 (멀티턴 대화)
- [ ] 음성 입력/출력
- [ ] 다국어 지원
- [ ] 챗봇 성격 선택 (친절형/전문가형 등)

---

## 📞 문의

문제가 발생하면 이슈를 남겨주세요!

Enjoy your AI-powered flight tracking! ✈️🤖
