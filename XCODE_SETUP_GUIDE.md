# 🚀 Xcode에서 IITP_T2 앱 실행하기

## 방법 1: 새 프로젝트 생성 (추천)

### 1단계: Xcode 실행 및 새 프로젝트 생성

```bash
# 터미널에서 Xcode 실행
open -a Xcode
```

또는 Spotlight(⌘ + Space)에서 "Xcode" 검색

### 2단계: 새 프로젝트 만들기

1. **Xcode 실행 후**
   - `File` → `New` → `Project...` (또는 ⌘ + Shift + N)

2. **템플릿 선택**
   - `iOS` 탭 선택
   - `App` 선택
   - `Next` 클릭

3. **프로젝트 설정**
   ```
   Product Name: IITP_T2
   Team: None (개인 개발)
   Organization Identifier: com.iitp (또는 원하는 것)
   Bundle Identifier: com.iitp.IITP-T2 (자동 생성됨)
   Interface: SwiftUI ✅
   Language: Swift ✅
   Storage: None
   □ Include Tests (체크 해제해도 됨)
   ```
   - `Next` 클릭

4. **저장 위치**
   - `/Users/sb/Downloads/workspace/` 선택
   - **⚠️ 주의**: `IITP-AI-Studio-Project-Team2` 폴더 안이 아니라 그 부모 폴더
   - 프로젝트 이름을 `IITP_T2`로 저장

### 3단계: 기존 코드 파일 교체

프로젝트가 생성되면 자동으로 다음 파일들이 만들어집니다:
- `IITP_T2App.swift`
- `ContentView.swift`
- `Assets.xcassets`

**이제 우리가 만든 파일들로 교체합니다:**

1. **Finder에서 파일 복사**
   ```bash
   # 터미널에서 실행
   cd /Users/sb/Downloads/workspace/IITP-AI-Studio-Project-Team2
   
   # 새로 만든 프로젝트로 파일 복사
   cp app/IITP_T2App.swift ~/Downloads/workspace/IITP_T2/IITP_T2/
   cp app/ContentView.swift ~/Downloads/workspace/IITP_T2/IITP_T2/
   cp app/APIClient.swift ~/Downloads/workspace/IITP_T2/IITP_T2/
   cp app/ModelsAPI.swift ~/Downloads/workspace/IITP_T2/IITP_T2/
   cp app/ChatModels.swift ~/Downloads/workspace/IITP_T2/IITP_T2/
   cp app/ChatBotClient.swift ~/Downloads/workspace/IITP_T2/IITP_T2/
   cp app/Config.swift ~/Downloads/workspace/IITP_T2/IITP_T2/
   ```

2. **Xcode에서 파일 추가**
   - Xcode에서 왼쪽 프로젝트 네비게이터의 `IITP_T2` 폴더에 마우스 우클릭
   - `Add Files to "IITP_T2"...` 선택
   - 위에서 복사한 파일들 선택:
     - APIClient.swift
     - ModelsAPI.swift
     - ChatModels.swift
     - ChatBotClient.swift
     - Config.swift
   - `Add` 클릭

### 4단계: 빌드 및 실행

1. **시뮬레이터 선택**
   - Xcode 상단의 디바이스 선택 메뉴에서 `iPhone 15 Pro` 선택 (또는 원하는 디바이스)

2. **빌드**
   - `Product` → `Build` (또는 ⌘ + B)
   - 에러가 없는지 확인

3. **실행!**
   - `Product` → `Run` (또는 ⌘ + R)
   - 또는 왼쪽 상단의 ▶️ 버튼 클릭

---

## 방법 2: 기존 폴더에서 직접 열기 (더 빠름!)

```bash
# 터미널에서 실행
cd /Users/sb/Downloads/workspace/IITP-AI-Studio-Project-Team2/app
open -a Xcode IITP_T2App.swift
```

Xcode가 열리면:

1. **Swift Package 생성 요청이 뜨면** "Create" 클릭
2. **왼쪽 네비게이터에서** 모든 .swift 파일이 보이는지 확인
3. **상단에서 타겟 추가**:
   - File → New → Target
   - iOS → App
   - Product Name: IITP_T2
   - Add to project: IITP_T2

---

## 🐛 문제 해결

### 문제 1: "No such module 'SwiftUI'"
→ iOS 타겟 버전 확인 (iOS 17.0 이상 필요)

### 문제 2: "Config.swift not found"
→ Xcode 프로젝트에 Config.swift 추가했는지 확인

### 문제 3: "App crashes on launch"
→ Info.plist에 네트워크 권한 추가 필요:
```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

---

## 📱 실행 후 테스트

1. **항공편 검색**
   - 검색창에 `AA0021` 입력
   - 또는 hex 코드 (예: `71c218`)

2. **챗봇 테스트**
   - 하단 카드의 `Ask` 버튼 클릭
   - 질문: "나 지금 어디야?"
   - AI 응답이 오는지 확인!

---

## 💡 팁

### 실시간 프리뷰 보기
```swift
// ContentView.swift 맨 아래에 있음
#Preview { ContentView() }
```
- Canvas에서 실시간으로 UI 확인 가능
- Canvas 안 보이면: `Editor` → `Canvas` (또는 ⌘ + Option + Return)

### 디버깅
- 콘솔 로그 보기: `View` → `Debug Area` → `Show Debug Area` (⌘ + Shift + Y)
- API 호출 로그가 콘솔에 출력됩니다

---

어떤 방법을 선호하시나요? 제가 터미널 명령어로 도와드릴까요?
