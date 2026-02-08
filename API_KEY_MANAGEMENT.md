# API Key 관리 가이드

## 🔐 방법 1: Config.swift 파일 분리 (✅ 현재 적용됨)

### 설정 방법

1. **`app/Config.swift`** 파일 열기
2. API Key 입력:
   ```swift
   static let geminiAPIKey = "여기에_실제_API키_입력"
   ```

### 장점
- ✅ 간단하고 직관적
- ✅ `.gitignore`로 자동으로 보호됨
- ✅ 팀원들은 `Config.swift.example`을 복사해서 사용

### 주의사항
```bash
# Config.swift는 Git에 커밋되지 않습니다
# 새로운 환경에서는 Config.swift.example을 복사해서 사용하세요
cp app/Config.swift.example app/Config.swift
```

---

## 🔐 방법 2: Info.plist (환경변수 느낌)

### 구현 방법

1. **Info.plist** 생성 (Xcode에서)
2. Key-Value 추가:
   ```xml
   <key>GEMINI_API_KEY</key>
   <string>$(GEMINI_API_KEY)</string>
   ```

3. **Xcode Scheme에서 환경변수 설정**:
   - Product → Scheme → Edit Scheme
   - Run → Arguments → Environment Variables
   - `GEMINI_API_KEY` = 실제 키 입력

4. **코드에서 읽기**:
   ```swift
   let apiKey = Bundle.main.infoDictionary?["GEMINI_API_KEY"] as? String ?? ""
   ```

### 장점
- ✅ 진짜 환경변수처럼 사용
- ✅ 빌드 설정별로 다른 키 사용 가능 (Development/Production)

### 단점
- ❌ 설정이 복잡함
- ❌ Xcode 설정 파일도 `.gitignore` 필요

---

## 🔐 방법 3: Keychain (가장 안전)

### 구현 방법

```swift
import Security

class KeychainManager {
    static func save(key: String, value: String) {
        let data = value.data(using: .utf8)!
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data
        ]
        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }
    
    static func load(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]
        var result: AnyObject?
        SecItemCopyMatching(query as CFDictionary, &result)
        if let data = result as? Data {
            return String(data: data, encoding: .utf8)
        }
        return nil
    }
}

// 사용
KeychainManager.save(key: "geminiAPIKey", value: "실제키")
let apiKey = KeychainManager.load(key: "geminiAPIKey") ?? ""
```

### 장점
- ✅ 가장 안전 (iOS 시스템 레벨 암호화)
- ✅ 앱 삭제 후에도 유지 가능

### 단점
- ❌ 구현 복잡
- ❌ 최초 설정 UI 필요

---

## 🔐 방법 4: 원격 서버에서 가져오기

### 구조

```
iOS App
    ↓ 인증 토큰
Backend Server (당신이 관리)
    ↓ API 키 전달
Gemini API
```

### 장점
- ✅ 가장 안전 (API 키가 앱에 없음)
- ✅ 키 교체가 쉬움

### 단점
- ❌ 백엔드 서버 필요
- ❌ 복잡도 증가

---

## 📊 방법 비교

| 방법 | 난이도 | 보안성 | 추천도 |
|------|--------|--------|--------|
| **Config.swift** | ⭐ 쉬움 | ⭐⭐⭐ 보통 | ✅ **개발/소규모** |
| Info.plist | ⭐⭐ 중간 | ⭐⭐⭐ 보통 | ✅ 다중 환경 |
| Keychain | ⭐⭐⭐ 어려움 | ⭐⭐⭐⭐⭐ 매우 높음 | ✅ 프로덕션 |
| 원격 서버 | ⭐⭐⭐⭐ 매우 어려움 | ⭐⭐⭐⭐⭐ 매우 높음 | ✅ 대규모 앱 |

---

## 🎯 현재 프로젝트 권장

**방법 1 (Config.swift)** ← 현재 적용됨 ✅

이유:
- 간단하고 빠르게 시작 가능
- 팀 협업에 적합
- 학교 프로젝트/프로토타입에 충분

---

## 🚀 사용 방법 (현재 설정)

### 1. API 키 설정
```bash
# app/Config.swift 파일 열기
# geminiAPIKey에 실제 키 입력
```

### 2. Git 관리
```bash
# Config.swift는 자동으로 무시됨
git status  # Config.swift가 보이지 않아야 함

# Config.swift.example은 커밋 OK
git add app/Config.swift.example
git commit -m "Add API key configuration template"
```

### 3. 팀원 세팅
```bash
# 새로운 팀원이 클론 후
cp app/Config.swift.example app/Config.swift
# 그리고 Config.swift에 자신의 API 키 입력
```

---

## ⚠️ 보안 체크리스트

- [x] `Config.swift`가 `.gitignore`에 있음
- [x] `Config.swift.example`에는 실제 키가 없음
- [ ] 커밋 전에 `git status`로 확인
- [ ] 실수로 커밋했다면 즉시 API 키 재발급

---

## 💡 나중에 업그레이드한다면?

프로덕션 앱으로 발전시킬 때:
1. **Keychain으로 이전** (보안 강화)
2. **CI/CD에서 자동 주입** (GitHub Actions 등)
3. **원격 서버 프록시** (완벽한 보안)

지금은 Config.swift로 충분합니다! 🎉
