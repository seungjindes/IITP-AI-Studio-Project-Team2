#!/bin/bash
# Xcode 프로젝트 생성 및 실행 스크립트

echo "🚀 IITP_T2 Xcode 프로젝트 생성 중..."

PROJECT_DIR="/Users/sb/Downloads/workspace/IITP-AI-Studio-Project-Team2"
cd "$PROJECT_DIR"

# SwiftPM 패키지 생성
mkdir -p IITP_T2.xcodeproj

# Package.swift 생성 (iOS App용)
cat > Package.swift << 'EOF'
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "IITP_T2",
    platforms: [
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "IITP_T2",
            targets: ["IITP_T2"])
    ],
    targets: [
        .target(
            name: "IITP_T2",
            path: "app")
    ]
)
EOF

echo "✅ 프로젝트 파일 생성 완료!"
echo ""
echo "📱 Xcode 실행 방법:"
echo "1. Finder에서 app 폴더 열기"
echo "2. IITP_T2App.swift를 Xcode로 드래그"
echo "3. 또는 아래 명령어 실행:"
echo ""
echo "   open -a Xcode app/"
echo ""
