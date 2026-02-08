//
//  APIClient.swift
//  IITP_T2
//
//  Created by 소유림 on 1/4/26.
//

import Foundation

enum APIError: LocalizedError {
    case badURL
    case badStatus(Int, body: String)
    case decodeFailed(String)
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .badURL:
            return "Invalid URL"
        case .badStatus(let code, let body):
            return "HTTP \(code): \(body)"
        case .decodeFailed(let msg):
            return "Decode failed: \(msg)"
        case .transport(let msg):
            return "Network error: \(msg)"
        }
    }
}

final class APIClient {
    static let shared = APIClient()
    private init() {}

    // ✅ 시뮬레이터에서 Mac 로컬 uvicorn을 때릴 때 보통 OK
    // 실기기는 Mac의 IP로 바꿔야 함 (예: http://192.168.x.x:8080)
    private let baseURL = "http://127.0.0.1:8080"

    // ✅ FastAPI Query 제한 le=99360 과 맞춤
    private let maxMinutes = 99360

    func fetchTrack(hex rawHex: String, minutes rawMinutes: Int) async throws -> TrackResponse {
        let hex = normalizeHex(rawHex)
        let minutes = min(max(rawMinutes, 1), maxMinutes)

        var comps = URLComponents(string: "\(baseURL)/track")
        comps?.queryItems = [
            URLQueryItem(name: "hex", value: hex),
            URLQueryItem(name: "minutes", value: String(minutes))
        ]
        guard let url = comps?.url else { throw APIError.badURL }

        print("📡 GET:", url.absoluteString)

        do {
            let (data, resp) = try await URLSession.shared.data(from: url)

            guard let http = resp as? HTTPURLResponse else {
                throw APIError.transport("No HTTPURLResponse")
            }

            guard (200...299).contains(http.statusCode) else {
                let body = String(data: data, encoding: .utf8) ?? "(no body)"
                throw APIError.badStatus(http.statusCode, body: body)
            }

            do {
                return try JSONDecoder().decode(TrackResponse.self, from: data)
            } catch {
                let body = String(data: data, encoding: .utf8) ?? "(not utf8)"
                throw APIError.decodeFailed("error=\(error)\nbody=\(body)")
            }
        } catch {
            throw APIError.transport(error.localizedDescription)
        }
    }

    // hex 입력 정리: 공백 제거 / 0x 제거 / 소문자 / hex문자만 남김
    private func normalizeHex(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if s.hasPrefix("0x") { s.removeFirst(2) }
        s = s.filter { ("0"..."9").contains($0) || ("a"..."f").contains($0) }
        return s
    }
}
