//
//  ChatModels.swift
//  IITP_T2
//
//  Created for Gemini Chatbot Integration
//

import Foundation
import CoreLocation

// MARK: - Flight Context for Chatbot

struct FlightContext: Codable {
    let flightId: String
    let status: String
    let origin: String
    let destination: String
    let currentLocation: LocationInfo
    let eta: String
    let remainingSeconds: Int
    let lastUpdate: Date
    
    struct LocationInfo: Codable {
        let latitude: Double
        let longitude: Double
        let description: String
    }
    
    /// Create context from ContentView's Flight object
    static func from(flight: Flight, currentCoord: CLLocationCoordinate2D) -> FlightContext {
        return FlightContext(
            flightId: flight.displayId,
            status: flight.statusText,
            origin: flight.originName,
            destination: flight.destName,
            currentLocation: LocationInfo(
                latitude: currentCoord.latitude,
                longitude: currentCoord.longitude,
                description: "(\(String(format: "%.4f", currentCoord.latitude)), \(String(format: "%.4f", currentCoord.longitude)))"
            ),
            eta: flight.etaText,
            remainingSeconds: flight.remainingSeconds,
            lastUpdate: flight.lastUpdate
        )
    }
}

// MARK: - Gemini API Request/Response Models

struct GeminiRequest: Codable {
    let contents: [Content]
    let generationConfig: GenerationConfig?
    
    struct Content: Codable {
        let parts: [Part]
        let role: String?
    }
    
    struct Part: Codable {
        let text: String
    }
    
    struct GenerationConfig: Codable {
        let temperature: Double?
        let topK: Int?
        let topP: Double?
        let maxOutputTokens: Int?
    }
}

struct GeminiResponse: Codable {
    let candidates: [Candidate]
    
    struct Candidate: Codable {
        let content: Content
        let finishReason: String?
        
        struct Content: Codable {
            let parts: [Part]
            let role: String?
        }
        
        struct Part: Codable {
            let text: String
        }
    }
    
    var text: String? {
        candidates.first?.content.parts.first?.text
    }
}

// MARK: - Error Types

enum ChatBotError: LocalizedError {
    case noAPIKey
    case invalidURL
    case networkError(String)
    case noResponse
    case decodingError(String)
    case apiError(String)
    
    var errorDescription: String? {
        switch self {
        case .noAPIKey:
            return "Gemini API Key is not configured."
        case .invalidURL:
            return "Invalid API URL."
        case .networkError(let msg):
            return "Network error: \(msg)"
        case .noResponse:
            return "No response received from API."
        case .decodingError(let msg):
            return "Response parsing error: \(msg)"
        case .apiError(let msg):
            return "API error: \(msg)"
        }
    }
}
