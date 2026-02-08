//
//  ChatBotClient.swift
//  IITP_T2
//
//  Gemini API Integration for Flight Chatbot
//

import Foundation
import CoreLocation

final class ChatBotClient {
    static let shared = ChatBotClient()
    
    // 🔑 API Key is managed in Config.swift file
    private let apiKey = Config.geminiAPIKey
    
    private let modelName = "gemini-2.0-flash-exp"  // Free tier: gemini-2.0-flash-exp or gemini-1.5-flash
    private let baseURL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    private init() {}
    
    // MARK: - Public API
    
    /// Send message to chatbot (with flight context)
    func sendMessage(
        _ userMessage: String,
        flightContext: FlightContext?
    ) async throws -> String {
        
        guard apiKey != "YOUR_GEMINI_API_KEY_HERE" else {
            throw ChatBotError.noAPIKey
        }
        
        // Build system prompt + context
        let systemPrompt = buildSystemPrompt(context: flightContext)
        let fullPrompt = "\(systemPrompt)\n\nUser question: \(userMessage)"
        
        // Call Gemini API
        let response = try await callGeminiAPI(prompt: fullPrompt)
        
        // Post-process response
        return postProcessResponse(response)
    }
    
    // MARK: - System Prompt Engineering
    
    private func buildSystemPrompt(context: FlightContext?) -> String {
        var prompt = """
        You are a friendly and helpful flight tracking AI assistant.
        Answer user questions concisely and clearly.
        
        Response rules:
        - Respond in the same language as the user's question
        - Keep answers to 2-3 sentences
        - If information is unavailable, honestly say "I don't know"
        """
        
        // Add flight context if available
        if let ctx = context {
            prompt += """
            
            
            Current flight being tracked:
            - Flight ID: \(ctx.flightId)
            - Status: \(ctx.status)
            - Origin: \(ctx.origin)
            - Destination: \(ctx.destination)
            - Current location: Latitude \(String(format: "%.4f", ctx.currentLocation.latitude)), Longitude \(String(format: "%.4f", ctx.currentLocation.longitude))
            - Estimated arrival: \(ctx.eta)
            - Time remaining: \(formatRemainingTime(ctx.remainingSeconds))
            - Last update: \(formatRelativeTime(ctx.lastUpdate))
            """
        } else {
            prompt += """
            
            
            ⚠️ No flight is currently being tracked.
            Ask the user to search for a flight first.
            """
        }
        
        return prompt
    }
    
    // MARK: - Gemini API Call
    
    private func callGeminiAPI(prompt: String) async throws -> String {
        // Build URL
        guard var urlComponents = URLComponents(string: "\(baseURL)/\(modelName):generateContent") else {
            throw ChatBotError.invalidURL
        }
        urlComponents.queryItems = [URLQueryItem(name: "key", value: apiKey)]
        
        guard let url = urlComponents.url else {
            throw ChatBotError.invalidURL
        }
        
        // Build request body
        let request = GeminiRequest(
            contents: [
                GeminiRequest.Content(
                    parts: [GeminiRequest.Part(text: prompt)],
                    role: "user"
                )
            ],
            generationConfig: GeminiRequest.GenerationConfig(
                temperature: 0.7,
                topK: 40,
                topP: 0.95,
                maxOutputTokens: 1024
            )
        )
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        
        print("🤖 Calling Gemini API: \(url.absoluteString)")
        
        // API call
        do {
            let (data, response) = try await URLSession.shared.data(for: urlRequest)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ChatBotError.networkError("No HTTP response")
            }
            
            // Check status code
            guard (200...299).contains(httpResponse.statusCode) else {
                let errorBody = String(data: data, encoding: .utf8) ?? "(no body)"
                throw ChatBotError.apiError("HTTP \(httpResponse.statusCode): \(errorBody)")
            }
            
            // Parse response
            do {
                let geminiResponse = try JSONDecoder().decode(GeminiResponse.self, from: data)
                
                guard let text = geminiResponse.text, !text.isEmpty else {
                    throw ChatBotError.noResponse
                }
                
                print("✅ Gemini response received: \(text.prefix(100))...")
                return text
                
            } catch {
                let bodyPreview = String(data: data, encoding: .utf8) ?? "(not utf8)"
                throw ChatBotError.decodingError("Parse error: \(error)\nBody: \(bodyPreview)")
            }
            
        } catch let error as ChatBotError {
            throw error
        } catch {
            throw ChatBotError.networkError(error.localizedDescription)
        }
    }
    
    // MARK: - Response Post-processing
    
    private func postProcessResponse(_ response: String) -> String {
        var processed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Remove markdown code blocks if present
        processed = processed.replacingOccurrences(of: "```", with: "")
        
        // Truncate if too long (500 char limit)
        if processed.count > 500 {
            let index = processed.index(processed.startIndex, offsetBy: 500)
            processed = String(processed[..<index]) + "..."
        }
        
        return processed
    }
    
    // MARK: - Helpers
    
    private func formatRemainingTime(_ seconds: Int) -> String {
        if seconds <= 0 { return "arriving soon" }
        let h = seconds / 3600
        let m = (seconds % 3600) / 60
        if h > 0 { return "\(h)h \(m)m" }
        return "\(m)m"
    }
    
    private func formatRelativeTime(_ date: Date) -> String {
        let diff = Int(Date().timeIntervalSince(date))
        if diff < 3 { return "just now" }
        if diff < 60 { return "\(diff)s ago" }
        let m = diff / 60
        if m < 60 { return "\(m)m ago" }
        let h = m / 60
        return "\(h)h ago"
    }
}
