//
//  ModelsAPI.swift
//  IITP_T2
//
//  Created by 소유림 on 1/4/26.
//

import Foundation

struct TrackResponse: Codable {
    let icao24: String
    let timestamp: String
    let lat: Double
    let lon: Double
    let altitude: Double?
    let ground_speed: Double?
    let callsign: String?
    let ts_generated: String?
    let ts_logged: String?
}
