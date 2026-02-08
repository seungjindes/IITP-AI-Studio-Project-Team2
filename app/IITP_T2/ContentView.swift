import SwiftUI
import MapKit
import Combine
import CoreLocation

// MARK: - Models

struct Flight: Identifiable {
    let id = UUID()

    // ✅ 화면 표시용 (AA0021 또는 사용자가 입력한 것)
    var displayId: String

    // ✅ 서버 조회용 hex (실데이터일 때만 존재)
    // AA0021 더미일 때는 nil
    var hexKey: String?

    var originName: String
    var originCoord: CLLocationCoordinate2D
    var destName: String
    var destCoord: CLLocationCoordinate2D

    var etaText: String
    var statusText: String
    var remainingSeconds: Int
    var lastUpdate: Date
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let isUser: Bool
    let text: String
    let time: Date
}

// MARK: - Main View

struct ContentView: View {

    // Search UI
    @State private var flightInput: String = ""
    @State private var hasSearchedSuccessfully: Bool = false

    @State private var searchStatus: String = "Enter AA0021 (dummy) or a hex (e.g., 71c218) and tap Search."
    @State private var statusKind: StatusKind = .info
    enum StatusKind { case info, success, error }

    // Map mode
    enum MapMode { case reset, follow }
    @State private var mapMode: MapMode = .reset

    // Flight + Map
    @State private var currentFlight: Flight? = nil
    @State private var mapPosition: MapCameraPosition = .automatic

    // Track points
    @State private var trackPoints: [CLLocationCoordinate2D] = []
    @State private var planeHeadingDeg: Double = 0.0

    // Tracking ticker
    @State private var tickCount: Int = 0
    private let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()

    // Chat
    @State private var isChatPresented: Bool = false
    @State private var chatInput: String = ""
    @State private var messages: [ChatMessage] = [
        ChatMessage(isUser: false, text: "Hi! Ask me things like “나 지금 어디야?”", time: Date())
    ]

    private var isLive: Bool { currentFlight != nil && mapMode == .follow }

    var body: some View {
        ZStack {
            mapLayer

            if !hasSearchedSuccessfully {
                VStack(spacing: 10) {
                    topSearchBar
                    statusBanner
                }
                .padding(.horizontal, 14)
                .padding(.top, 10)
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity, alignment: .top)
            }

            if isLive {
                liveBadge
                    .padding(.top, 12)
                    .padding(.trailing, 14)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            }
        }
        .safeAreaInset(edge: .bottom) {
            if currentFlight != nil {
                bottomFlightCard
            }
        }
        .onReceive(timer) { _ in
            tick()
        }
        .sheet(isPresented: $isChatPresented) {
            chatSheet
        }
    }

    // MARK: - Map

    private var mapLayer: some View {
        Map(position: $mapPosition) {
            if let flight = currentFlight {
                Annotation(flight.originName, coordinate: flight.originCoord) {
                    markerView(title: flight.originName, systemImage: "airplane.departure")
                }
                Annotation(flight.destName, coordinate: flight.destCoord) {
                    markerView(title: flight.destName, systemImage: "airplane.arrival")
                }
            }

            // (A) Past track (curved)
            if trackPoints.count >= 2 {
                let smooth = smoothTrack(trackPoints, samplesPerSegment: 8)
                MapPolyline(coordinates: smooth)
                    .stroke(.blue, lineWidth: 4)
            }

            // (B) Future guide line: current -> destination (great-circle, dashed)
            if let flight = currentFlight, let plane = trackPoints.last {
                let future = greatCirclePath(from: plane, to: flight.destCoord, samples: 48)
                MapPolyline(coordinates: future)
                    .stroke(.blue.opacity(0.35),
                            style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [7, 7]))
            }

            // Plane marker
            if let plane = trackPoints.last {
                Annotation("Plane", coordinate: plane) {
                    planeMarkerView
                }
            }
        }
        .mapStyle(.standard)
        .ignoresSafeArea()
    }

    private func markerView(title: String, systemImage: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.title3)
                .padding(10)
                .background(Material.thin)
                .clipShape(Circle())

            Text(title)
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Material.thin)
                .clipShape(Capsule())
        }
    }

    private var planeMarkerView: some View {
        VStack(spacing: 6) {
            Image(systemName: "airplane")
                .font(.title2)
                .rotationEffect(.degrees(planeHeadingDeg))
                .padding(10)
                .background(Material.ultraThin)
                .clipShape(Circle())
                .overlay(Circle().strokeBorder(Color.blue.opacity(0.55), lineWidth: 2))

            if let f = currentFlight {
                Text(f.displayId)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Material.ultraThin)
                    .clipShape(Capsule())
            }
        }
    }

    // MARK: - Top UI (Search)

    private var topSearchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)

            TextField("AA0021 (dummy) or hex (e.g., 71c218)", text: $flightInput)
                .textInputAutocapitalization(.characters)
                .autocorrectionDisabled(true)
                .submitLabel(.search)
                .onSubmit { performSearch() }

            Button("Search") { performSearch() }
                .buttonStyle(.borderedProminent)
                .disabled(flightInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(12)
        .background(Material.thin)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Color.secondary.opacity(0.18), lineWidth: 1))
    }

    private var statusBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: statusIconName)
                .font(.subheadline.weight(.semibold))

            Text(searchStatus)
                .font(.footnote.weight(.semibold))
                .foregroundStyle(.primary)
                .lineLimit(2)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(statusBackground)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(statusBorder, lineWidth: 1))
    }

    private var statusIconName: String {
        switch statusKind {
        case .info: return "info.circle"
        case .success: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    private var statusBackground: some ShapeStyle {
        switch statusKind {
        case .info: return AnyShapeStyle(Material.thin)
        case .success: return AnyShapeStyle(Color.green.opacity(0.20))
        case .error: return AnyShapeStyle(Color.red.opacity(0.20))
        }
    }

    private var statusBorder: Color {
        switch statusKind {
        case .info: return Color.secondary.opacity(0.25)
        case .success: return Color.green.opacity(0.45)
        case .error: return Color.red.opacity(0.45)
        }
    }

    private var liveBadge: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color.blue)
                .frame(width: 8, height: 8)
                .overlay(Circle().stroke(Color.blue.opacity(0.35), lineWidth: 6).blur(radius: 2))
            Text("LIVE")
                .font(.caption.weight(.bold))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Material.thin)
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Color.blue.opacity(0.35), lineWidth: 1))
    }

    // MARK: - Bottom Card

    private var bottomFlightCard: some View {
        guard let flight = currentFlight else { return AnyView(EmptyView()) }

        return AnyView(
            VStack(spacing: 12) {
                Capsule()
                    .fill(Color.secondary.opacity(0.35))
                    .frame(width: 44, height: 5)
                    .padding(.top, 8)

                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 10) {
                            Text(flight.displayId)
                                .font(.headline)

                            Text(flight.statusText)
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(Color.blue.opacity(0.14))
                                .clipShape(Capsule())
                        }

                        Text("\(flight.originName) → \(flight.destName)")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Text("ETA \(flight.etaText)")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Button {
                        resetToSearchState()
                    } label: {
                        Label("Change", systemImage: "pencil")
                    }
                    .buttonStyle(.bordered)
                }

                Divider()

                HStack {
                    statCell(title: "Status", value: flight.statusText)
                    Spacer()
                    statCell(title: "Time remaining", value: timeRemainingText(flight.remainingSeconds))
                    Spacer()
                    statCell(title: "Last update", value: relativeTime(from: flight.lastUpdate))
                }

                GeometryReader { geo in
                    HStack(spacing: 10) {
                        modeSegment
                            .frame(width: geo.size.width * 0.60)

                        Button {
                            isChatPresented = true
                        } label: {
                            Label("Ask", systemImage: "sparkles")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .frame(width: geo.size.width * 0.40)
                    }
                }
                .frame(height: 44)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 14)
            .background(Material.thin)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(Color.secondary.opacity(0.18), lineWidth: 1))
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 10)
        )
    }

    private var modeSegment: some View {
        HStack(spacing: 0) {
            segmentButton(title: "Follow", systemImage: "location", isSelected: mapMode == .follow) {
                setMapMode(.follow)
            }
            segmentButton(title: "Reset", systemImage: "arrow.counterclockwise", isSelected: mapMode == .reset) {
                setMapMode(.reset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.primary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Color.secondary.opacity(0.18), lineWidth: 1))
    }

    private func segmentButton(title: String, systemImage: String, isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(isSelected ? Color.white : Color.primary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        .background(
            Group {
                if isSelected {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.blue)
                        .padding(2)
                } else {
                    Color.clear
                }
            }
        )
    }

    private func statCell(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
    }

    // MARK: - Chat

    private var chatSheet: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            ForEach(messages) { msg in
                                chatBubble(msg).id(msg.id)
                            }
                        }
                        .padding(14)
                    }
                    .onChange(of: messages.count) { _, _ in
                        if let last = messages.last {
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                }

                Divider()

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        suggestionChip("나 지금 어디야?")
                        suggestionChip("Status 뭐야?")
                        suggestionChip("남은 시간 알려줘")
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                }

                HStack(spacing: 10) {
                    TextField("Message", text: $chatInput, axis: .vertical)
                        .textFieldStyle(.roundedBorder)

                    Button("Send") { sendChat() }
                        .buttonStyle(.borderedProminent)
                        .disabled(chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(14)
            }
            .navigationTitle("Chat")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Close") { isChatPresented = false }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func suggestionChip(_ text: String) -> some View {
        Button {
            chatInput = text
            sendChat()
        } label: {
            Text(text)
                .font(.footnote.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(Material.thin)
                .clipShape(Capsule())
        }
    }

    private func chatBubble(_ msg: ChatMessage) -> some View {
        HStack {
            if msg.isUser { Spacer(minLength: 40) }
            Text(msg.text)
                .font(.body)
                .padding(12)
                .background(msg.isUser ? Color.blue.opacity(0.18) : Color(.systemBackground).opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .frame(maxWidth: 520, alignment: msg.isUser ? .trailing : .leading)
            if !msg.isUser { Spacer(minLength: 40) }
        }
    }

    // MARK: - Search / State

    private func performSearch() {
        let input = flightInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return }
        searchFlight(input: input)
    }

    private func searchFlight(input: String) {
        let clean = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let upper = clean.uppercased()

        // ✅ AA0021만 더미로 동작
        if upper == "AA0021" {
            loadDummyFlightAA0021()
            hasSearchedSuccessfully = true
            flightInput = ""
            return
        }

        // ✅ 나머지는 hex 조회
        let hex = normalizeHex(clean)

        statusKind = .info
        searchStatus = "Fetching from server…"

        Task {
            do {
                let res = try await APIClient.shared.fetchTrack(hex: hex, minutes: 99360) // 최근 1일 윈도우
                await MainActor.run {
                    let point = CLLocationCoordinate2D(latitude: res.lat, longitude: res.lon)

                    // 목적지는 아직 모르니 임시 DEST
                    let dummyDest = CLLocationCoordinate2D(latitude: 40.4406, longitude: -79.9959) // PIT

                    currentFlight = Flight(
                        displayId: upper,   // 화면에 표시
                        hexKey: hex,        // ✅ 서버 조회는 이걸로만
                        originName: "LIVE",
                        originCoord: point,
                        destName: "DEST",
                        destCoord: dummyDest,
                        etaText: "--:--",
                        statusText: "LIVE",
                        remainingSeconds: 0,
                        lastUpdate: Date()
                    )

                    trackPoints = [point]
                    planeHeadingDeg = planeRotationDegrees(from: point, to: dummyDest)

                    hasSearchedSuccessfully = true
                    flightInput = ""
                    statusKind = .success
                    searchStatus = "✅ Live point loaded."
                    fitToFullRoute(animated: true)
                }
            } catch {
                await MainActor.run {
                    statusKind = .error
                    searchStatus = "❌ Failed: \(error.localizedDescription)"
                    resetTrackingOnly()
                    currentFlight = nil
                    hasSearchedSuccessfully = false
                }
            }
        }
    }

    private func resetToSearchState() {
        currentFlight = nil
        flightInput = ""
        hasSearchedSuccessfully = false
        statusKind = .info
        searchStatus = "Enter AA0021 (dummy) or a hex (e.g., 71c218) and tap Search."
        resetTrackingOnly()
        mapPosition = .automatic
        mapMode = .reset
    }

    private func resetTrackingOnly() {
        trackPoints = []
        planeHeadingDeg = 0
        tickCount = 0
    }

    // MARK: - Dummy AA0021

    private func loadDummyFlightAA0021() {
        let nyc = CLLocationCoordinate2D(latitude: 40.7128, longitude: -74.0060)
        let pit = CLLocationCoordinate2D(latitude: 40.4406, longitude: -79.9959)

        currentFlight = Flight(
            displayId: "AA0021",
            hexKey: nil, // ✅ 더미!
            originName: "NYC",
            originCoord: nyc,
            destName: "PIT",
            destCoord: pit,
            etaText: "18:42",
            statusText: "ENROUTE (DUMMY)",
            remainingSeconds: 2 * 60 * 60,
            lastUpdate: Date().addingTimeInterval(-20)
        )

        trackPoints = [nyc]
        planeHeadingDeg = planeRotationDegrees(from: nyc, to: pit)

        statusKind = .success
        searchStatus = "Found: NYC → PIT (AA0021 dummy). Route & tracking started."

        mapMode = .reset
        fitToFullRoute(animated: true)
    }

    // MARK: - Tick

    private func tick() {
        guard let flight = currentFlight else { return }
        tickCount += 1

        if flight.hexKey == nil {
            // ✅ 더미면 기존 시뮬레이션
            simulateIncomingTrackPoint(for: flight)
        } else {
            // ✅ 실데이터면 5초마다 서버에서 최신 포인트
            if tickCount % 5 == 0, let hex = flight.hexKey {
                Task {
                    do {
                        let res = try await APIClient.shared.fetchTrack(hex: hex, minutes: 99360)
                        await MainActor.run {
                            let point = CLLocationCoordinate2D(latitude: res.lat, longitude: res.lon)
                            appendTrackPoint(point, destination: flight.destCoord)
                            currentFlight?.lastUpdate = Date()
                        }
                    } catch {
                        // 원하면 상태바에 찍기:
                        // await MainActor.run { searchStatus = "⚠️ \(error.localizedDescription)" }
                    }
                }
            }
        }

        // 남은 시간 카운트다운(더미용 느낌)
        if flight.hexKey == nil {
            if let remain = currentFlight?.remainingSeconds, remain > 0 {
                currentFlight?.remainingSeconds = max(0, remain - 1)
            }
        }

        if mapMode == .follow {
            centerOnPlane(animated: false)
        }
    }

    private func simulateIncomingTrackPoint(for flight: Flight) {
        guard let last = trackPoints.last else { return }
        let dest = flight.destCoord

        let towardLat = (dest.latitude - last.latitude) * 0.015
        let towardLon = (dest.longitude - last.longitude) * 0.015

        let driftLat = Double.random(in: -0.01...0.01) * 0.15
        let driftLon = Double.random(in: -0.01...0.01) * 0.15

        let new = CLLocationCoordinate2D(
            latitude: last.latitude + towardLat + driftLat,
            longitude: last.longitude + towardLon + driftLon
        )

        appendTrackPoint(new, destination: dest)
        currentFlight?.lastUpdate = Date()
    }

    private func appendTrackPoint(_ newPoint: CLLocationCoordinate2D, destination: CLLocationCoordinate2D) {
        if let last = trackPoints.last {
            let km = distanceKm(from: last, to: newPoint)
            if km < 0.8 { return }
        }

        trackPoints.append(newPoint)
        planeHeadingDeg = planeRotationDegrees(from: newPoint, to: destination)
    }

    private func distanceKm(from: CLLocationCoordinate2D, to: CLLocationCoordinate2D) -> Double {
        let a = CLLocation(latitude: from.latitude, longitude: from.longitude)
        let b = CLLocation(latitude: to.latitude, longitude: to.longitude)
        return a.distance(from: b) / 1000.0
    }

    // MARK: - Map controls

    private func setMapMode(_ mode: MapMode) {
        mapMode = mode
        switch mode {
        case .follow:
            centerOnPlane(animated: true)
        case .reset:
            fitToFullRoute(animated: true)
        }
    }

    private func fitToFullRoute(animated: Bool) {
        guard let f = currentFlight else { return }

        var coords: [CLLocationCoordinate2D] = []
        coords.append(f.originCoord)
        coords.append(contentsOf: trackPoints)
        coords.append(f.destCoord)

        let unique = dedupClosePoints(coords, minKm: 0.2)
        let region = regionThatFits(unique.count >= 2 ? unique : [f.originCoord, f.destCoord])
        setMapPosition(.region(region), animated: animated)
    }

    private func centerOnPlane(animated: Bool) {
        guard let p = trackPoints.last else { return }
        let region = MKCoordinateRegion(
            center: p,
            span: MKCoordinateSpan(latitudeDelta: 1.4, longitudeDelta: 1.4)
        )
        setMapPosition(.region(region), animated: animated)
    }

    private func setMapPosition(_ pos: MapCameraPosition, animated: Bool) {
        if animated {
            withAnimation(.easeInOut(duration: 0.35)) { mapPosition = pos }
        } else {
            mapPosition = pos
        }
    }

    private func regionThatFits(_ coords: [CLLocationCoordinate2D]) -> MKCoordinateRegion {
        var minLat = coords[0].latitude
        var maxLat = coords[0].latitude
        var minLon = coords[0].longitude
        var maxLon = coords[0].longitude

        for c in coords {
            minLat = min(minLat, c.latitude)
            maxLat = max(maxLat, c.latitude)
            minLon = min(minLon, c.longitude)
            maxLon = max(maxLon, c.longitude)
        }

        let center = CLLocationCoordinate2D(latitude: (minLat + maxLat) / 2,
                                            longitude: (minLon + maxLon) / 2)

        let latDelta = max(0.5, (maxLat - minLat) * 1.8)
        let lonDelta = max(0.5, (maxLon - minLon) * 1.8)

        return MKCoordinateRegion(center: center,
                                  span: MKCoordinateSpan(latitudeDelta: latDelta, longitudeDelta: lonDelta))
    }

    private func dedupClosePoints(_ coords: [CLLocationCoordinate2D], minKm: Double) -> [CLLocationCoordinate2D] {
        guard !coords.isEmpty else { return [] }
        var out: [CLLocationCoordinate2D] = [coords[0]]
        for c in coords.dropFirst() {
            if let last = out.last {
                if distanceKm(from: last, to: c) >= minKm {
                    out.append(c)
                }
            }
        }
        return out
    }

    // MARK: - Time / heading

    private func timeRemainingText(_ seconds: Int) -> String {
        if seconds <= 0 { return "arriving soon" }
        let h = seconds / 3600
        let m = (seconds % 3600) / 60
        if h > 0 { return "\(h)h \(m)m left" }
        return "\(m)m left"
    }

    private func relativeTime(from date: Date) -> String {
        let diff = Int(Date().timeIntervalSince(date))
        if diff < 3 { return "just now" }
        if diff < 60 { return "\(diff)s ago" }
        let m = diff / 60
        return "\(m)m ago"
    }

    private func planeRotationDegrees(from: CLLocationCoordinate2D, to: CLLocationCoordinate2D) -> Double {
        bearingDegrees(from: from, to: to) - 90.0
    }

    private func bearingDegrees(from: CLLocationCoordinate2D, to: CLLocationCoordinate2D) -> Double {
        let lat1 = degreesToRadians(from.latitude)
        let lon1 = degreesToRadians(from.longitude)
        let lat2 = degreesToRadians(to.latitude)
        let lon2 = degreesToRadians(to.longitude)

        let dLon = lon2 - lon1
        let y = sin(dLon) * cos(lat2)
        let x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)

        var brng = atan2(y, x)
        brng = radiansToDegrees(brng)
        brng = (brng + 360).truncatingRemainder(dividingBy: 360)
        return brng
    }

    private func degreesToRadians(_ deg: Double) -> Double { deg * .pi / 180.0 }
    private func radiansToDegrees(_ rad: Double) -> Double { rad * 180.0 / .pi }

    // hex 입력 정리 (ContentView에서도 동일하게)
    private func normalizeHex(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if s.hasPrefix("0x") { s.removeFirst(2) }
        s = s.filter { ("0"..."9").contains($0) || ("a"..."f").contains($0) }
        return s
    }

    // MARK: - Chat logic

    private func sendChat() {
        let userText = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !userText.isEmpty else { return }

        messages.append(ChatMessage(isUser: true, text: userText, time: Date()))
        chatInput = ""

        let reply = generateReply(for: userText)
        messages.append(ChatMessage(isUser: false, text: reply, time: Date()))
    }

    private func generateReply(for text: String) -> String {
        let lower = text.lowercased()

        guard let flight = currentFlight, let plane = trackPoints.last else {
            return "먼저 항공편을 검색해줘! (AA0021 또는 hex 예: 71c218)"
        }

        if lower.contains("status") || lower.contains("상태") {
            return "현재 상태는 \(flight.statusText)로 표시되고 있어요."
        }
        if lower.contains("어디") || lower.contains("where") {
            return "현재 좌표는 (\(String(format: "%.4f", plane.latitude)), \(String(format: "%.4f", plane.longitude))) 근처 상공이에요."
        }
        if lower.contains("남은") || lower.contains("remaining") {
            return "남은 시간은 \(timeRemainingText(flight.remainingSeconds)) 정도예요."
        }

        return "예: “나 지금 어디야?”, “Status 뭐야?”, “남은 시간 알려줘” 같이 물어봐줘 🙂"
    }

    // MARK: - Smoothing

    private func smoothTrack(_ pts: [CLLocationCoordinate2D], samplesPerSegment: Int = 8) -> [CLLocationCoordinate2D] {
        guard pts.count >= 3 else { return pts }

        var out: [CLLocationCoordinate2D] = []
        out.reserveCapacity(pts.count * samplesPerSegment)

        let extended = [pts.first!] + pts + [pts.last!]

        for i in 1..<(extended.count - 2) {
            let p0 = extended[i - 1]
            let p1 = extended[i]
            let p2 = extended[i + 1]
            let p3 = extended[i + 2]

            for s in 0..<samplesPerSegment {
                let t = Double(s) / Double(samplesPerSegment)
                out.append(catmullRom(p0, p1, p2, p3, t))
            }
        }

        out.append(pts.last!)
        return out
    }

    // MARK: - Great-circle (future path)

    private func greatCirclePath(from: CLLocationCoordinate2D,
                                 to: CLLocationCoordinate2D,
                                 samples: Int = 48) -> [CLLocationCoordinate2D] {
        guard samples >= 2 else { return [from, to] }

        let a = unitVector(from)
        let b = unitVector(to)

        let omega = acos(clamp(dot(a, b), -1.0, 1.0))
        if omega < 1e-8 { return [from, to] }

        var pts: [CLLocationCoordinate2D] = []
        pts.reserveCapacity(samples)

        for i in 0..<samples {
            let t = Double(i) / Double(samples - 1)

            let sinOmega = sin(omega)
            let w1 = sin((1 - t) * omega) / sinOmega
            let w2 = sin(t * omega) / sinOmega

            let x = w1 * a.x + w2 * b.x
            let y = w1 * a.y + w2 * b.y
            let z = w1 * a.z + w2 * b.z

            pts.append(coordFromUnitVector((x, y, z)))
        }

        return pts
    }

    private func unitVector(_ c: CLLocationCoordinate2D) -> (x: Double, y: Double, z: Double) {
        let lat = degreesToRadians(c.latitude)
        let lon = degreesToRadians(c.longitude)
        let x = cos(lat) * cos(lon)
        let y = cos(lat) * sin(lon)
        let z = sin(lat)
        return (x, y, z)
    }

    private func coordFromUnitVector(_ v: (Double, Double, Double)) -> CLLocationCoordinate2D {
        let mag = sqrt(v.0*v.0 + v.1*v.1 + v.2*v.2)
        let x = v.0 / mag
        let y = v.1 / mag
        let z = v.2 / mag

        let lat = atan2(z, sqrt(x*x + y*y))
        let lon = atan2(y, x)

        return CLLocationCoordinate2D(latitude: radiansToDegrees(lat),
                                      longitude: radiansToDegrees(lon))
    }

    private func dot(_ a: (x: Double, y: Double, z: Double),
                     _ b: (x: Double, y: Double, z: Double)) -> Double {
        a.x*b.x + a.y*b.y + a.z*b.z
    }

    private func clamp(_ v: Double, _ lo: Double, _ hi: Double) -> Double {
        min(max(v, lo), hi)
    }

    private func catmullRom(
        _ p0: CLLocationCoordinate2D,
        _ p1: CLLocationCoordinate2D,
        _ p2: CLLocationCoordinate2D,
        _ p3: CLLocationCoordinate2D,
        _ t: Double
    ) -> CLLocationCoordinate2D {

        let tension = 0.55

        func blend(_ a0: Double, _ a1: Double, _ a2: Double, _ a3: Double) -> Double {
            let t2 = t * t
            let t3 = t2 * t

            let m1 = (a2 - a0) * tension
            let m2 = (a3 - a1) * tension

            let h00 =  2*t3 - 3*t2 + 1
            let h10 =      t3 - 2*t2 + t
            let h01 = -2*t3 + 3*t2
            let h11 =      t3 -   t2

            return h00*a1 + h10*m1 + h01*a2 + h11*m2
        }

        let lat = blend(p0.latitude,  p1.latitude,  p2.latitude,  p3.latitude)
        let lon = blend(p0.longitude, p1.longitude, p2.longitude, p3.longitude)
        return CLLocationCoordinate2D(latitude: lat, longitude: lon)
    }
}

#Preview { ContentView() }
