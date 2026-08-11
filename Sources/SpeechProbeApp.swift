//
//  SpeechProbeApp.swift
//  SpeechProbe
//
//  Defensive QA / fuzzing harness for on-device audio decode + speech
//  recognition robustness testing. Runs entirely on the user's own
//  dev-signed sideloaded device. No third-party dependencies.
//
//  Frameworks: SwiftUI, AVFoundation, Speech, Foundation only.
//

import SwiftUI
import AVFoundation
import Speech
import os.log
import Foundation

// MARK: - Logging

enum ProbeLog {
    static let subsystem = "com.khalifa.speechprobe"
    static let decode = OSLog(subsystem: subsystem, category: "decode")
    static let asr = OSLog(subsystem: subsystem, category: "asr")
    static let live = OSLog(subsystem: subsystem, category: "live")
    static let perm = OSLog(subsystem: subsystem, category: "perm")
    static let loop = OSLog(subsystem: subsystem, category: "loop")

    static func log(_ log: OSLog, _ msg: String) {
        os_log("%{public}@", log: log, type: .default, msg)
    }
}

// MARK: - Shared UI Log Sink

final class UILogSink: ObservableObject {
    @Published var lines: [String] = []
    private let maxLines = 500
    private let queue = DispatchQueue(label: "com.khalifa.speechprobe.uilogsink")

    func append(_ s: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let full = "[\(timestamp)] \(s)"
        queue.async {
            DispatchQueue.main.async {
                self.lines.append(full)
                if self.lines.count > self.maxLines {
                    self.lines.removeFirst(self.lines.count - self.maxLines)
                }
            }
        }
    }
}

// MARK: - Result bookkeeping

struct FileResult {
    let name: String
    var decodeStatus: String = "PENDING"
    var playerStatus: String = "PENDING"
    var asrStatus: String = "PENDING"
    var finished: Bool = false
}

// MARK: - Fuzz Engine

final class FuzzEngine: NSObject, ObservableObject {

    @Published var statusText: String = "Idle"
    @Published var isRunning: Bool = false
    @Published var liveRunning: Bool = false
    @Published var processedCount: Int = 0
    @Published var totalCount: Int = 0

    let uiLog = UILogSink()

    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var currentAVPlayer: AVAudioPlayer?
    private var fileQueue: [URL] = []
    private var processedFiles: Set<String> = []
    private var isProcessingQueue = false
    private let engineQueue = DispatchQueue(label: "com.khalifa.speechprobe.engine", qos: .utility)

    private var docsInURL: URL {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let inDir = docs.appendingPathComponent("In", isDirectory: true)
        if !fm.fileExists(atPath: inDir.path) {
            try? fm.createDirectory(at: inDir, withIntermediateDirectories: true)
        }
        return inDir
    }

    private var docsRootURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // Live capture
    private let audioEngine = AVAudioEngine()
    private var liveRecognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var liveRecognitionTask: SFSpeechRecognitionTask?

    private var fsWatcher: DispatchSourceFileSystemObject?
    private var watchedFD: Int32 = -1

    // MARK: Permissions

    func requestPermissionsAndBootstrap() {
        ProbeLog.log(ProbeLog.perm, "REQUEST_SPEECH_AUTH_START")
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            let statusStr = self?.describeSpeechAuth(status) ?? "unknown"
            ProbeLog.log(ProbeLog.perm, "SPEECH_AUTH_RESULT \(statusStr)")
            self?.uiLog.append("Speech auth: \(statusStr)")

            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                ProbeLog.log(ProbeLog.perm, "MIC_AUTH_RESULT \(granted)")
                self?.uiLog.append("Mic auth granted: \(granted)")

                DispatchQueue.main.async {
                    self?.bootstrapExistingFiles()
                    self?.startWatchingInDirectory()
                }
            }
        }
    }

    private func describeSpeechAuth(_ s: SFSpeechRecognizerAuthorizationStatus) -> String {
        switch s {
        case .authorized: return "authorized"
        case .denied: return "denied"
        case .restricted: return "restricted"
        case .notDetermined: return "notDetermined"
        @unknown default: return "unknown"
        }
    }

    // MARK: Bootstrap — process files already present at launch

    func bootstrapExistingFiles() {
        let fm = FileManager.default
        var urls: [URL] = []

        for dir in [docsInURL, docsRootURL] {
            if let items = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) {
                for u in items where !u.hasDirectoryPath {
                    urls.append(u)
                }
            }
        }

        ProbeLog.log(ProbeLog.loop, "BOOTSTRAP_FOUND_FILES count=\(urls.count)")
        uiLog.append("Bootstrap: found \(urls.count) file(s) already in Documents.")

        enqueue(urls: urls)
    }

    // MARK: Directory watching (new file arrivals)

    private func startWatchingInDirectory() {
        let path = docsInURL.path
        let fd = open(path, O_EVTONLY)
        guard fd >= 0 else {
            uiLog.append("Failed to open In/ for watching (fd=\(fd)).")
            return
        }
        watchedFD = fd

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: [.write, .extend, .rename, .delete],
            queue: engineQueue
        )
        source.setEventHandler { [weak self] in
            self?.rescanInDirectory()
        }
        source.setCancelHandler {
            close(fd)
        }
        source.resume()
        fsWatcher = source

        ProbeLog.log(ProbeLog.loop, "WATCH_STARTED path=\(path)")
        uiLog.append("Watching \(path) for new files.")
    }

    private func rescanInDirectory() {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: docsInURL, includingPropertiesForKeys: nil) else { return }
        let newOnes = items.filter { !$0.hasDirectoryPath && !processedFiles.contains($0.path) }
        if !newOnes.isEmpty {
            ProbeLog.log(ProbeLog.loop, "WATCH_NEW_FILES count=\(newOnes.count)")
            enqueue(urls: newOnes)
        }
    }

    // MARK: Queue management

    func enqueue(urls: [URL]) {
        engineQueue.async { [weak self] in
            guard let self = self else { return }
            let fresh = urls.filter { !self.processedFiles.contains($0.path) }
            self.fileQueue.append(contentsOf: fresh)
            DispatchQueue.main.async {
                self.totalCount += fresh.count
            }
            self.pumpQueue()
        }
    }

    func manualRun() {
        DispatchQueue.main.async {
            self.statusText = "Running"
            self.isRunning = true
        }
        bootstrapExistingFiles()
    }

    private func pumpQueue() {
        engineQueue.async { [weak self] in
            guard let self = self else { return }
            guard !self.isProcessingQueue else { return }
            self.isProcessingQueue = true
            self.drainQueueLoop()
        }
    }

    private func drainQueueLoop() {
        guard let next = fileQueue.first else {
            isProcessingQueue = false
            DispatchQueue.main.async {
                self.statusText = "Idle (queue empty)"
                self.isRunning = false
            }
            return
        }
        fileQueue.removeFirst()
        processedFiles.insert(next.path)

        DispatchQueue.main.async {
            self.statusText = "Processing \(next.lastPathComponent)"
            self.isRunning = true
        }

        processOneFile(next) { [weak self] in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.processedCount += 1
            }
            self.engineQueue.async {
                self.drainQueueLoop()
            }
        }
    }

    // MARK: Per-file pipeline (with hang guards)

    private func processOneFile(_ url: URL, completion: @escaping () -> Void) {
        let name = url.lastPathComponent
        ProbeLog.log(ProbeLog.decode, "DECODE_START \(name)")
        uiLog.append("--- \(name) ---")

        // Guard the WHOLE per-file pipeline with a watchdog so one file
        // can never wedge the loop for more than ~65s total.
        let overallGuard = HangGuard(seconds: 65) { [weak self] in
            ProbeLog.log(ProbeLog.loop, "FILE_HANG \(name)")
            self?.uiLog.append("HANG on \(name) — watchdog forcing continue.")
        }

        runDecodeStage(url: url, name: name) { [weak self] in
            guard let self = self else { return }
            self.runPlayerStage(url: url, name: name) {
                self.runASRStage(url: url, name: name) {
                    overallGuard.cancel()
                    ProbeLog.log(ProbeLog.loop, "DONE \(name) COMPLETE")
                    self.uiLog.append("DONE \(name)")
                    completion()
                }
            }
        }

        overallGuard.onFire = {
            // Fired only if pipeline never called back in time.
            completion()
        }
    }

    // Stage 1: AVAudioFile decode
    private func runDecodeStage(url: URL, name: String, next: @escaping () -> Void) {
        engineQueue.async {
            do {
                let file = try AVAudioFile(forReading: url)
                let format = file.processingFormat
                let frameCount = AVAudioFrameCount(min(file.length, 48_000 * 30))
                if frameCount > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(frameCount, 1)) {
                    do {
                        try file.read(into: buffer)
                        ProbeLog.log(ProbeLog.decode, "DECODE_OK \(name) frames=\(buffer.frameLength) sr=\(format.sampleRate) ch=\(format.channelCount)")
                        self.uiLog.append("DECODE_OK \(name) frames=\(buffer.frameLength)")
                    } catch {
                        ProbeLog.log(ProbeLog.decode, "DECODE_READ_ERROR \(name): \(error.localizedDescription)")
                        self.uiLog.append("DECODE_READ_ERROR \(name): \(error.localizedDescription)")
                    }
                } else {
                    ProbeLog.log(ProbeLog.decode, "DECODE_EMPTY_OR_NO_BUFFER \(name) length=\(file.length)")
                    self.uiLog.append("DECODE_EMPTY \(name)")
                }
            } catch {
                ProbeLog.log(ProbeLog.decode, "DECODE_OPEN_ERROR \(name): \(error.localizedDescription)")
                self.uiLog.append("DECODE_OPEN_ERROR \(name): \(error.localizedDescription)")
            }
            DispatchQueue.main.async { next() }
        }
    }

    // Stage 2: AVAudioPlayer playback (0.5s)
    private func runPlayerStage(url: URL, name: String, next: @escaping () -> Void) {
        engineQueue.async {
            do {
                let session = AVAudioSession.sharedInstance()
                try? session.setCategory(.playback, mode: .default, options: [])
                try? session.setActive(true, options: [])

                let player = try AVAudioPlayer(contentsOf: url)
                self.currentAVPlayer = player
                let prepared = player.prepareToPlay()
                ProbeLog.log(ProbeLog.decode, "PLAYER_PREPARE \(name) ok=\(prepared) dur=\(player.duration)")
                let started = player.play()
                ProbeLog.log(ProbeLog.decode, "PLAYER_PLAY_START \(name) started=\(started)")

                let stopGuard = HangGuard(seconds: 3) { [weak self] in
                    ProbeLog.log(ProbeLog.decode, "PLAYER_STOP_HANG \(name)")
                    self?.uiLog.append("PLAYER_STOP_HANG \(name)")
                }

                DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) {
                    player.stop()
                    stopGuard.cancel()
                    ProbeLog.log(ProbeLog.decode, "PLAYER_STOPPED \(name)")
                    self.currentAVPlayer = nil
                    DispatchQueue.main.async { next() }
                }
            } catch {
                ProbeLog.log(ProbeLog.decode, "PLAYER_ERROR \(name): \(error.localizedDescription)")
                self.uiLog.append("PLAYER_ERROR \(name): \(error.localizedDescription)")
                DispatchQueue.main.async { next() }
            }
        }
    }

    // Stage 3: SFSpeechURLRecognitionRequest with 60s hang guard
    private func runASRStage(url: URL, name: String, next: @escaping () -> Void) {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            ProbeLog.log(ProbeLog.asr, "ASR_UNAVAILABLE \(name)")
            uiLog.append("ASR_UNAVAILABLE \(name)")
            next()
            return
        }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            ProbeLog.log(ProbeLog.asr, "ASR_SKIPPED_NOAUTH \(name)")
            uiLog.append("ASR_SKIPPED_NOAUTH \(name)")
            next()
            return
        }

        ProbeLog.log(ProbeLog.asr, "ASR_START \(name)")

        let request = SFSpeechURLRecognitionRequest(url: url)
        request.shouldReportPartialResults = false

        var completedOnce = false
        let completeLock = NSLock()
        var task: SFSpeechRecognitionTask?

        func finishOnce(_ reason: String) {
            completeLock.lock()
            if completedOnce {
                completeLock.unlock()
                return
            }
            completedOnce = true
            completeLock.unlock()
            ProbeLog.log(ProbeLog.asr, reason)
            uiLog.append(reason)
            next()
        }

        let hangGuard = HangGuard(seconds: 60) {
            ProbeLog.log(ProbeLog.asr, "ASR_HANG \(name)")
            self.uiLog.append("ASR_HANG \(name)")
            task?.cancel()
            finishOnce("ASR_HANG_CANCELLED \(name)")
        }

        task = recognizer.recognitionTask(with: request) { result, error in
            if let error = error {
                hangGuard.cancel()
                finishOnce("ASR_ERROR \(name): \(error.localizedDescription)")
                return
            }
            if let result = result, result.isFinal {
                hangGuard.cancel()
                let text = result.bestTranscription.formattedString
                finishOnce("ASR_RESULT \(name): \"\(text)\"")
            }
        }
    }

    // MARK: Live PCM path (separate from file fuzzing)

    func startLiveCapture() {
        guard !liveRunning else { return }
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            uiLog.append("LIVE_UNAVAILABLE: recognizer not available.")
            ProbeLog.log(ProbeLog.live, "LIVE_UNAVAILABLE")
            return
        }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            uiLog.append("LIVE_SKIPPED_NOAUTH")
            ProbeLog.log(ProbeLog.live, "LIVE_SKIPPED_NOAUTH")
            return
        }

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: [])
            try session.setActive(true, options: [])
        } catch {
            uiLog.append("LIVE_SESSION_ERROR: \(error.localizedDescription)")
            ProbeLog.log(ProbeLog.live, "LIVE_SESSION_ERROR \(error.localizedDescription)")
            return
        }

        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        liveRecognitionRequest = req

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        ProbeLog.log(ProbeLog.live, "LIVE_START fmt_sr=\(recordingFormat.sampleRate) ch=\(recordingFormat.channelCount)")

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: recordingFormat) { [weak self] buffer, _ in
            self?.liveRecognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            liveRunning = true
            uiLog.append("LIVE capture started.")
        } catch {
            uiLog.append("LIVE_ENGINE_START_ERROR: \(error.localizedDescription)")
            ProbeLog.log(ProbeLog.live, "LIVE_ENGINE_START_ERROR \(error.localizedDescription)")
            inputNode.removeTap(onBus: 0)
            return
        }

        liveRecognitionTask = recognizer.recognitionTask(with: req) { [weak self] result, error in
            if let error = error {
                ProbeLog.log(ProbeLog.live, "LIVE_ERROR \(error.localizedDescription)")
                self?.uiLog.append("LIVE_ERROR: \(error.localizedDescription)")
                return
            }
            if let result = result {
                let text = result.bestTranscription.formattedString
                ProbeLog.log(ProbeLog.live, "LIVE_PARTIAL len=\(text.count)")
            }
        }
    }

    func stopLiveCapture() {
        guard liveRunning else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        liveRecognitionRequest?.endAudio()
        liveRecognitionTask?.cancel()
        liveRecognitionRequest = nil
        liveRecognitionTask = nil
        liveRunning = false
        ProbeLog.log(ProbeLog.live, "LIVE_STOPPED")
        uiLog.append("LIVE capture stopped.")

        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }
}

// MARK: - HangGuard: deadline-based watchdog

final class HangGuard {
    private var fired = false
    private let lock = NSLock()
    private let workItem: DispatchWorkItem
    var onFire: (() -> Void)?

    init(seconds: TimeInterval, onFireImmediate: @escaping () -> Void) {
        let item = DispatchWorkItem { }
        self.workItem = item
        let localFire: () -> Void = { [weak self] in
            guard let self = self else { return }
            self.lock.lock()
            if self.fired {
                self.lock.unlock()
                return
            }
            self.fired = true
            self.lock.unlock()
            onFireImmediate()
            self.onFire?()
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + seconds, execute: localFire)
    }

    func cancel() {
        lock.lock()
        fired = true
        lock.unlock()
    }
}

// MARK: - SwiftUI Root

@main
struct SpeechProbeApp: App {
    @StateObject private var engine = FuzzEngine()

    var body: some Scene {
        WindowGroup {
            RootView(engine: engine)
                .onAppear {
                    engine.requestPermissionsAndBootstrap()
                }
        }
    }
}

struct RootView: View {
    @ObservedObject var engine: FuzzEngine

    var body: some View {
        VStack(spacing: 12) {
            Text("SpeechProbe")
                .font(.title2)
                .bold()

            Text(engine.statusText)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Text("\(engine.processedCount) / \(engine.totalCount) processed")
                .font(.caption)
                .foregroundColor(.secondary)

            HStack(spacing: 16) {
                Button("RUN / RESCAN") {
                    engine.manualRun()
                }
                .buttonStyle(.borderedProminent)

                Button(engine.liveRunning ? "STOP LIVE" : "START LIVE") {
                    if engine.liveRunning {
                        engine.stopLiveCapture()
                    } else {
                        engine.startLiveCapture()
                    }
                }
                .buttonStyle(.bordered)
            }

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(engine.uiLog.lines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: 10, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.horizontal, 8)
            }
        }
        .padding()
    }
}
