import Foundation
import HealthKit
import Photos

/// Photo metadata and health samples. Both read-only, and both deliberately
/// narrow: photo *contents* never leave the device (we never load an image),
/// and health data is summarized rather than dumped sample by sample.
@MainActor
enum LifeTools {
    static let all: [Tool] = [photos, health]

    static let photos = Tool(
        name: "photos_summary",
        description: """
        Summarize photo library metadata over a date range — how many photos, \
        when, and where they were taken.

        You get dates, coordinates, and album names only. The image contents \
        are never available to you, so never describe what a photo shows or \
        guess at its subject.
        """,
        schema: schema([
            "days_back": intProp("How many days back to look. Defaults to 30.")
        ]),
        run: { input in
            let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            guard status == .authorized || status == .limited else {
                return EventKitAccess.denied("the photo library", toggle: "Photos")
            }
            let days = max(1, input["days_back"] as? Int ?? 30)
            let since = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()

            let options = PHFetchOptions()
            options.predicate = NSPredicate(format: "creationDate >= %@", since as NSDate)
            options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            let assets = PHAsset.fetchAssets(with: options)

            guard assets.count > 0 else { return "No photos in the last \(days) day(s)." }

            var located = 0
            var byDay: [String: Int] = [:]
            var videos = 0
            let dayFormat = DateFormatter()
            dayFormat.dateFormat = "yyyy-MM-dd"

            assets.enumerateObjects { asset, _, _ in
                if asset.location != nil { located += 1 }
                if asset.mediaType == .video { videos += 1 }
                if let created = asset.creationDate {
                    byDay[dayFormat.string(from: created), default: 0] += 1
                }
            }
            let busiest = byDay.sorted { $0.value > $1.value }.prefix(5)
                .map { "\($0.key): \($0.value)" }.joined(separator: ", ")

            return """
            \(assets.count) item(s) in the last \(days) day(s) — \
            \(assets.count - videos) photo(s), \(videos) video(s).
            \(located) have location data.
            Busiest days: \(busiest.isEmpty ? "n/a" : busiest)
            (Metadata only — image contents are not readable.)
            """
        }
    )

    static let health = Tool(
        name: "health_summary",
        description: """
        Summarize health metrics — steps, sleep, and workouts — over recent days.

        These are the user's own medical figures. Report them plainly. Do not \
        diagnose, do not infer conditions, and if a question needs clinical \
        interpretation say so and suggest a clinician rather than answering it.
        """,
        schema: schema([
            "days_back": intProp("How many days back to summarize. Defaults to 7.")
        ]),
        run: { input in
            guard HKHealthStore.isHealthDataAvailable() else {
                return "Health data is not available on this device."
            }
            let store = HKHealthStore()
            let steps = HKQuantityType(.stepCount)
            let sleep = HKCategoryType(.sleepAnalysis)
            let types: Set<HKObjectType> = [steps, sleep, HKObjectType.workoutType()]

            do {
                try await store.requestAuthorization(toShare: [], read: types)
            } catch {
                return EventKitAccess.denied("health data", toggle: "Health")
            }

            let days = max(1, input["days_back"] as? Int ?? 7)
            let since = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
            let window = HKQuery.predicateForSamples(withStart: since, end: Date())

            let totalSteps: Double = await withCheckedContinuation { continuation in
                let query = HKStatisticsQuery(
                    quantityType: steps, quantitySamplePredicate: window,
                    options: .cumulativeSum
                ) { _, stats, _ in
                    continuation.resume(
                        returning: stats?.sumQuantity()?.doubleValue(for: .count()) ?? 0)
                }
                store.execute(query)
            }

            let workouts: Int = await withCheckedContinuation { continuation in
                let query = HKSampleQuery(
                    sampleType: HKObjectType.workoutType(), predicate: window,
                    limit: HKObjectQueryNoLimit, sortDescriptors: nil
                ) { _, samples, _ in
                    continuation.resume(returning: samples?.count ?? 0)
                }
                store.execute(query)
            }

            let asleep: Double = await withCheckedContinuation { continuation in
                let query = HKSampleQuery(
                    sampleType: sleep, predicate: window,
                    limit: HKObjectQueryNoLimit, sortDescriptors: nil
                ) { _, samples, _ in
                    let seconds = (samples as? [HKCategorySample] ?? [])
                        .filter { $0.value != HKCategoryValueSleepAnalysis.inBed.rawValue }
                        .reduce(0.0) { $0 + $1.endDate.timeIntervalSince($1.startDate) }
                    continuation.resume(returning: seconds / 3600)
                }
                store.execute(query)
            }

            if totalSteps == 0 && workouts == 0 && asleep == 0 {
                return """
                No health samples in the last \(days) day(s). This can mean \
                permission was granted but read access to these specific types \
                was declined — iOS reports that identically to having no data, \
                so it cannot be distinguished from here.
                """
            }
            return """
            Last \(days) day(s):
            steps: \(Int(totalSteps)) total, ~\(Int(totalSteps) / days)/day
            sleep: \(String(format: "%.1f", asleep))h total, ~\(String(format: "%.1f", asleep / Double(days)))h/night
            workouts: \(workouts)
            """
        }
    )
}
