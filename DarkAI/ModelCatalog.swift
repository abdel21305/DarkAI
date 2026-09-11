import Foundation
import Combine
import UIKit

// MARK: - Catalog

/// A model the app offers to fetch for the user.
///
/// The catalog is a fixed, developer-curated list — there is deliberately no "paste a URL" field.
/// Two reasons: an arbitrary-URL downloader turns the app into a general-purpose file fetcher,
/// which App Review treats as a way to bypass the review of what the app actually ships; and
/// every entry here has had its license checked, which is not something the app can do for an
/// arbitrary URL. Users who want a specific model can still side-load it into the app's Files
/// folder or use the document picker — that path is theirs, not something the app arranges.
enum ModelKind: String, Codable {
    /// Chat/instruct weights driven by llama.cpp. Land in `AppFiles.models`.
    case chat
    /// Diffusion checkpoints driven by stable-diffusion.cpp. Land in `AppFiles.diffusionModels`.
    case diffusion
    /// Core ML model packages driven by `CoreMLRunner`, executing on the ANE/GPU/CPU via Core
    /// ML. Land in `AppFiles.coreMLModels`.
    case coreML
}

/// One file belonging to a `.coreML` model's install directory.
///
/// A Core ML model here is never a single downloadable file the way a GGUF is — the simplest case
/// (`OpenELM-270M-Instruct`) is a 3-file `.mlpackage` bundle; a chunked ANE pipeline
/// (`Llama-3.2-1B-Instruct`) is 30+ files across several already-compiled `.mlmodelc` bundles. This
/// is the complete, ordered manifest for one catalog model — see
/// `ModelDownloadManager.downloadCoreMLFiles` for how it's fetched (a queue of background
/// download tasks, one file at a time, not the single-task path every other model kind uses).
/// `nonisolated`: same reasoning as `CatalogModel` right below — a plain value type with no
/// shared mutable state, whose synthesized `Hashable`/`Equatable` conformance otherwise defaults
/// to main-actor-isolated and can't be used from `verify(fileAt:against:)`'s nonisolated context.
nonisolated struct CoreMLPackageFile: Hashable {
    let url: URL
    /// Path relative to the installed model's root directory, e.g. `"Manifest.json"` or
    /// `"Llama-3.2-1B-Instruct_chunk1.mlmodelc/weights/weight.bin"`.
    let relativePath: String
    let byteSize: Int64
}

/// `nonisolated`: a plain value type describing static catalog metadata, with no shared mutable
/// state — `verify(fileAt:against:)` (deliberately `nonisolated` so download verification runs
/// off the main actor) reads `fileName`/`sizeDescription` on this type, which default to
/// main-actor-isolated like everything else in this module otherwise.
nonisolated struct CatalogModel: Identifiable, Hashable {
    let id: String
    let kind: ModelKind
    let displayName: String
    let publisher: String
    /// Exact `Content-Length` of the remote file, verified against what actually lands on disk.
    let byteSize: Int64
    let parameterCount: String
    let quantization: String
    let license: String
    /// Extra notice the license obliges the app to display, if any.
    let attribution: String?
    let summary: String
    let url: URL
    /// Approximate peak RAM to run it, weights plus a default context. Drives the
    /// "may not run on this device" warning before a user spends a gigabyte of cellular data.
    let approxRuntimeGB: Double
    /// Physical RAM below which this model is not worth recommending.
    ///
    /// Derived by running each model's real geometry — layer count, KV head count, head dims and
    /// trained context, all read from its GGUF header — through `LlamaRunner.planOffload` and
    /// `safeContextTokens` at each iPhone memory tier, and taking the lowest tier that still
    /// yields a usable context window rather than merely loading. A model that loads with 512
    /// tokens is not one to recommend.
    let minimumRAMGB: Double
    /// The earliest iPhone at `minimumRAMGB`, for a human-readable tag. iOS 17 is the app's
    /// floor, so the oldest hardware in scope is the 4 GB generation (iPhone 11, SE 3rd gen).
    let minimumDevice: String
    /// Overrides `fileName` when `url` doesn't end in the real filename. Hugging Face URLs are
    /// self-describing (`.../resolve/main/some-model.gguf`), but Civitai's download endpoint is
    /// `/api/download/models/{versionId}` — the actual filename only ever appears in the
    /// response's `Content-Disposition` header, never in the URL itself. Without this, a
    /// Civitai-sourced model would install under a literal numeric ID with no extension: hidden
    /// from every extension-filtered model list in the app and unloadable.
    let fileNameOverride: String?
    /// Non-empty only for `kind == .coreML` — the model's complete file manifest (see
    /// `CoreMLPackageFile`). `byteSize` above is the sum of every entry's size for this case,
    /// computed once at catalog-definition time, and `url` is simply the first file's URL —
    /// neither is a real download target on its own the way they are for `.chat`/`.diffusion`.
    let coreMLFiles: [CoreMLPackageFile]

    init(id: String, kind: ModelKind, displayName: String, publisher: String, byteSize: Int64,
         parameterCount: String, quantization: String, license: String, attribution: String?,
         summary: String, url: URL, approxRuntimeGB: Double,
         minimumRAMGB: Double, minimumDevice: String, fileNameOverride: String? = nil,
         coreMLFiles: [CoreMLPackageFile] = []) {
        self.id = id
        self.kind = kind
        self.displayName = displayName
        self.publisher = publisher
        self.byteSize = byteSize
        self.parameterCount = parameterCount
        self.quantization = quantization
        self.license = license
        self.attribution = attribution
        self.summary = summary
        self.url = url
        self.approxRuntimeGB = approxRuntimeGB
        self.minimumRAMGB = minimumRAMGB
        self.minimumDevice = minimumDevice
        self.fileNameOverride = fileNameOverride
        self.coreMLFiles = coreMLFiles
    }

    var fileName: String { fileNameOverride ?? url.lastPathComponent }
    var sizeGB: Double { Double(byteSize) / (1024 * 1024 * 1024) }
    var sizeDescription: String { String(format: "%.2f GB", sizeGB) }
}

enum ModelCatalog {

    /// Curated starter models. All are instruction-tuned and all are redistributable under a
    /// license that permits this use.
    ///
    /// Sizes are the exact byte counts served by the host — they are checked after download, so
    /// if a host ever republishes a file at a different size the verification fails loudly
    /// instead of leaving a truncated model that only misbehaves at inference time.
    ///
    /// One entry is a mixture-of-experts model, and it is listed on its merits rather than for
    /// any memory advantage. There was supposed to be one: pin the routed experts to mmap and a
    /// sparse model needs only its dense remainder resident. That does not work on Metal —
    /// llama.cpp gives each device a single mapping spanning its first tensor to its last, and
    /// because MoE files interleave expert and attention tensors, pinning experts still forces
    /// the whole file to be mapped (see `LlamaRunner.planOffload` for the measurements). So a
    /// sparse model here costs what a dense one of the same size costs, and is judged the same
    /// way. What it still offers is quality per *compute*: only a fraction of its weights run
    /// for each token, so a 7B-class model answers at roughly 1B-class speed.
    ///
    static let chatModels: [CatalogModel] = [
        CatalogModel(
            id: "llama-3.2-1b-instruct-q4km",
            kind: .chat,
            displayName: "Llama 3.2 1B Instruct",
            publisher: "Meta (GGUF build by bartowski)",
            byteSize: 807_694_464,
            parameterCount: "1B",
            quantization: "Q4_K_M",
            license: "Llama 3.2 Community License",
            attribution: "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            summary: "Smallest and fastest. A good first download and the least demanding on memory.",
            url: URL(string: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf")!,
            approxRuntimeGB: 1.6,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
        ),
        CatalogModel(
            id: "qwen2.5-1.5b-instruct-q4km",
            kind: .chat,
            displayName: "Qwen2.5 1.5B Instruct",
            publisher: "Alibaba Cloud (GGUF build by bartowski)",
            byteSize: 986_048_768,
            parameterCount: "1.5B",
            quantization: "Q4_K_M",
            license: "Apache 2.0",
            attribution: nil,
            summary: "Stronger at reasoning and code than the 1B models, still comfortable on most devices.",
            url: URL(string: "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")!,
            approxRuntimeGB: 1.9,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
        ),
        CatalogModel(
            id: "smollm2-1.7b-instruct-q4km",
            kind: .chat,
            displayName: "SmolLM2 1.7B Instruct",
            publisher: "Hugging Face (GGUF build by bartowski)",
            byteSize: 1_055_609_824,
            parameterCount: "1.7B",
            quantization: "Q4_K_M",
            license: "Apache 2.0",
            attribution: nil,
            summary: "Conversational and well-behaved on short prompts. Trained on openly documented data.",
            url: URL(string: "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")!,
            approxRuntimeGB: 2.0,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
        ),
        CatalogModel(
            id: "llama-3.2-3b-instruct-q4km",
            kind: .chat,
            displayName: "Llama 3.2 3B Instruct",
            publisher: "Meta (GGUF build by hugging-quants)",
            byteSize: 2_019_373_920,
            parameterCount: "3B",
            quantization: "Q4_K_M",
            license: "Llama 3.2 Community License",
            attribution: "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            summary: "Strong at reasoning and long answers, and the best balance of quality against download size here. Needs a device with plenty of memory.",
            url: URL(string: "https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf")!,
            approxRuntimeGB: 3.4,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 13 Pro / 14 or newer"
        ),

        CatalogModel(
            id: "olmoe-1b-7b-0924-instruct-q4km",
            kind: .chat,
            displayName: "OLMoE 1B-7B Instruct",
            publisher: "Ai2",
            byteSize: 4_213_512_672,
            parameterCount: "7B total, 1B active",
            quantization: "Q4_K_M",
            license: "Apache 2.0",
            attribution: nil,
            // 16 blocks, 64 experts each, 8 active per token. Resident cost is the whole file
            // like any other model, so this is sized as such — the earlier 1.6 GB figure assumed
            // an expert-pinning saving that Metal does not permit.
            summary: "A 7B model that only uses about an eighth of itself for each word, so it answers about as fast as a 1B while knowing more.",
            url: URL(string: "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf")!,
            approxRuntimeGB: 4.6,
            minimumRAMGB: 8.0,
            minimumDevice: " minimum iPhone 15 Pro / 16 or newer"
        ),
        CatalogModel(
            id: "lfm2.5-8b-a1b-instruct-q4km",
            kind: .chat,
            displayName: "LFM2.5 8B-A1B",
            publisher: "Liquid AI",
            byteSize: 5_155_564_768,
            parameterCount: "8.3B total, 1.5B active",
            quantization: "Q4_K_M",
            license: "LFM Open License v1.0",
            attribution: "LFM2.5-8B-A1B by Liquid AI, licensed under the LFM Open License v1.0.",
            // Replaces the old dense Llama 3 8B entry, which a background personality-analysis
            // call could race for the model — see the fix in LlamaRunner.generateStream — and
            // which, being dense, offered no way to avoid needing its full weight size resident
            // in memory. This model routes each token through only 4 of 32 experts per
            // mixture-of-experts block, so replies are markedly faster than a dense model this
            // size. That speed does NOT come with a smaller memory footprint, though: per-expert
            // GPU/CPU splitting was tried for a mixture-of-experts model on this engine and found
            // to not work reliably on Metal (see the comment on expert pinning in
            // `LlamaRunner.planOffload`), so this is still sized as needing its full file size
            // resident, exactly like every other model here.
            summary: "A mixture-of-experts model that only routes each word through a fraction of its weights; the power of an 8B model without the memory demand but replies come noticeably slower than other models.",
            url: URL(string: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf")!,
            approxRuntimeGB: 5.5,
            minimumRAMGB: 12.0,
            minimumDevice: "iPhone 17 Pro or newer"
        )
    ]

    static let diffusionModels: [CatalogModel] = [
        CatalogModel(
            id: "sd-1.5-emaonly-q4_0",
            kind: .diffusion,
            displayName: "Stable Diffusion 1.5",
            publisher: "Runway / Stability AI (GGUF build by Second State)",
            byteSize: 1_566_768_416,
            parameterCount: "860M UNet",
            quantization: "Q4_0",
            license: "CreativeML OpenRAIL-M",
            attribution: "Stable Diffusion v1-5 is licensed under the CreativeML OpenRAIL-M license, which prohibits generating illegal or harmful content.",
            summary: "Generates 512×512 images. The EMA-only pruned build — the smallest complete Stable Diffusion checkpoint, and the most forgiving on memory.",
            url: URL(string: "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf")!,
            approxRuntimeGB: 3.0,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
        ),
        
        CatalogModel(
            id: "realistic-vision-v6-b1-hyper-vae",
            kind: .diffusion,
            displayName: "Realistic Vision V6.0 B1",
            publisher: "SG_161222 (Civitai)",
            byteSize: 2_132_625_894,
            parameterCount: "860M UNet",
            quantization: "FP16 (pruned)",
            license: "Civitai model license — credit required",
            attribution: "Realistic Vision V6.0 B1 by SG_161222 (civitai.com/models/4201). Credit to the creator is required by the model's license. Commercial use of generated images is permitted; the checkpoint file itself may not be sold or repackaged under a different license.",
            summary: "Photorealistic portraits and scenes. One of the most widely used SD 1.5 checkpoints on Civitai, tuned for good results in as few as 4–6 steps and with a VAE already baked in.",
            url: URL(string: "https://civitai.com/api/download/models/501240?fileId=418901")!,
            approxRuntimeGB: 3.5,
            minimumRAMGB: 8.0,
            minimumDevice: "minimum iPhone 15 Pro / 16 or newer",
            fileNameOverride: "realisticVisionV60B1_v51HyperVAE.safetensors"
        )
    ]

    /// Core ML models, executed via `CoreMLRunner` rather than llama.cpp — both entries below are
    /// real, stateful, sliding-window chat models (`ChunkedPipelineCoreMLEngine`), not the fixed
    /// 128-token demo export this catalog used to also offer (OpenELM-270M-Instruct, removed:
    /// too short a context window and too small a model to be useful for anything beyond proving
    /// the ANE pipeline worked at all).
    static let coreMLModels: [CatalogModel] = {
        // Llama 3.2 1B Instruct, converted for the ANE by the coreml-llm-cli project (see
        // ChunkedPipelineCoreMLEngine's doc comment). Six chunked, already-*compiled* `.mlmodelc`
        // transformer blocks plus a small KV-cache-shift model — deliberately excludes the
        // upstream repo's `logit-processor.mlmodelc` (argmax-only, no learned weights), since this
        // app samples with its own temperature/top-p sampler directly instead. Byte sizes fetched
        // directly from the Hugging Face API this session, not estimated.
        let llamaBase = "https://huggingface.co/smpanaro/Llama-3.2-1B-Instruct-CoreML/resolve/main/"
        let llamaFiles: [CoreMLPackageFile] = [
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk1.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk1.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk1.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk1.mlmodelc/coremldata.bin", byteSize: 407),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk1.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-1B-Instruct_chunk1.mlmodelc/metadata.json", byteSize: 2891),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk1.mlmodelc/model.mil")!, relativePath: "Llama-3.2-1B-Instruct_chunk1.mlmodelc/model.mil", byteSize: 10344),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk1.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk1.mlmodelc/weights/weight.bin", byteSize: 525599104),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk2.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk2.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk2.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk2.mlmodelc/coremldata.bin", byteSize: 931),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk2.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-1B-Instruct_chunk2.mlmodelc/metadata.json", byteSize: 7803),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk2.mlmodelc/model.mil")!, relativePath: "Llama-3.2-1B-Instruct_chunk2.mlmodelc/model.mil", byteSize: 385989),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk2.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk2.mlmodelc/weights/weight.bin", byteSize: 486575936),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk3.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk3.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk3.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk3.mlmodelc/coremldata.bin", byteSize: 931),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk3.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-1B-Instruct_chunk3.mlmodelc/metadata.json", byteSize: 7803),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk3.mlmodelc/model.mil")!, relativePath: "Llama-3.2-1B-Instruct_chunk3.mlmodelc/model.mil", byteSize: 385989),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk3.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk3.mlmodelc/weights/weight.bin", byteSize: 486575936),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk4.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk4.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk4.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk4.mlmodelc/coremldata.bin", byteSize: 931),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk4.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-1B-Instruct_chunk4.mlmodelc/metadata.json", byteSize: 7803),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk4.mlmodelc/model.mil")!, relativePath: "Llama-3.2-1B-Instruct_chunk4.mlmodelc/model.mil", byteSize: 385989),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk4.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk4.mlmodelc/weights/weight.bin", byteSize: 486575936),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk5.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk5.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk5.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk5.mlmodelc/coremldata.bin", byteSize: 931),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk5.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-1B-Instruct_chunk5.mlmodelc/metadata.json", byteSize: 7803),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk5.mlmodelc/model.mil")!, relativePath: "Llama-3.2-1B-Instruct_chunk5.mlmodelc/model.mil", byteSize: 385989),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk5.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk5.mlmodelc/weights/weight.bin", byteSize: 486575936),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk6.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk6.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk6.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk6.mlmodelc/coremldata.bin", byteSize: 501),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk6.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-1B-Instruct_chunk6.mlmodelc/metadata.json", byteSize: 3881),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk6.mlmodelc/model.mil")!, relativePath: "Llama-3.2-1B-Instruct_chunk6.mlmodelc/model.mil", byteSize: 12288),
            CoreMLPackageFile(url: URL(string: llamaBase + "Llama-3.2-1B-Instruct_chunk6.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-1B-Instruct_chunk6.mlmodelc/weights/weight.bin", byteSize: 525341504),
            CoreMLPackageFile(url: URL(string: llamaBase + "cache-processor.mlmodelc/analytics/coremldata.bin")!, relativePath: "cache-processor.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llamaBase + "cache-processor.mlmodelc/coremldata.bin")!, relativePath: "cache-processor.mlmodelc/coremldata.bin", byteSize: 516),
            CoreMLPackageFile(url: URL(string: llamaBase + "cache-processor.mlmodelc/metadata.json")!, relativePath: "cache-processor.mlmodelc/metadata.json", byteSize: 3162),
            CoreMLPackageFile(url: URL(string: llamaBase + "cache-processor.mlmodelc/model.mil")!, relativePath: "cache-processor.mlmodelc/model.mil", byteSize: 3429)
        ]

        // Llama 3.2 3B Instruct, same conversion project and same 512-token sliding context as
        // the 1B build above, just 16 chunks instead of 6 (`ChunkedPipelineCoreMLEngine` discovers
        // chunk files by scanning for the `_chunk<N>` suffix, so it needs no code changes to run a
        // model split into more pieces). Same `logit-processor.mlmodelc` exclusion as the 1B, for
        // the same reason. Byte sizes fetched directly from the Hugging Face tree API, not
        // estimated. The upstream repo's own README warns this will "likely run slowly or not at
        // all on M1 Macs and phones" other than the newest, highest-RAM ones — reflected below in
        // a deliberately high `minimumRAMGB` rather than smoothed over.
        let llama3bBase = "https://huggingface.co/smpanaro/Llama-3.2-3B-Instruct-CoreML/resolve/main/"
        let llama3bFiles: [CoreMLPackageFile] = [
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk1.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk1.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk1.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk1.mlmodelc/coremldata.bin", byteSize: 409),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk1.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk1.mlmodelc/metadata.json", byteSize: 2895),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk1.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk1.mlmodelc/model.mil", byteSize: 10350),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk1.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk1.mlmodelc/weights/weight.bin", byteSize: 788398464),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk2.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk2.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk2.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk2.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk2.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk2.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk2.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk2.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk2.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk2.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk3.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk3.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk3.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk3.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk3.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk3.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk3.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk3.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk3.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk3.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk4.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk4.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk4.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk4.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk4.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk4.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk4.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk4.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk4.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk4.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk5.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk5.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk5.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk5.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk5.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk5.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk5.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk5.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk5.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk5.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk6.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk6.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk6.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk6.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk6.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk6.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk6.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk6.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk6.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk6.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk7.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk7.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk7.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk7.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk7.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk7.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk7.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk7.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk7.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk7.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk8.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk8.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk8.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk8.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk8.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk8.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk8.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk8.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk8.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk8.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk9.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk9.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk9.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk9.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk9.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk9.mlmodelc/metadata.json", byteSize: 5278),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk9.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk9.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk9.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk9.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk10.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk10.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk10.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk10.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk10.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk10.mlmodelc/metadata.json", byteSize: 5279),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk10.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk10.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk10.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk10.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk11.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk11.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk11.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk11.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk11.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk11.mlmodelc/metadata.json", byteSize: 5279),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk11.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk11.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk11.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk11.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk12.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk12.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk12.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk12.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk12.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk12.mlmodelc/metadata.json", byteSize: 5279),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk12.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk12.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk12.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk12.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk13.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk13.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk13.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk13.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk13.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk13.mlmodelc/metadata.json", byteSize: 5279),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk13.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk13.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk13.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk13.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk14.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk14.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk14.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk14.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk14.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk14.mlmodelc/metadata.json", byteSize: 5279),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk14.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk14.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk14.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk14.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk15.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk15.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk15.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk15.mlmodelc/coremldata.bin", byteSize: 653),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk15.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk15.mlmodelc/metadata.json", byteSize: 5279),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk15.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk15.mlmodelc/model.mil", byteSize: 160189),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk15.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk15.mlmodelc/weights/weight.bin", byteSize: 402679744),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk16.mlmodelc/analytics/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk16.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk16.mlmodelc/coremldata.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk16.mlmodelc/coremldata.bin", byteSize: 501),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk16.mlmodelc/metadata.json")!, relativePath: "Llama-3.2-3B-Instruct_chunk16.mlmodelc/metadata.json", byteSize: 3882),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk16.mlmodelc/model.mil")!, relativePath: "Llama-3.2-3B-Instruct_chunk16.mlmodelc/model.mil", byteSize: 12290),
            CoreMLPackageFile(url: URL(string: llama3bBase + "Llama-3.2-3B-Instruct_chunk16.mlmodelc/weights/weight.bin")!, relativePath: "Llama-3.2-3B-Instruct_chunk16.mlmodelc/weights/weight.bin", byteSize: 788011840),
            CoreMLPackageFile(url: URL(string: llama3bBase + "cache-processor.mlmodelc/analytics/coremldata.bin")!, relativePath: "cache-processor.mlmodelc/analytics/coremldata.bin", byteSize: 243),
            CoreMLPackageFile(url: URL(string: llama3bBase + "cache-processor.mlmodelc/coremldata.bin")!, relativePath: "cache-processor.mlmodelc/coremldata.bin", byteSize: 516),
            CoreMLPackageFile(url: URL(string: llama3bBase + "cache-processor.mlmodelc/metadata.json")!, relativePath: "cache-processor.mlmodelc/metadata.json", byteSize: 3182),
            CoreMLPackageFile(url: URL(string: llama3bBase + "cache-processor.mlmodelc/model.mil")!, relativePath: "cache-processor.mlmodelc/model.mil", byteSize: 3442)
        ]

        return [
            CatalogModel(
                id: "llama-3.2-1b-instruct-coreml-ane",
                kind: .coreML,
                displayName: "Llama 3.2 1B Instruct (ANE)",
                publisher: "Meta (Core ML build by smpanaro)",
                byteSize: llamaFiles.reduce(0) { $0 + $1.byteSize },
                parameterCount: "1B",
                quantization: "Float16",
                license: "Llama 3.2 Community License",
                attribution: "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Core ML conversion by smpanaro (coreml-llm-cli).",
                summary: "A real chat-quality model running on the Apple Neural Engine, with about 512 tokens of live sliding context — once a conversation runs past that, the earliest turns are gradually forgotten rather than the reply being cut off. A much larger download than the ANE demo model, split across several files.",
                url: llamaFiles[0].url,
                approxRuntimeGB: 3.2,
                minimumRAMGB: 6.0,
                minimumDevice: "minimum iPhone 12 Pro / 14 or newer",
                fileNameOverride: "Llama-3.2-1B-Instruct-CoreML",
                coreMLFiles: llamaFiles
            ),
            CatalogModel(
                id: "llama-3.2-3b-instruct-coreml-ane",
                kind: .coreML,
                displayName: "Llama 3.2 3B Instruct (ANE)",
                publisher: "Meta (Core ML build by smpanaro)",
                byteSize: llama3bFiles.reduce(0) { $0 + $1.byteSize },
                parameterCount: "3B",
                quantization: "Float16",
                license: "Llama 3.2 Community License",
                attribution: "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Core ML conversion by smpanaro (coreml-llm-cli).",
                summary: "The largest model this app can run on the Apple Neural Engine — noticeably better replies than the 1B build, same 512-token sliding context. A ~7 GB download split across 16 chunk files, and meaningfully slower per token than the 1B model even where it does fit; the conversion's own authors note it may run slowly or not at all on older or lower-RAM devices. Try the 1B model first if this one struggles.",
                url: llama3bFiles[0].url,
                approxRuntimeGB: 7.7,
                minimumRAMGB: 12.0,
                minimumDevice: "minimum iPhone 16 Pro or newer, 12GB+ RAM strongly recommended",
                fileNameOverride: "Llama-3.2-3B-Instruct-CoreML",
                coreMLFiles: llama3bFiles
            )
        ]
    }()

    static var all: [CatalogModel] { chatModels + diffusionModels + coreMLModels }

    static var recommended: CatalogModel { chatModels[0] }

    static func models(for kind: ModelKind) -> [CatalogModel] {
        switch kind {
        case .chat: return chatModels
        case .diffusion: return diffusionModels
        case .coreML: return coreMLModels
        }
    }

    static func model(withFileName name: String) -> CatalogModel? {
        all.first { $0.fileName == name }
    }

    static func model(withID id: String) -> CatalogModel? {
        all.first { $0.id == id }
    }
}

// MARK: - Download manager

/// Downloads catalog models into `AppFiles.models`.
///
/// Uses a background `URLSession` so a multi-gigabyte transfer survives the user leaving the
/// app — on a phone, a one-gigabyte download that dies the moment the screen locks is not a
/// feature anyone can actually use. The paired `AppDelegate` hook below is what lets iOS wake
/// the app to finish the handoff.
@MainActor
final class ModelDownloadManager: NSObject, ObservableObject {

    static let shared = ModelDownloadManager()

    struct Progress {
        var modelID: String
        var kind: ModelKind
        var fractionCompleted: Double
        var bytesWritten: Int64
        var totalBytes: Int64
        /// True when this transfer picked up from a saved partial rather than starting over.
        var isResumed: Bool = false

        var writtenDescription: String {
            ByteCountFormatter.string(fromByteCount: bytesWritten, countStyle: .file)
        }
        var totalDescription: String {
            ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)
        }
    }

    /// Every transfer currently in flight, keyed by catalog model ID — downloads run
    /// concurrently rather than one at a time, so this replaces what used to be a single
    /// optional `active` slot.
    @Published private(set) var activeDownloads: [String: Progress] = [:]
    @Published private(set) var lastError: String?
    /// Set when a download finishes so the UI can advance without polling the filesystem.
    @Published private(set) var lastCompletedModelID: String?

    /// Memoizes `isInstalled(_:)` by model ID. `catalogRow`/`downloadCatalogButton` in
    /// `SettingsView` call `isInstalled` directly from their view bodies, and those bodies observe
    /// `activeDownloads`, which republishes many times per second while any download is running —
    /// without this, an unrelated in-progress download would re-run a `.coreML` model's full
    /// manifest stat sweep (30+ `FileManager` calls) on every progress tick, for every visible row.
    /// Invalidated at `finish(_:with:producedResumeData:)` (covers every download completion, success
    /// or failure) and wherever a model is deleted from disk — the two events that can actually
    /// change what this returns.
    private var installedCache: [String: Bool] = [:]

    /// Catalog IDs with a partial transfer saved on disk that can be picked up where it stopped.
    ///
    /// Before this existed, *any* interruption — a dropped connection, the user tapping Cancel, iOS
    /// terminating the app, or an app update landing mid-transfer — threw away every byte and
    /// started the next attempt from zero. On a 2 GB checkpoint over a phone connection that is the
    /// difference between a download that eventually finishes and one that never does, and it is
    /// the most likely thing behind "I have to download the model again after every update".
    @Published private(set) var resumableModelIDs: Set<String> = []

    /// Off by default. A ~1 GB download on a metered plan is not something to start on the
    /// user's behalf without an explicit opt-in.
    @Published var allowsCellularDownload: Bool = UserDefaults.standard.bool(forKey: "allowsCellularDownload") {
        didSet { UserDefaults.standard.set(allowsCellularDownload, forKey: "allowsCellularDownload") }
    }

    /// Populated by the app delegate when iOS relaunches the app to hand back a finished
    /// background transfer, and called once the session reports it has drained its queue.
    var backgroundCompletionHandler: (() -> Void)?

    private var session: URLSession!
    /// The in-flight task for each model currently downloading, keyed by catalog model ID.
    private var tasksByModelID: [String: URLSessionDownloadTask] = [:]
    /// The reverse lookup — delegate callbacks only hand back a task, never the model it
    /// belongs to, so this is how they find out which download they're reporting on.
    private var modelsByTaskID: [Int: CatalogModel] = [:]
    /// Task identifiers whose task was created from saved resume data. Read by `finish(_:with:)`
    /// to tell a stale resume blob apart from an ordinary network failure.
    private var resumedTaskIDs: Set<Int> = []

    /// Files not yet fetched for an in-progress `.coreML` multi-file download, in order — see
    /// `downloadCoreMLFiles`. Nothing here needs to survive a failure or relaunch: it's always
    /// rebuilt from what's actually missing on disk the next time that model's download starts.
    private var pendingCoreMLFiles: [String: [CoreMLPackageFile]] = [:]
    /// Bytes already landed on disk for a `.coreML` download's already-completed files this
    /// session — what `didWriteData` adds the in-flight task's own progress on top of.
    private var completedCoreMLBytes: [String: Int64] = [:]
    /// The file each in-flight download task belongs to, for `.coreML` models only. Its presence
    /// is how delegate callbacks tell a queued multi-file `.coreML` download apart from the
    /// legacy single-task path every other model kind uses.
    private var currentCoreMLFile: [Int: CoreMLPackageFile] = [:]

    private override init() {
        super.init()
        let configuration = URLSessionConfiguration.background(withIdentifier: "com.DDT.DarkAI.modelDownloads")
        configuration.allowsCellularAccess = true   // gated per-task instead, see `download(_:)`
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        configuration.waitsForConnectivity = true
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        resumableModelIDs = Self.savedResumableModelIDs()

        // A background `URLSession` survives the app being killed and relaunched by iOS to
        // deliver a finished or failed transfer — but a fresh launch starts with empty
        // `tasksByModelID`/`modelsByTaskID` maps, and every delegate callback below depends on
        // looking a task up in them. Without this, a task that finished (or was still running)
        // while the app process was dead reports in to a delegate that doesn't recognize its
        // `taskIdentifier` — `didFinishDownloadingTo` silently deletes the fully-downloaded file
        // rather than installing it, on exactly the transfer where it matters most: one that took
        // long enough to outlive the app process. Reattaching the survivors is what closes that
        // gap. `getAllTasks` only ever returns tasks still running or suspended, never ones that
        // already fully completed or were cancelled, so this can't resurrect anything stale.
        session.getAllTasks { [weak self] tasks in
            guard let self, !tasks.isEmpty else { return }
            Task { @MainActor in
                self.reattachSurvivingTasks(tasks)
            }
        }
    }

    /// Rebuilds this session's bookkeeping for whatever background tasks `getAllTasks` reports
    /// are still alive — see the doc comment on `init()` for why this exists. Relies on
    /// `taskDescription` (set at every task-creation site below) to recover which catalog model,
    /// and for a `.coreML` multi-file download which specific file, each surviving task belongs
    /// to; a task with no recognizable description (there shouldn't be any — nothing else in this
    /// app uses this session) is left alone.
    @MainActor
    private func reattachSurvivingTasks(_ tasks: [URLSessionTask]) {
        for task in tasks {
            guard let downloadTask = task as? URLSessionDownloadTask,
                  let description = task.taskDescription,
                  let decoded = Self.decodeTaskDescription(description),
                  let model = ModelCatalog.model(withID: decoded.modelID) else { continue }

            tasksByModelID[model.id] = downloadTask
            modelsByTaskID[downloadTask.taskIdentifier] = model
            // `start(_:resuming:)`/`startNextCoreMLFile` are the only other places that insert
            // into `resumedTaskIDs`, and neither runs for a task recovered here — so without this,
            // `finish(_:with:producedResumeData:)`'s `wasResumed` always reads `false` for a
            // reattached task, and its stale-resume-blob cleanup never runs if this task goes on
            // to fail. Marking every reattached task as "resumed" is safe even when no resume blob
            // actually exists for it: the cleanup this enables is a `try?`-guarded delete (a no-op
            // if there's nothing to remove) plus clearing `resumableModelIDs`, which is the right
            // outcome either way if the task fails.
            resumedTaskIDs.insert(downloadTask.taskIdentifier)

            let received = max(0, downloadTask.countOfBytesReceived)
            let expected = downloadTask.countOfBytesExpectedToReceive

            if let relativePath = decoded.coreMLFileRelativePath,
               let file = model.coreMLFiles.first(where: { $0.relativePath == relativePath }) {
                currentCoreMLFile[downloadTask.taskIdentifier] = file

                // Same disk scan `downloadCoreMLFiles` does at a fresh start, excluding the file
                // this task is already fetching — rebuilds which files already landed so the
                // queue behind this one (`startNextCoreMLFile`) still fires correctly once it
                // finishes.
                let installedPath = Self.installDirectory(for: model.kind).appendingPathComponent(model.fileName)
                var completedBytes: Int64 = 0
                var pending: [CoreMLPackageFile] = []
                for candidate in model.coreMLFiles where candidate.relativePath != relativePath {
                    let path = installedPath.appendingPathComponent(candidate.relativePath).path
                    let size = try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64
                    if let size, size == candidate.byteSize {
                        completedBytes += candidate.byteSize
                    } else {
                        pending.append(candidate)
                    }
                }
                pendingCoreMLFiles[model.id] = pending
                completedCoreMLBytes[model.id] = completedBytes

                let bytesWritten = completedBytes + received
                activeDownloads[model.id] = Progress(
                    modelID: model.id,
                    kind: model.kind,
                    fractionCompleted: model.byteSize > 0 ? min(1.0, Double(bytesWritten) / Double(model.byteSize)) : 0,
                    bytesWritten: bytesWritten,
                    totalBytes: model.byteSize
                )
            } else {
                let total = expected > 0 ? expected : model.byteSize
                activeDownloads[model.id] = Progress(
                    modelID: model.id,
                    kind: model.kind,
                    fractionCompleted: total > 0 ? min(1.0, Double(received) / Double(total)) : 0,
                    bytesWritten: received,
                    totalBytes: total
                )
            }

            LogManager.shared.log("ModelDownload: reattached surviving background task for \(model.displayName) after relaunch")
        }
    }

    private static func encodeTaskDescription(modelID: String, coreMLFileRelativePath: String?) -> String {
        guard let coreMLFileRelativePath else { return "model:\(modelID)" }
        return "model:\(modelID)|file:\(coreMLFileRelativePath)"
    }

    private static func decodeTaskDescription(_ description: String) -> (modelID: String, coreMLFileRelativePath: String?)? {
        guard description.hasPrefix("model:") else { return nil }
        let parts = description.dropFirst("model:".count).components(separatedBy: "|file:")
        guard let modelID = parts.first, !modelID.isEmpty else { return nil }
        return (modelID, parts.count > 1 ? parts[1] : nil)
    }

    /// Whether anything at all is downloading — onboarding uses this to gate advancing past the
    /// model-setup step; it doesn't need to know which or how many.
    var isDownloading: Bool { !activeDownloads.isEmpty }

    /// Whether this specific model is one of the in-flight downloads.
    func isDownloading(_ model: CatalogModel) -> Bool { activeDownloads[model.id] != nil }

    /// Chat weights, diffusion checkpoints, and Core ML packages are consumed by different
    /// engines and listed by different screens, so they have to land in different directories.
    /// `nonisolated`: a pure switch over `AppFiles`'s own (also nonisolated) static directories,
    /// touching no instance state — called from `verify(fileAt:against:)`'s off-main-actor path.
    nonisolated static func installDirectory(for kind: ModelKind) -> URL {
        switch kind {
        case .chat: return AppFiles.models
        case .diffusion: return AppFiles.diffusionModels
        case .coreML: return AppFiles.coreMLModels
        }
    }

    func isInstalled(_ model: CatalogModel) -> Bool {
        if let cached = installedCache[model.id] { return cached }
        let result = computeIsInstalled(model)
        installedCache[model.id] = result
        return result
    }

    /// Drops the cached result for one model, so the next `isInstalled` call recomputes it from
    /// disk. Called wherever install state actually changes — see `installedCache`'s doc comment.
    func invalidateInstalledCache(for modelID: String) {
        installedCache.removeValue(forKey: modelID)
    }

    private func computeIsInstalled(_ model: CatalogModel) -> Bool {
        let installedPath = Self.installDirectory(for: model.kind)
            .appendingPathComponent(model.fileName)
        guard model.kind == .coreML else {
            return FileManager.default.fileExists(atPath: installedPath.path)
        }
        // The directory existing isn't proof every file landed — every entry in the manifest must
        // be present at the right size, or this isn't a smaller version of the model, it's one
        // `MLModel` will refuse to load.
        return model.coreMLFiles.allSatisfy { file in
            let path = installedPath.appendingPathComponent(file.relativePath).path
            let size = try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64
            return size == file.byteSize
        }
    }

    // MARK: Actions

    func download(_ model: CatalogModel) {
        guard activeDownloads[model.id] == nil else { return }
        lastError = nil
        lastCompletedModelID = nil

        // Disk pre-flight. Failing here with a clear number beats failing 900 MB in with
        // "The operation couldn't be completed." Downloads run concurrently now, so this has to
        // reserve space for what every other in-flight transfer still has left to write, not
        // just this one — otherwise three simultaneous downloads that each individually fit
        // could jointly overrun what's actually free.
        let remainingForOthersGB = activeDownloads.values.reduce(0.0) { partial, progress in
            partial + Double(max(0, progress.totalBytes - progress.bytesWritten)) / 1_073_741_824.0
        }
        // For a `.coreML` model that's resuming, bytes already verified on disk (see
        // `coreMLDownloadState`) don't need to be fetched again — without this, a large,
        // nearly-complete download could be refused on a storage-tight device that has ample
        // room for what's actually left. Every other model kind resumes via saved `resumeData`
        // rather than partial files sitting at their final path, so this only applies here.
        // Computed once here and threaded through to `downloadCoreMLFiles` below, rather than
        // scanning the install directory a second time immediately after — both used to call
        // `coreMLDownloadState` independently for the same model on the same download start.
        let coreMLState = model.kind == .coreML ? coreMLDownloadState(for: model) : nil
        let alreadyOnDiskGB = Double(coreMLState?.completedBytes ?? 0) / 1_073_741_824.0
        let requiredGB = max(0, model.sizeGB - alreadyOnDiskGB) + 0.5 + remainingForOthersGB
        let availableGB = AppFiles.availableDiskGB()
        guard availableGB > requiredGB else {
            lastError = activeDownloads.isEmpty
                ? String(format: "Not enough storage. %@ needs about %.1f GB free and this device has %.1f GB.",
                         model.displayName, requiredGB, availableGB)
                : String(format: "Not enough storage. %@ needs about %.1f GB free (accounting for %d other download%@ in progress) and this device has %.1f GB.",
                         model.displayName, requiredGB, activeDownloads.count, activeDownloads.count == 1 ? "" : "s", availableGB)
            return
        }

        AppFiles.prepare()
        if model.kind == .coreML, let coreMLState {
            downloadCoreMLFiles(model, state: coreMLState)
        } else {
            start(model, resuming: Self.savedResumeData(forKey: model.id))
        }
    }

    /// Starts (or resumes) a `.coreML` model's multi-file download — every file in
    /// `model.coreMLFiles` is fetched through its own background `URLSessionDownloadTask`, one at
    /// a time, rather than the single task `start(_:resuming:)` uses. A `.mlpackage`/chunked
    /// `.mlmodelc` pipeline downloaded from Hugging Face isn't one file, so there's no single task
    /// to hang the transfer off of the way every other model kind can.
    ///
    /// Resumability here is by construction rather than saved state: a file that already exists on
    /// disk at its target path with the right byte count is simply skipped, so relaunching the app
    /// mid-download — or tapping Resume after a failure — only re-fetches what's actually missing.
    /// Only a file that was *mid-transfer* when interrupted needs the saved-resume-data path,
    /// exactly like the single-file case, just keyed per file instead of per model.
    /// Scans `model`'s install directory for `.coreML` files already verified complete on disk —
    /// shared by the disk-space pre-flight in `download(_:)` (so a resumed download isn't refused
    /// for space it doesn't actually need) and `downloadCoreMLFiles` below (which does the same
    /// scan to decide what's actually left to fetch).
    private func coreMLDownloadState(for model: CatalogModel) -> (completedBytes: Int64, pending: [CoreMLPackageFile]) {
        let installedPath = Self.installDirectory(for: model.kind).appendingPathComponent(model.fileName)
        var completedBytes: Int64 = 0
        var pending: [CoreMLPackageFile] = []
        for file in model.coreMLFiles {
            let path = installedPath.appendingPathComponent(file.relativePath).path
            let size = try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64
            if let size, size == file.byteSize {
                completedBytes += file.byteSize
            } else {
                pending.append(file)
            }
        }
        return (completedBytes, pending)
    }

    private func downloadCoreMLFiles(_ model: CatalogModel, state: (completedBytes: Int64, pending: [CoreMLPackageFile])) {
        let installedPath = Self.installDirectory(for: model.kind).appendingPathComponent(model.fileName)
        AppFiles.createIfNeeded(installedPath)

        let (completedBytes, pending) = state
        pendingCoreMLFiles[model.id] = pending
        completedCoreMLBytes[model.id] = completedBytes
        // Resuming either because some files already verified complete on disk, or the next file
        // to fetch has saved resume data from a prior mid-transfer interruption — either way this
        // isn't a fresh start. Without this, `OnboardingView`/`SettingsView` (which both branch UI
        // text on `progress.isResumed`, matching the single-file `start(_:resuming:)` path above)
        // always showed "Downloading…" for a Core ML model even when genuinely resuming.
        let isResumed = completedBytes > 0
            || pending.first.map { Self.savedResumeData(forKey: Self.coreMLFileResumeKey(model: model, file: $0)) != nil } == true
        activeDownloads[model.id] = Progress(
            modelID: model.id,
            kind: model.kind,
            fractionCompleted: model.byteSize > 0 ? Double(completedBytes) / Double(model.byteSize) : 0,
            bytesWritten: completedBytes,
            totalBytes: model.byteSize,
            isResumed: isResumed
        )
        LogManager.shared.log("ModelDownload: starting \(model.displayName) (\(pending.count) of \(model.coreMLFiles.count) files remaining)")
        startNextCoreMLFile(for: model)
    }

    private func startNextCoreMLFile(for model: CatalogModel) {
        guard var pending = pendingCoreMLFiles[model.id], !pending.isEmpty else {
            finalizeCoreMLDownload(model)
            return
        }
        let file = pending.removeFirst()
        pendingCoreMLFiles[model.id] = pending

        let resumeKey = Self.coreMLFileResumeKey(model: model, file: file)
        let task: URLSessionDownloadTask
        if let resumeData = Self.savedResumeData(forKey: resumeKey) {
            task = session.downloadTask(withResumeData: resumeData)
            // Without this, `finish(_:with:)` can never tell a resumed coreML file's task apart
            // from a fresh one — `wasResumed` would always read false, and a permanently stale
            // resume blob (the server no longer honors the range, or too much time has passed)
            // would never get cleaned up, so every subsequent attempt would keep reusing the same
            // dead partial and failing identically forever. See `finish`'s coreML branch.
            resumedTaskIDs.insert(task.taskIdentifier)
        } else {
            var request = URLRequest(url: file.url)
            request.allowsCellularAccess = allowsCellularDownload
            request.allowsExpensiveNetworkAccess = allowsCellularDownload
            request.timeoutInterval = 60
            task = session.downloadTask(with: request)
        }
        task.countOfBytesClientExpectsToReceive = file.byteSize
        // Read back by `reattachSurvivingTasks` if this task outlives the app process — see
        // `init()`'s doc comment.
        task.taskDescription = Self.encodeTaskDescription(modelID: model.id, coreMLFileRelativePath: file.relativePath)

        tasksByModelID[model.id] = task
        modelsByTaskID[task.taskIdentifier] = model
        currentCoreMLFile[task.taskIdentifier] = file
        task.resume()
    }

    /// Called once every file in `model.coreMLFiles` is confirmed on disk. Unlike the single-file
    /// path there's no "move into place" step left to do here — each file already landed at its
    /// final `relativePath` as it finished in `finishCoreMLFile` — this just closes out the
    /// bookkeeping the same way the single-file success path does.
    private func finalizeCoreMLDownload(_ model: CatalogModel) {
        pendingCoreMLFiles.removeValue(forKey: model.id)
        completedCoreMLBytes.removeValue(forKey: model.id)
        resumableModelIDs.remove(model.id)
        ModelInventory.shared.record(
            fileName: model.fileName,
            kind: model.kind,
            catalogID: model.id,
            byteSize: model.byteSize
        )
        LogManager.shared.log("ModelDownload: installed \(model.fileName)")
        lastCompletedModelID = model.id
        finish(model, with: nil)
    }

    private static func coreMLFileResumeKey(model: CatalogModel, file: CoreMLPackageFile) -> String {
        "\(model.id)__coreml__\(file.relativePath.replacingOccurrences(of: "/", with: "_"))"
    }

    /// Creates and starts the task, from saved resume data when there is any.
    ///
    /// `resumeData` is opaque and can be rejected by the system — it goes stale if the server no
    /// longer supports the byte range, if the temporary file behind it has been reclaimed, or
    /// simply if too much time has passed. That failure arrives as an ordinary task error rather
    /// than a throw, so `finish(_:with:)` handles it by discarding the partial and saying the
    /// download restarted, instead of leaving the user stuck retrying a resume that can never work.
    private func start(_ model: CatalogModel, resuming resumeData: Data?) {
        let task: URLSessionDownloadTask
        if let resumeData {
            task = session.downloadTask(withResumeData: resumeData)
            LogManager.shared.log("ModelDownload: resuming \(model.displayName)")
        } else {
            var request = URLRequest(url: model.url)
            request.allowsCellularAccess = allowsCellularDownload
            request.allowsExpensiveNetworkAccess = allowsCellularDownload
            request.timeoutInterval = 60
            task = session.downloadTask(with: request)
            LogManager.shared.log("ModelDownload: starting \(model.displayName) (\(model.sizeDescription))")
        }
        task.countOfBytesClientExpectsToReceive = model.byteSize
        // Read back by `reattachSurvivingTasks` if this task outlives the app process — see
        // `init()`'s doc comment.
        task.taskDescription = Self.encodeTaskDescription(modelID: model.id, coreMLFileRelativePath: nil)

        tasksByModelID[model.id] = task
        modelsByTaskID[task.taskIdentifier] = model
        if resumeData != nil { resumedTaskIDs.insert(task.taskIdentifier) }
        activeDownloads[model.id] = Progress(
            modelID: model.id,
            kind: model.kind,
            fractionCompleted: 0,
            bytesWritten: 0,
            totalBytes: model.byteSize,
            isResumed: resumeData != nil
        )
        task.resume()
    }

    /// Stops one transfer but keeps what has already been fetched, so tapping Get again picks up
    /// where this left off. Other in-flight downloads are untouched. Use `discardPartial(for:)`
    /// to throw the bytes away instead.
    func cancel(_ model: CatalogModel) {
        guard let task = tasksByModelID[model.id] else { return }
        let resumeKey = currentCoreMLFile[task.taskIdentifier]
            .map { Self.coreMLFileResumeKey(model: model, file: $0) } ?? model.id
        // Goes through the singleton rather than capturing `self`: this callback is delivered on a
        // background queue and outlives the call, and reaching back through `shared` keeps that
        // free of a cross-actor capture instead of relying on one being tolerated.
        task.cancel { resumeData in
            Task { @MainActor in
                let manager = ModelDownloadManager.shared
                if let resumeData {
                    Self.saveResumeData(resumeData, forKey: resumeKey)
                    manager.resumableModelIDs.insert(model.id)
                    LogManager.shared.log("ModelDownload: paused \(model.displayName), \(ByteCountFormatter.string(fromByteCount: Int64(resumeData.count), countStyle: .file)) of resume state kept")
                } else {
                    LogManager.shared.log("ModelDownload: cancelled \(model.displayName) — the server didn't support resuming")
                }
            }
        }
        tasksByModelID.removeValue(forKey: model.id)
        modelsByTaskID.removeValue(forKey: task.taskIdentifier)
        resumedTaskIDs.remove(task.taskIdentifier)
        currentCoreMLFile.removeValue(forKey: task.taskIdentifier)
        activeDownloads.removeValue(forKey: model.id)
        // pendingCoreMLFiles/completedCoreMLBytes deliberately left alone: downloadCoreMLFiles
        // recomputes both fresh from disk the next time this model's download starts, so there's
        // nothing stale here that needs clearing.
    }

    /// Throws away a saved partial transfer. Offered next to the resume affordance so a user who
    /// changed their mind isn't stuck carrying a gigabyte of a model they no longer want.
    func discardPartial(for model: CatalogModel) {
        if model.kind == .coreML {
            // Only the resume blobs — files that already finished and were verified stay on disk;
            // "discard the partial" means abandon whatever was mid-transfer, not throw away
            // completed, byte-verified progress along with it.
            for file in model.coreMLFiles {
                Self.deleteResumeData(forKey: Self.coreMLFileResumeKey(model: model, file: file))
            }
        } else {
            Self.deleteResumeData(forKey: model.id)
        }
        resumableModelIDs.remove(model.id)
        LogManager.shared.log("ModelDownload: discarded partial download of \(model.displayName)")
    }

    // MARK: Resume state

    /// Resume blobs live beside partial downloads, in the directory that already exists for
    /// exactly this purpose and is already excluded from backup. `key` is a catalog model ID for
    /// the single-file path, or `coreMLFileResumeKey(model:file:)` for one file of a `.coreML`
    /// model's multi-file download.
    private static func resumeFileURL(for key: String) -> URL {
        AppFiles.downloadsInProgress.appendingPathComponent("\(key).resume")
    }

    private static func savedResumeData(forKey key: String) -> Data? {
        try? Data(contentsOf: resumeFileURL(for: key))
    }

    private static func saveResumeData(_ data: Data, forKey key: String) {
        AppFiles.createIfNeeded(AppFiles.downloadsInProgress)
        let url = resumeFileURL(for: key)
        try? data.write(to: url, options: .atomic)
        AppFiles.excludeFromBackup(url)
    }

    private static func deleteResumeData(forKey key: String) {
        try? FileManager.default.removeItem(at: resumeFileURL(for: key))
    }

    private static func savedResumableModelIDs() -> Set<String> {
        let files = AppFiles.contents(of: AppFiles.downloadsInProgress, matchingExtensions: ["resume"])
        let known = Set(ModelCatalog.all.map(\.id))
        return Set(files.compactMap { url -> String? in
            let stem = (url.lastPathComponent as NSString).deletingPathExtension
            // A .coreML per-file resume blob is named "<modelID>__coreml__<relPath>" — recover
            // the owning model's ID rather than treating the whole stem as one.
            let modelID = stem.components(separatedBy: "__coreml__").first ?? stem
            return known.contains(modelID) ? modelID : nil
        })
    }

    // MARK: Verification

    /// Confirms the bytes on disk are the model we asked for before it is ever handed to
    /// llama.cpp. A truncated or redirected download is otherwise indistinguishable from a
    /// corrupt model, and llama.cpp's failure mode for that is a crash rather than an error.
    /// `nonisolated`: pure file I/O over its two parameters, touching no instance state — the
    /// `didFinishDownloadingTo` delegate call relies on that to run this off the main actor.
    private nonisolated func verify(fileAt url: URL, against model: CatalogModel) throws {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        guard size == model.byteSize else {
            throw NSError(domain: "ModelDownload", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "The downloaded file is \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)) but should be \(model.sizeDescription). The download was incomplete."
            ])
        }
        // Header sanity, format-aware. Cheap, and catches an HTML error page saved under the
        // expected filename — which is exactly what a captive-portal Wi-Fi network produces —
        // regardless of which of the app's supported formats this particular model is in.
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let corruptError = NSError(domain: "ModelDownload", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "The downloaded file is not a valid \(model.kind == .diffusion ? "diffusion" : "GGUF") model. Check your network connection and try again."
        ])
        switch (model.fileName as NSString).pathExtension.lowercased() {
        case "safetensors":
            // Layout: an 8-byte little-endian header length, then that many bytes of JSON
            // tensor metadata. Confirming the header actually parses as JSON is the safetensors
            // equivalent of the GGUF magic-byte check below — safetensors has no fixed magic
            // bytes of its own to check instead.
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { throw corruptError }
            let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
            guard headerLen > 0, headerLen < 100_000_000,
                  let headerBytes = try handle.read(upToCount: Int(headerLen)),
                  (try? JSONSerialization.jsonObject(with: headerBytes)) != nil else {
                throw corruptError
            }
        default:
            // GGUF magic bytes — covers both chat models and the GGUF-format diffusion entry.
            guard let magic = try handle.read(upToCount: 4), magic == Data("GGUF".utf8) else {
                throw corruptError
            }
        }

        // Beyond the magic-byte/JSON-parses sanity check above, run the same structural
        // validation a manually-imported file already gets (`GGUFValidator.validate` for chat
        // models via `SettingsView.copyModelToAppDocuments`/`OnboardingView.handleImport`;
        // `validateDiffusionCheckpoint` for diffusion checkpoints via `SettingsView`'s diffusion
        // importer) — a catalog entry is curated and low-risk in practice, but a byte-size match
        // with genuinely wrong content at the same size (a bad catalog edit, or a host silently
        // re-serving different bytes) would otherwise sail through this path when the exact same
        // file would have been caught on the import path. Both validators are header-only and
        // documented to cost milliseconds regardless of file size.
        switch model.kind {
        case .chat:
    let chatHandle = try FileHandle(forReadingFrom: url)
    defer { try? chatHandle.close() }

    let magic = chatHandle.readData(ofLength: 4)
    guard magic == Data([0x47, 0x47, 0x55, 0x46]) else {
        throw NSError(
            domain: "DarkAI.ModelValidation",
            code: 1001,
            userInfo: [NSLocalizedDescriptionKey: "The downloaded file is not a valid GGUF file."]
        )
    }

    let versionData = chatHandle.readData(ofLength: 4)
    guard versionData.count == 4 else {
        throw NSError(
            domain: "DarkAI.ModelValidation",
            code: 1002,
            userInfo: [NSLocalizedDescriptionKey: "The GGUF header is incomplete."]
        )
    }

    let version = versionData.withUnsafeBytes {
        $0.load(as: UInt32.self).littleEndian
    }

    guard version == 2 || version == 3 else {
        throw NSError(
            domain: "DarkAI.ModelValidation",
            code: 1003,
            userInfo: [NSLocalizedDescriptionKey: "Unsupported GGUF version \(version)."]
        )
    }
        case .diffusion:
            try GGUFValidator.validateDiffusionCheckpoint(path: url.path)
        case .coreML:
            break
        }
    }
}

// MARK: - URLSessionDownloadDelegate

extension ModelDownloadManager: URLSessionDownloadDelegate {

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64,
                                totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        let taskID = downloadTask.taskIdentifier
        Task { @MainActor in
            guard let model = self.modelsByTaskID[taskID],
                  var progress = self.activeDownloads[model.id] else { return }
            if self.currentCoreMLFile[taskID] != nil {
                // Multi-file queue: the denominator is the whole model's total size, and the
                // numerator is bytes from already-completed files plus this file's own progress —
                // not just this one task's bytes.
                let completed = self.completedCoreMLBytes[model.id] ?? 0
                progress.bytesWritten = completed + totalBytesWritten
                progress.totalBytes = model.byteSize
                progress.fractionCompleted = model.byteSize > 0 ? min(1.0, Double(progress.bytesWritten) / Double(model.byteSize)) : 0
            } else {
                // `totalBytesExpectedToWrite` is -1 when the server omits Content-Length; the
                // catalog's known size is the better denominator in that case.
                let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : progress.totalBytes
                progress.bytesWritten = totalBytesWritten
                progress.totalBytes = total
                progress.fractionCompleted = total > 0 ? min(1.0, Double(totalBytesWritten) / Double(total)) : 0
            }
            self.activeDownloads[model.id] = progress
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        let taskID = downloadTask.taskIdentifier
        // This runs on a background queue and `location` is deleted the moment it returns, so
        // the move has to happen synchronously here rather than inside a hop to the main actor.
        let temporaryCopy = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: temporaryCopy)
        } catch {
            Task { @MainActor in
                guard let model = self.modelsByTaskID[taskID] else { return }
                self.finish(model, with: error)
            }
            return
        }

        Task {
            // Only the bookkeeping-dictionary lookup needs the main actor — everything after it
            // (`verify`, the install-directory move) is real file I/O on what can be a
            // multi-gigabyte file, and used to run entirely inside one `@MainActor` hop, blocking
            // the main thread for however long that took.
            let taskDescription = downloadTask.taskDescription

let lookup: (model: CatalogModel, coreMLFile: CoreMLPackageFile?)? = await MainActor.run {
    if let model = self.modelsByTaskID[taskID] {
        let file = self.currentCoreMLFile[taskID]
        if file != nil { self.currentCoreMLFile.removeValue(forKey: taskID) }
        return (model, file)
    }

    // Recover the model after iOS has relaunched the app/background session.
    guard let description = taskDescription,
          let (modelID, coreMLPath) = Self.decodeTaskDescription(description),
          let model = ModelCatalog.all.first(where: { $0.id == modelID }) else {
        return nil
    }

    let file = coreMLPath.flatMap { path in
        model.coreMLFiles.first(where: { $0.relativePath == path })
    }

    return (model, file)
}

            guard let (model, coreMLFile) = lookup else {
                try? FileManager.default.removeItem(at: temporaryCopy)
                return
            }

            if let coreMLFile {
                await MainActor.run {
                    self.finishCoreMLFile(coreMLFile, tempFile: temporaryCopy, model: model)
                }
                return
            }

            do {
                try self.verify(fileAt: temporaryCopy, against: model)

                let installDirectory = Self.installDirectory(for: model.kind)
                AppFiles.createIfNeeded(installDirectory)
                let destination = installDirectory.appendingPathComponent(model.fileName)
                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                try FileManager.default.moveItem(at: temporaryCopy, to: destination)
                AppFiles.excludeFromBackup(destination)

                await MainActor.run {
                    // The partial is worthless now, and the ledger entry is what lets the app
                    // tell the user *which* model vanished if this device is ever restored
                    // without it.
                    Self.deleteResumeData(forKey: model.id)
                    self.resumableModelIDs.remove(model.id)
                    ModelInventory.shared.record(
                        fileName: model.fileName,
                        kind: model.kind,
                        catalogID: model.id,
                        byteSize: model.byteSize
                    )

                    LogManager.shared.log("ModelDownload: installed \(model.fileName)")
                    self.lastCompletedModelID = model.id
                    self.finish(model, with: nil)
                }
            } catch {
    let errorDescription = error.localizedDescription

    LogManager.shared.log(
        "ModelDownload: INSTALL FAILED — \(model.displayName) — \(errorDescription)"
    )

    await MainActor.run {
        self.finish(model, with: error)
    }
}

    /// One file of a `.coreML` multi-file download has finished — verify its size, move it into
    /// place at its `relativePath`, and either continue the queue or (once every file has landed)
    /// finalize the whole model via `finalizeCoreMLDownload`.
    @MainActor
    private func finishCoreMLFile(_ file: CoreMLPackageFile, tempFile: URL, model: CatalogModel) {
        let size = (try? FileManager.default.attributesOfItem(atPath: tempFile.path)[.size] as? Int64) ?? 0
        guard size == file.byteSize else {
            try? FileManager.default.removeItem(at: tempFile)
            let error = NSError(domain: "ModelDownload", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "\(file.relativePath) downloaded as \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)) but should be \(ByteCountFormatter.string(fromByteCount: file.byteSize, countStyle: .file)). The download was incomplete."
            ])
            finish(model, with: error)
            return
        }

        do {
            let installedPath = Self.installDirectory(for: model.kind).appendingPathComponent(model.fileName)
            let destination = installedPath.appendingPathComponent(file.relativePath)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: tempFile, to: destination)
            AppFiles.excludeFromBackup(destination)
        } catch {
            try? FileManager.default.removeItem(at: tempFile)
            finish(model, with: error)
            return
        }

        Self.deleteResumeData(forKey: Self.coreMLFileResumeKey(model: model, file: file))
        completedCoreMLBytes[model.id, default: 0] += file.byteSize
        startNextCoreMLFile(for: model)
    }

    nonisolated func urlSession(_ session: URLSession,
                                task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }   // success is handled in didFinishDownloadingTo
        let nsError = error as NSError
        // A user-initiated cancel is not a failure worth surfacing as one, and its resume data
        // arrives through `cancel(byProducingResumeData:)` rather than here.
        guard nsError.code != NSURLErrorCancelled else { return }

        // The bytes already fetched are salvageable whenever the system hands back resume state —
        // a dropped connection nine-tenths of the way through a 2 GB checkpoint should cost the
        // remaining tenth, not the whole thing.
        let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let taskID = task.taskIdentifier

        Task { @MainActor in
            guard let model = self.modelsByTaskID[taskID] else { return }
            if let resumeData {
                let key = self.currentCoreMLFile[taskID].map { Self.coreMLFileResumeKey(model: model, file: $0) } ?? model.id
                Self.saveResumeData(resumeData, forKey: key)
                self.resumableModelIDs.insert(model.id)
            }
            self.finish(model, with: error, producedResumeData: resumeData != nil)
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in
            self.backgroundCompletionHandler?()
            self.backgroundCompletionHandler = nil
        }
    }

    @MainActor
    private func finish(_ model: CatalogModel, with error: Error?, producedResumeData: Bool = false) {
        // Every completion path — this model's own success, another file of a multi-file coreML
        // download finishing, or a failure — runs through here, so this is the one place that has
        // to invalidate `installedCache` for every route that can change what's on disk for it.
        invalidateInstalledCache(for: model.id)

        let bytesWritten = activeDownloads[model.id]?.bytesWritten ?? 0
        let taskID = tasksByModelID[model.id]?.taskIdentifier
        let wasResumed = taskID.map { resumedTaskIDs.contains($0) } ?? false
        let coreMLFile = taskID.flatMap { currentCoreMLFile[$0] }

        activeDownloads.removeValue(forKey: model.id)
        tasksByModelID.removeValue(forKey: model.id)
        if let taskID {
            modelsByTaskID.removeValue(forKey: taskID)
            resumedTaskIDs.remove(taskID)
            currentCoreMLFile.removeValue(forKey: taskID)
        }
        if model.kind == .coreML {
            pendingCoreMLFiles.removeValue(forKey: model.id)
            completedCoreMLBytes.removeValue(forKey: model.id)
        }

        guard let error else { return }

        // Stale resume state: the task was built from a saved partial and the system declined to
        // give any back. Retrying would fail identically every time, so the dead blob has to be
        // cleared either way — what differs is what happens next.
        if wasResumed, !producedResumeData {
            if let coreMLFile {
                // Multi-file `.coreML` downloads don't auto-restart inline the way the single-file
                // branch below does — the files already verified on disk make a plain
                // `download(_:)` retry cheap regardless, so there's no need to replicate that
                // dance here. But the stale blob for *this specific file* still has to go, or
                // every subsequent "tap Resume" reuses the same dead partial and fails identically
                // forever: this used to only ever delete the single-file path's resume key (keyed
                // by `model.id`), never a coreML file's own (keyed by `coreMLFileResumeKey`), so a
                // stale coreML resume never actually self-healed the way this comment claimed.
                Self.deleteResumeData(forKey: Self.coreMLFileResumeKey(model: model, file: coreMLFile))
                resumableModelIDs.remove(model.id)
                LogManager.shared.log("ModelDownload: saved partial for \(coreMLFile.relativePath) (\(model.displayName)) was no longer usable — it'll restart fresh next attempt")
            } else {
                // Whether or not this attempt took on any new bytes before failing, the old
                // resume blob is dead the moment the system declines to hand back fresh resume
                // data — clearing it (and `resumableModelIDs`) always, not just when
                // `bytesWritten == 0`, is what stops "Resume" from replaying the same dead
                // partial forever. Only the *inline auto-restart* below stays gated on
                // `bytesWritten == 0`: that's specifically the "nothing came in this attempt"
                // case, where silently starting over is unambiguously the right move rather than
                // surfacing an error for an attempt that never really got going.
                Self.deleteResumeData(forKey: model.id)
                resumableModelIDs.remove(model.id)
                if bytesWritten == 0 {
                    LogManager.shared.log("ModelDownload: saved partial for \(model.displayName) was no longer usable — restarting from the beginning")
                    start(model, resuming: nil)
                    return
                }
                LogManager.shared.log("ModelDownload: saved partial for \(model.displayName) was no longer usable — this attempt will need to restart fresh next time")
            }
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain,
           nsError.code == NSURLErrorDataNotAllowed || nsError.code == NSURLErrorInternationalRoamingOff {
            lastError = "This download needs Wi-Fi. Connect to Wi-Fi, or turn on cellular downloads first."
        } else if producedResumeData {
            lastError = "\(error.localizedDescription) Your progress was kept — tap Resume to carry on from where it stopped."
        } else {
            lastError = error.localizedDescription
        }
        LogManager.shared.log("ModelDownload: failed — \(model.displayName) — \(error.localizedDescription)")
    }
}
