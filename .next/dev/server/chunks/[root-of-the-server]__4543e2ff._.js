module.exports = [
"[externals]/next/dist/compiled/next-server/app-route-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-route-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/@opentelemetry/api [external] (next/dist/compiled/@opentelemetry/api, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/@opentelemetry/api", () => require("next/dist/compiled/@opentelemetry/api"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/next-server/app-page-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-page-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-unit-async-storage.external.js [external] (next/dist/server/app-render/work-unit-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-unit-async-storage.external.js", () => require("next/dist/server/app-render/work-unit-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-async-storage.external.js [external] (next/dist/server/app-render/work-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-async-storage.external.js", () => require("next/dist/server/app-render/work-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/shared/lib/no-fallback-error.external.js [external] (next/dist/shared/lib/no-fallback-error.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/shared/lib/no-fallback-error.external.js", () => require("next/dist/shared/lib/no-fallback-error.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/after-task-async-storage.external.js [external] (next/dist/server/app-render/after-task-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/after-task-async-storage.external.js", () => require("next/dist/server/app-render/after-task-async-storage.external.js"));

module.exports = mod;
}),
"[project]/lib/config.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

// Configuration for the ML backend
// The Python FastAPI server should be running on this URL
__turbopack_context__.s([
    "ML_CONFIG",
    ()=>ML_CONFIG
]);
const ML_CONFIG = {
    // The URL where your Python ML backend is running
    // For development, this is typically localhost:8001
    // For production, replace with your actual backend URL
    BACKEND_URL: process.env.NEXT_PUBLIC_ML_BACKEND_URL || "http://localhost:8001",
    // API endpoints
    ENDPOINTS: {
        ANALYZE_TEXT: "/api/analyze/text",
        ANALYZE_IMAGE: "/api/analyze/image",
        ANALYZE_FRAME: "/api/analyze/frame",
        HEALTH: "/api/health"
    },
    // Emotion labels matching the model output order
    EMOTION_LABELS: [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral"
    ],
    // Confidence threshold (same as Python model)
    CONFIDENCE_THRESHOLD: 0.25,
    // Frame analysis interval for webcam (ms)
    FRAME_INTERVAL: 500
};
}),
"[project]/lib/emotion-api.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "analyzeFrame",
    ()=>analyzeFrame,
    "analyzeImage",
    ()=>analyzeImage,
    "analyzeText",
    ()=>analyzeText,
    "analyzeTextLocally",
    ()=>analyzeTextLocally,
    "checkBackendHealth",
    ()=>checkBackendHealth,
    "generateDemoEmotion",
    ()=>generateDemoEmotion,
    "transformBackendEmotionResult",
    ()=>transformBackendEmotionResult
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$config$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/config.ts [app-route] (ecmascript)");
;
const EMOTION_LABELS = __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$config$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ML_CONFIG"].EMOTION_LABELS;
const EMPTY_DISTRIBUTION = {
    Angry: 0,
    Disgust: 0,
    Fear: 0,
    Happy: 0,
    Sad: 0,
    Surprise: 0,
    Neutral: 0
};
// Convert ML backend predictions array to EmotionDistribution
function predictionsToDistribution(predictions) {
    const total = predictions.reduce((a, b)=>a + b, 0);
    return {
        Angry: Math.round(predictions[0] / total * 100),
        Disgust: Math.round(predictions[1] / total * 100),
        Fear: Math.round(predictions[2] / total * 100),
        Happy: Math.round(predictions[3] / total * 100),
        Sad: Math.round(predictions[4] / total * 100),
        Surprise: Math.round(predictions[5] / total * 100),
        Neutral: Math.round(predictions[6] / total * 100)
    };
}
async function checkBackendHealth() {
    try {
        const response = await fetch(`${__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$config$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ML_CONFIG"].BACKEND_URL}${__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$config$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ML_CONFIG"].ENDPOINTS.HEALTH}`, {
            method: "GET",
            signal: AbortSignal.timeout(3000)
        });
        return response.ok;
    } catch  {
        return false;
    }
}
async function analyzeText(text) {
    const response = await fetch("/api/analyze/text", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text
        })
    });
    if (!response.ok) {
        throw new Error("Text analysis failed");
    }
    return response.json();
}
async function analyzeImage(imageData) {
    const response = await fetch("/api/analyze/image", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            image: imageData
        })
    });
    if (!response.ok) {
        throw new Error("Image analysis failed");
    }
    return response.json();
}
async function analyzeFrame(frameData) {
    const response = await fetch("/api/analyze/frame", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            frame: frameData
        })
    });
    if (!response.ok) {
        throw new Error("Frame analysis failed");
    }
    return response.json();
}
function analyzeTextLocally(text) {
    const lowerText = text.toLowerCase();
    const emotionKeywords = {
        Happy: [
            "happy",
            "joy",
            "love",
            "excited",
            "great",
            "wonderful",
            "amazing",
            "awesome",
            "fantastic",
            "delighted",
            "pleased",
            "glad",
            "cheerful",
            "thrilled",
            "elated",
            "good",
            "best",
            "beautiful"
        ],
        Sad: [
            "sad",
            "unhappy",
            "depressed",
            "cry",
            "tears",
            "miserable",
            "heartbroken",
            "grief",
            "sorrow",
            "disappointed",
            "lonely",
            "hopeless",
            "gloomy",
            "melancholy",
            "hurt",
            "pain"
        ],
        Angry: [
            "angry",
            "mad",
            "furious",
            "hate",
            "annoyed",
            "frustrated",
            "irritated",
            "outraged",
            "enraged",
            "livid",
            "bitter",
            "resentful",
            "hostile",
            "rage"
        ],
        Fear: [
            "afraid",
            "scared",
            "fear",
            "terrified",
            "anxious",
            "worried",
            "nervous",
            "panic",
            "dread",
            "frightened",
            "alarmed",
            "horrified",
            "petrified",
            "terror"
        ],
        Surprise: [
            "surprised",
            "shocked",
            "amazed",
            "astonished",
            "unexpected",
            "wow",
            "unbelievable",
            "startled",
            "stunned",
            "bewildered",
            "astounded",
            "whoa"
        ],
        Disgust: [
            "disgusted",
            "gross",
            "nasty",
            "revolting",
            "sick",
            "yuck",
            "ew",
            "horrible",
            "awful",
            "repulsed",
            "vile",
            "distaste"
        ],
        Neutral: [
            "okay",
            "fine",
            "normal",
            "alright",
            "so-so",
            "whatever",
            "meh",
            "indifferent",
            "average",
            "ordinary"
        ]
    };
    const scores = {
        Happy: 0,
        Sad: 0,
        Angry: 0,
        Fear: 0,
        Surprise: 0,
        Disgust: 0,
        Neutral: 10
    };
    Object.entries(emotionKeywords).forEach(([emotion, keywords])=>{
        keywords.forEach((keyword)=>{
            if (lowerText.includes(keyword)) {
                scores[emotion] += 15;
            }
        });
    });
    // Punctuation analysis
    const exclamationCount = (text.match(/!/g) || []).length;
    if (exclamationCount > 2) {
        scores.Happy += 5;
        scores.Surprise += 5;
        scores.Angry += 3;
    }
    // Caps analysis (shouting)
    const capsRatio = (text.match(/[A-Z]/g) || []).length / text.length;
    if (capsRatio > 0.5 && text.length > 10) {
        scores.Angry += 10;
        scores.Surprise += 5;
    }
    // Normalize
    const total = Object.values(scores).reduce((a, b)=>a + b, 0);
    const distribution = {
        Angry: Math.round(scores.Angry / total * 100),
        Disgust: Math.round(scores.Disgust / total * 100),
        Fear: Math.round(scores.Fear / total * 100),
        Happy: Math.round(scores.Happy / total * 100),
        Sad: Math.round(scores.Sad / total * 100),
        Surprise: Math.round(scores.Surprise / total * 100),
        Neutral: Math.round(scores.Neutral / total * 100)
    };
    const primaryEmotion = Object.entries(distribution).reduce((a, b)=>a[1] > b[1] ? a : b)[0];
    const maxScore = Math.max(...Object.values(distribution));
    const confidence = Math.min(95, maxScore + Math.random() * 15);
    return {
        primaryEmotion,
        confidence,
        distribution,
        timestamp: Date.now(),
        source: "text"
    };
}
function generateDemoEmotion(source) {
    const rawScores = EMOTION_LABELS.map(()=>Math.random() * 100);
    const total = rawScores.reduce((a, b)=>a + b, 0);
    const distribution = {
        Angry: Math.round(rawScores[0] / total * 100),
        Disgust: Math.round(rawScores[1] / total * 100),
        Fear: Math.round(rawScores[2] / total * 100),
        Happy: Math.round(rawScores[3] / total * 100),
        Sad: Math.round(rawScores[4] / total * 100),
        Surprise: Math.round(rawScores[5] / total * 100),
        Neutral: Math.round(rawScores[6] / total * 100)
    };
    const primaryEmotion = Object.entries(distribution).reduce((a, b)=>a[1] > b[1] ? a : b)[0];
    const maxScore = Math.max(...Object.values(distribution));
    const confidence = Math.min(98, maxScore + Math.random() * 20);
    return {
        primaryEmotion,
        confidence,
        distribution,
        timestamp: Date.now(),
        source,
        faceData: source === "webcam" ? {
            x: 150,
            y: 80,
            width: 200,
            height: 250
        } : undefined
    };
}
const EMOTION_LABEL_SET = new Set(EMOTION_LABELS);
function transformBackendEmotionResult(data, source) {
    const predictions = Array.isArray(data.predictions) ? data.predictions.map((value)=>typeof value === "number" ? value : Number(value) || 0) : [];
    const hasValidPredictions = predictions.length === EMOTION_LABELS.length;
    let distribution = {
        ...EMPTY_DISTRIBUTION
    };
    let confidence = Math.min(100, Math.max(0, (data.confidence ?? 0) * 100));
    if (hasValidPredictions) {
        distribution = predictionsToDistribution(predictions);
        const maxValue = Math.max(...predictions);
        const maxIndex = predictions.indexOf(maxValue);
        confidence = Number((maxValue / (predictions.reduce((sum, v)=>sum + v, 0) || 1) * 100).toFixed(1));
        return {
            primaryEmotion: EMOTION_LABELS[maxIndex],
            confidence,
            distribution,
            timestamp: Date.now(),
            source,
            faceData: data.face_bbox ? {
                x: data.face_bbox[0],
                y: data.face_bbox[1],
                width: data.face_bbox[2],
                height: data.face_bbox[3]
            } : undefined
        };
    }
    const fallbackEmotion = typeof data.emotion === "string" && EMOTION_LABEL_SET.has(data.emotion) ? data.emotion : "Neutral";
    return {
        primaryEmotion: fallbackEmotion,
        confidence,
        distribution,
        timestamp: Date.now(),
        source,
        faceData: data.face_bbox ? {
            x: data.face_bbox[0],
            y: data.face_bbox[1],
            width: data.face_bbox[2],
            height: data.face_bbox[3]
        } : undefined
    };
}
}),
"[project]/app/api/analyze/text/route.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "POST",
    ()=>POST
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/server.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$config$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/config.ts [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$emotion$2d$api$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/emotion-api.ts [app-route] (ecmascript)");
;
;
;
async function POST(request) {
    try {
        const { text } = await request.json();
        if (!text || typeof text !== "string") {
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
                error: "Invalid text input"
            }, {
                status: 400
            });
        }
        // Try ML backend first
        try {
            const backendResponse = await fetch(`${__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$config$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ML_CONFIG"].BACKEND_URL}${__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$config$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ML_CONFIG"].ENDPOINTS.ANALYZE_TEXT}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    text
                }),
                signal: AbortSignal.timeout(5000)
            });
            if (backendResponse.ok) {
                const data = await backendResponse.json();
                return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json(data);
            }
        } catch  {
            // ML backend unavailable, use fallback
            console.log("[v0] ML backend unavailable, using local text analysis");
        }
        // Fallback to local analysis
        const result = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$emotion$2d$api$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__["analyzeTextLocally"])(text);
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json(result);
    } catch (error) {
        console.error("Text analysis error:", error);
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
            error: "Analysis failed"
        }, {
            status: 500
        });
    }
}
}),
];

//# sourceMappingURL=%5Broot-of-the-server%5D__4543e2ff._.js.map