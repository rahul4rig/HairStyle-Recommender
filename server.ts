import dotenv from "dotenv";
dotenv.config();

import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import Groq from "groq-sdk";
import { GoogleGenAI, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json());

interface SuggestionItem {
  name: string;
  match_score: number;
  reason: string;
  styling_tip?: string;
}

// Helper to initialize Gemini AI client on the server
function getGeminiClient() {
  if (!process.env.GEMINI_API_KEY) {
    return null;
  }
  return new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// Server-side runtime storage for GROQ_API_KEY (simulating .env.local in live classroom workshops)
let runtimeGroqApiKey: string =
  process.env.GROQ_API_KEY &&
  process.env.GROQ_API_KEY !== "gsk_xxxxxxxxxxxxxxxxx" &&
  process.env.GROQ_API_KEY !== "MY_GROQ_API_KEY"
    ? process.env.GROQ_API_KEY.trim()
    : "";

function maskGroqKey(key: string): string | null {
  if (!key) return null;
  if (key.length <= 8) return "gsk_••••";
  return `${key.slice(0, 6)}••••••••${key.slice(-4)}`;
}

// Helper to initialize Groq client on the server
function getGroqClient(overrideKey?: string) {
  const effectiveKey = (overrideKey || runtimeGroqApiKey || "").trim();
  if (!effectiveKey || effectiveKey === "gsk_xxxxxxxxxxxxxxxxx") {
    return null;
  }
  return new Groq({
    apiKey: effectiveKey,
  });
}

/**
 * GET /api/config/groq-key
 * Returns whether a GROQ_API_KEY is currently configured on the server (masked).
 */
app.get("/api/config/groq-key", (_req, res) => {
  return res.status(200).json({
    configured: Boolean(runtimeGroqApiKey),
    masked_key: maskGroqKey(runtimeGroqApiKey),
  });
});

/**
 * POST /api/config/groq-key
 * Allows the instructor or student to paste their Groq API key (gsk_...) and store it in server memory (.env.local simulation).
 */
app.post("/api/config/groq-key", (req, res) => {
  const { api_key } = req.body || {};
  const cleaned = typeof api_key === "string" ? api_key.trim() : "";
  runtimeGroqApiKey = cleaned;

  return res.status(200).json({
    configured: Boolean(runtimeGroqApiKey),
    masked_key: maskGroqKey(runtimeGroqApiKey),
    message: runtimeGroqApiKey
      ? "GROQ_API_KEY stored on the server."
      : "GROQ_API_KEY cleared from server.",
  });
});

/**
 * POST /api/suggestions
 *
 * Implements the exact classroom flow:
 * Hairstyle UI -> Your API (/api/suggestions) -> Groq -> Your API -> UI
 *
 * Features:
 * - Step 10 Validation: Returns 400 if face_shape, hair_type, or hair_length is missing.
 * - Step 11 Error Handling: Returns 502 if upstream AI provider fails or is simulated to fail.
 * - Step 7 & Step 8 Format Support: Returns both `result` (raw string) and `suggestions` (structured JSON array).
 */
app.post("/api/suggestions", async (req, res) => {
  const startTime = Date.now();
  const {
    face_shape,
    hair_type,
    hair_length,
    style_preference,
    response_mode: bodyResponseMode,
    simulate_502: bodySimulate502,
  } = req.body || {};

  // Support query parameters (?format=structured|raw&simulate_502=true) in addition to body flags
  const queryFormat = req.query.format as string | undefined;
  const querySimulate502 = req.query.simulate_502 === "true";

  const response_mode = queryFormat || bodyResponseMode || "structured";
  const simulate_502 = querySimulate502 || Boolean(bodySimulate502);

  // Step 10: Add validation (400 Bad Request)
  if (
    !face_shape ||
    !hair_type ||
    !hair_length ||
    String(face_shape).trim() === "" ||
    String(hair_type).trim() === "" ||
    String(hair_length).trim() === ""
  ) {
    return res.status(400).json({
      error: "Missing required hairstyle information",
      missing_fields: [
        !face_shape || String(face_shape).trim() === "" ? "face_shape" : null,
        !hair_type || String(hair_type).trim() === "" ? "hair_type" : null,
        !hair_length || String(hair_length).trim() === "" ? "hair_length" : null,
      ].filter(Boolean),
    });
  }

  // Step 11: Handle Groq failure (502 Bad Gateway simulation or real upstream error)
  if (simulate_502) {
    return res.status(502).json({
      error: "Unable to generate hairstyle recommendations",
    });
  }

  const effectivePreference = style_preference || "classic versatile";

  const systemPrompt =
    "You are a professional master hairstylist and editorial grooming director who gives personalized, anatomically tailored hairstyle recommendations.";

  const userPrompt = `Recommend hairstyles for this person:

Face shape: ${face_shape}
Hair type: ${hair_type}
Hair length: ${hair_length}
Style preference: ${effectivePreference}

Recommend 3 hairstyles.
For each hairstyle give:
- name
- match_score (integer out of 100)
- reason (2 concise sentences explaining why it suits their ${face_shape} face shape, ${hair_type} texture, and ${effectivePreference} aesthetic)
- styling_tip (short practical product or blow-dry instruction)`;

  try {
    let suggestions: SuggestionItem[] = [];
    let rawTextResult = "";
    let modelUsed = "openai/gpt-oss-20b";

    const groq = getGroqClient();

    if (groq) {
      // Live Groq SDK execution when GROQ_API_KEY is configured in server environment
      let completion;
      try {
        completion = await groq.chat.completions.create({
          model: "openai/gpt-oss-20b",
          messages: [
            {
              role: "system",
              content:
                systemPrompt +
                ' Always respond with valid JSON matching the structure: { "suggestions": [{ "name": string, "match_score": number, "reason": string, "styling_tip": string }] }',
            },
            {
              role: "user",
              content: userPrompt,
            },
          ],
          response_format: { type: "json_object" },
        });
      } catch (groqPrimaryErr: unknown) {
        // Fallback to llama-3.3-70b-versatile if openai/gpt-oss-20b is not enabled on the user's Groq account
        modelUsed = "llama-3.3-70b-versatile";
        completion = await groq.chat.completions.create({
          model: "llama-3.3-70b-versatile",
          messages: [
            {
              role: "system",
              content:
                systemPrompt +
                ' Always respond with valid JSON matching the structure: { "suggestions": [{ "name": string, "match_score": number, "reason": string, "styling_tip": string }] }',
            },
            {
              role: "user",
              content: userPrompt,
            },
          ],
          response_format: { type: "json_object" },
        });
      }

      const content = completion.choices[0]?.message?.content || "{}";
      const parsed = JSON.parse(content);
      if (Array.isArray(parsed.suggestions)) {
        suggestions = parsed.suggestions;
      }
    } else {
      // Server-side AI execution via @google/genai when running in AI Studio sandbox
      const ai = getGeminiClient();
      if (!ai) {
        throw new Error("Server AI provider credentials unavailable");
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.8-flash",
        contents: userPrompt,
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              suggestions: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    name: {
                      type: Type.STRING,
                      description: "Name of the recommended hairstyle.",
                    },
                    match_score: {
                      type: Type.INTEGER,
                      description: "Match score out of 100 (e.g. 94).",
                    },
                    reason: {
                      type: Type.STRING,
                      description:
                        "Short reason explaining why this cut works for the face shape, hair type, and style preference.",
                    },
                    styling_tip: {
                      type: Type.STRING,
                      description: "Brief product and styling tip.",
                    },
                  },
                  required: ["name", "match_score", "reason", "styling_tip"],
                },
              },
            },
            required: ["suggestions"],
          },
        },
      });

      const jsonStr = (response.text || "{}").trim();
      const parsed = JSON.parse(jsonStr);
      if (Array.isArray(parsed.suggestions)) {
        suggestions = parsed.suggestions;
      }
    }

    if (!suggestions || suggestions.length === 0) {
      throw new Error("Empty suggestions returned from model");
    }

    // Build the Step 7 formatted text string as well so students can toggle between Step 7 (raw string) and Step 8 (structured JSON)
    rawTextResult = suggestions
      .map(
        (s, idx) =>
          `${idx + 1}. ${s.name} — ${s.match_score}/100\nReason: ${s.reason}${
            s.styling_tip ? `\nStyling Tip: ${s.styling_tip}` : ""
          }`
      )
      .join("\n\n");

    const durationMs = Date.now() - startTime;

    if (response_mode === "raw") {
      // Step 7 lecture output format
      return res.status(200).json({
        result: rawTextResult,
        _meta: {
          model: modelUsed,
          duration_ms: durationMs,
          endpoint: "POST /api/suggestions",
        },
      });
    }

    // Step 8 structured output format (includes both suggestions array and result string for compatibility)
    return res.status(200).json({
      suggestions,
      result: rawTextResult,
      _meta: {
        model: modelUsed,
        duration_ms: durationMs,
        endpoint: "POST /api/suggestions",
      },
    });
  } catch (error) {
    console.error("Error in POST /api/suggestions:", error);
    // Step 11: 502 Bad Gateway on AI provider failure
    return res.status(502).json({
      error: "Unable to generate hairstyle recommendations",
    });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*all", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
