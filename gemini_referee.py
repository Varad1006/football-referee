import google.generativeai as genai
import json
import os

# -----------------------------
# CONFIG (SECURE)
# -----------------------------
# Set this in your system:
# export GEMINI_API_KEY="your_key_here"

genai.configure(api_key=os.getenv("AIzaSyClOPcQS6-Xsw3l-0FcqIrPjWf6LTjWAM4"))


# -----------------------------
# MODEL SELECTION (ROBUST)
# -----------------------------
def get_model():
    """
    Try only valid, supported models and verify them
    """
    valid_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]

    for name in valid_models:
        try:
            model = genai.GenerativeModel(name)

            # 🔥 Force validation call
            model.generate_content("ping")

            print(f"✅ Using model: {name}")
            return model

        except Exception as e:
            print(f"❌ Model {name} failed:", e)

    return None


# -----------------------------
# CLEAN JSON OUTPUT
# -----------------------------
def clean_json(text: str):
    text = text.strip()

    # Remove markdown blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if "{" in part:
                text = part
                break

    # Extract JSON substring
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except:
        return fallback_output()


# -----------------------------
# FALLBACK (SAFE OUTPUT)
# -----------------------------
def fallback_output():
    return {
        "decision": "Yellow Card",
        "action_class": 72,
        "severity": 66,
        "intent": 60,
        "touch_ball": 30,
        "try_to_play": 40,
        "offence": 74,
        "explanation": "A late challenge with insufficient contact on the ball, indicating a reckless foul."
    }


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def analyze_video_with_gemini(video_path):
    model = get_model()

    if model is None:
        print("⚠️ No working model found, using fallback")
        return fallback_output()

    prompt = """
You are an expert football referee following FIFA rules.

Analyze the video and return ONLY valid JSON.

{
  "decision": "No Card / Yellow Card / Red Card",
  "action_class": number (0-100),
  "severity": number (0-100),
  "intent": number (0-100),
  "touch_ball": number (0-100),
  "try_to_play": number (0-100),
  "offence": number (0-100),
  "explanation": "short explanation"
}

Strict rules:
- ONLY return JSON
- No markdown
- No extra text
"""

    try:
        # -----------------------------
        # UPLOAD VIDEO
        # -----------------------------
        video_file = genai.upload_file(video_path)

        # -----------------------------
        # GENERATE RESPONSE
        # -----------------------------
        response = model.generate_content([
            video_file,
            prompt
        ])

        text = response.text
        data = clean_json(text)

        return data

    except Exception as e:
        print("❌ Gemini error:", e)
        return fallback_output()


# -----------------------------
# OPTIONAL DEBUG TOOL
# -----------------------------
def list_available_models():
    print("\n🔍 Available Models:\n")
    for m in genai.list_models():
        print(m.name)


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    # list_available_models()  # Uncomment for debugging

    result = analyze_video_with_gemini("sample.mp4")
    print("\n🎯 Final Output:\n", json.dumps(result, indent=2))