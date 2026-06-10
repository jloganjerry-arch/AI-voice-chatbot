
import os
import uuid
import urllib.parse
from flask import Flask, jsonify, make_response, render_template, request, session
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "lynx-ai-secret-key-2024")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Main chat system prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are LYNX AI, a powerful, versatile, and intelligent AI assistant designed to help with any topic or task. You are knowledgeable, creative, and always precise in your communication.

Your capabilities include:
- Answering questions on any subject: science, technology, history, culture, arts, business, and more
- Writing, editing, summarizing, and generating creative content
- Coding assistance, debugging, and technical explanations
- Data analysis, research, and problem-solving
- Brainstorming ideas and providing strategic advice
- Engaging in natural, meaningful conversation on any topic
- Generating real images from text descriptions using integrated AI image generation

Guidelines:
- Be concise, clear, and direct. Avoid unnecessary filler.
- Adapt your tone to the context: professional for technical topics, friendly for casual conversation.
- When appropriate, structure responses with bullet points or numbered steps for clarity.
- Always be honest — if you don't know something, say so clearly.
- Remember the full conversation history to provide contextually relevant, continuous assistance.

You are LYNX AI. Intelligent, adaptable, and always here to help."""


# ── Image Prompt Engineer system prompt ───────────────────────────────
IMAGE_PROMPT_ENGINEER = """You are an AI Image Prompt Engineer.

When a user requests an image, convert their request into a highly detailed professional image-generation prompt.

Rules:
- Generate visually rich descriptions.
- Include lighting, environment, camera angle, quality, mood, colors, and artistic details.
- Produce prompts optimized for AI image models such as FLUX, SDXL, DALL-E, or Stable Diffusion.
- Default to ultra-realistic 4K quality unless the user requests another style.
- Include composition, depth, textures, and cinematic details.
- Never explain the prompt. Never add any extra text, preamble, or commentary.
- Output ONLY the final image prompt — nothing else. No quotes, no labels, no explanation.

Example:
User: Generate a lion image.
Output: Ultra-realistic majestic African lion standing on a rocky cliff during golden sunset, detailed fur textures, cinematic lighting, dramatic sky, shallow depth of field, ultra-sharp focus, 4K HDR photography, National Geographic style, highly detailed, professional wildlife photography."""


def _ensure_chat_state():
    if "chat_history" not in session:
        first_id = str(uuid.uuid4())
        session["chat_history"] = {
            first_id: {"title": "New Session", "messages": []}
        }
        session["active_chat_id"] = first_id


def _build_groq_messages(chat_messages):
    formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
    formatted.extend(
        {"role": msg["role"], "content": msg["content"]} for msg in chat_messages
    )
    return formatted


def _active_chat():
    _ensure_chat_state()
    chat_history = session["chat_history"]
    active_chat_id = session.get("active_chat_id")
    if active_chat_id not in chat_history:
        active_chat_id = next(iter(chat_history.keys()))
        session["active_chat_id"] = active_chat_id
    return active_chat_id, chat_history[active_chat_id]


@app.route("/", methods=["GET"])
def home():
    _ensure_chat_state()
    response = make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/state", methods=["GET"])
def get_state():
    active_chat_id, active_chat = _active_chat()
    return jsonify({
        "active_chat_id": active_chat_id,
        "active_messages": active_chat["messages"],
        "chats": [
            {"id": chat_id, "title": chat["title"]}
            for chat_id, chat in session["chat_history"].items()
        ],
    })


@app.route("/new-chat", methods=["POST"])
def new_chat():
    _ensure_chat_state()
    new_id = str(uuid.uuid4())
    session["chat_history"][new_id] = {"title": "New Session", "messages": []}
    session["active_chat_id"] = new_id
    session.modified = True
    return jsonify({"ok": True, "active_chat_id": new_id})


@app.route("/switch-chat", methods=["POST"])
def switch_chat():
    _ensure_chat_state()
    chat_id = request.json.get("chat_id", "")
    if chat_id in session["chat_history"]:
        session["active_chat_id"] = chat_id
        session.modified = True
        active_chat = session["chat_history"][chat_id]
        return jsonify({"ok": True, "messages": active_chat["messages"]})
    return jsonify({"ok": False, "error": "Chat not found"}), 404


@app.route("/delete-chat", methods=["POST"])
def delete_chat():
    _ensure_chat_state()
    chat_id = request.json.get("chat_id", "")
    chat_history = session["chat_history"]
    if chat_id not in chat_history:
        return jsonify({"ok": False, "error": "Chat not found"}), 404
    del chat_history[chat_id]
    if not chat_history:
        new_id = str(uuid.uuid4())
        chat_history[new_id] = {"title": "New Session", "messages": []}
        session["active_chat_id"] = new_id
    elif session.get("active_chat_id") == chat_id:
        session["active_chat_id"] = next(iter(chat_history.keys()))
    session["chat_history"] = chat_history
    session.modified = True
    return jsonify({"ok": True, "active_chat_id": session["active_chat_id"]})


@app.route("/chat", methods=["POST"])
def chat():
    _ensure_chat_state()
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"ok": False, "error": "Message cannot be empty"}), 400

    active_chat_id, active_chat = _active_chat()
    active_chat["messages"].append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=_build_groq_messages(active_chat["messages"]),
        temperature=0.7,
        max_tokens=1024,
    )
    reply = response.choices[0].message.content
    active_chat["messages"].append({"role": "assistant", "content": reply})

    if active_chat["title"] == "New Session":
        active_chat["title"] = user_input[:35] + ("..." if len(user_input) > 35 else "")

    session["chat_history"][active_chat_id] = active_chat
    session.modified = True
    return jsonify({"ok": True, "reply": reply})


@app.route("/enhance-prompt", methods=["POST"])
def enhance_prompt():
    """Use the Image Prompt Engineer AI to convert a user request into
    a rich, detailed image-generation prompt. Returns only the prompt string."""
    user_request = (request.json or {}).get("request", "").strip()
    if not user_request:
        return jsonify({"ok": False, "error": "Request is required"}), 400

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": IMAGE_PROMPT_ENGINEER},
                {"role": "user",   "content": user_request},
            ],
            temperature=0.8,
            max_tokens=300,
        )
        enhanced = response.choices[0].message.content.strip()
        # Strip any surrounding quotes the model might add
        enhanced = enhanced.strip('"\'')
        return jsonify({"ok": True, "prompt": enhanced})
    except Exception as e:
        # Fallback: return the original request as the prompt
        return jsonify({"ok": True, "prompt": user_request, "fallback": True})


@app.route("/generate-image", methods=["POST"])
def generate_image():
    """Return a Pollinations.ai image URL for the given prompt."""
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    model  = data.get("model", "flux")
    width  = int(data.get("width",  1024))
    height = int(data.get("height", 1024))

    if not prompt:
        return jsonify({"ok": False, "error": "Prompt is required"}), 400

    encoded = urllib.parse.quote(prompt)
    image_url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?model={model}&width={width}&height={height}&nologo=true"
    )
    return jsonify({"ok": True, "url": image_url, "prompt": prompt})


if __name__ == "__main__":
    app.run(debug=True)