
import os
import uuid
from flask import Flask, jsonify, render_template, request, session
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "lynx-ai-secret-key-2024")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are LYNX AI, a powerful, versatile, and intelligent AI assistant designed to help with any topic or task. You are knowledgeable, creative, and always precise in your communication.

Your capabilities include:
- Answering questions on any subject: science, technology, history, culture, arts, business, and more
- Writing, editing, summarizing, and generating creative content
- Coding assistance, debugging, and technical explanations
- Data analysis, research, and problem-solving
- Brainstorming ideas and providing strategic advice
- Engaging in natural, meaningful conversation on any topic

Guidelines:
- Be concise, clear, and direct. Avoid unnecessary filler.
- Adapt your tone to the context: professional for technical topics, friendly for casual conversation.
- When appropriate, structure responses with bullet points or numbered steps for clarity.
- Always be honest — if you don't know something, say so clearly.
- Remember the full conversation history to provide contextually relevant, continuous assistance.

You are LYNX AI. Intelligent, adaptable, and always here to help."""


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
    return render_template("index.html")


@app.route("/state", methods=["GET"])
def get_state():
    active_chat_id, active_chat = _active_chat()
    return jsonify(
        {
            "active_chat_id": active_chat_id,
            "active_messages": active_chat["messages"],
            "chats": [
                {"id": chat_id, "title": chat["title"]}
                for chat_id, chat in session["chat_history"].items()
            ],
        }
    )


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

    # Use the first user message as the chat title.
    if active_chat["title"] == "New Session":
        active_chat["title"] = user_input[:35] + ("..." if len(user_input) > 35 else "")

    session["chat_history"][active_chat_id] = active_chat
    session.modified = True

    return jsonify({"ok": True, "reply": reply})


if __name__ == "__main__":
    app.run(debug=True)