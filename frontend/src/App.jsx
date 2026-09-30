import { useState } from "react";

function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

async function handleSend() {
  const text = prompt.trim();

  if (!text || loading) {
    return;
  }

  setError("");
  setPrompt("");
  setLoading(true);

  setMessages((prev) => [
    ...prev,
    {
      role: "user",
      text,
    },
    {
      role: "assistant",
      text: "",
    },
  ]);

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/generate/stream",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: text,
          max_new_tokens: 80,
          temperature: 0.8,
          top_k: 50,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Server returned ${response.status}`
      );
    }

    if (!response.body) {
      throw new Error("Streaming not supported.");
    }

    const reader =
      response.body.getReader();

    const decoder = new TextDecoder();

    let buffer = "";

    while (true) {
      const { value, done } =
        await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(
        value,
        { stream: true }
      );

      const events = buffer.split(
        "\n\n"
      );

      buffer = events.pop() || "";

      for (const event of events) {
        const line = event
          .split("\n")
          .find((line) =>
            line.startsWith("data:")
          );

        if (!line) {
          continue;
        }

        const data = JSON.parse(
          line.slice(5).trim()
        );

        if (data.done) {
          continue;
        }

        setMessages((prev) => {
          const updated = [...prev];

          const lastIndex =
            updated.length - 1;

          updated[lastIndex] = {
            ...updated[lastIndex],
            text:
              updated[lastIndex].text +
              data.text,
          };

          return updated;
        });
      }
    }
  } catch (err) {
    console.error(err);

    setError("Could not connect to MicroLLM backend.");
  } finally {
    setLoading(false);
  }
}

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>MicroLLM</h1>
          <p>
            48.78M parameter language model
          </p>
        </div>

        <div className="status">
          <span className="status-dot"></span>

          <span>
            {loading
              ? "Generating..."
              : "Model Ready"}
          </span>
        </div>
      </header>

      <main className="main">
        <section className="chat-panel">
          <div className="welcome">
            <h2>Talk to MicroLLM</h2>

            <p>
              A language model trained from scratch
              under a 50M-parameter budget.
            </p>
          </div>

          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <p>
                  Ask MicroLLM a question to begin.
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${
                  message.role
                }`}
              >
                <span className="label">
                  {message.role === "user"
                    ? "You"
                    : "MicroLLM"}
                </span>

                <p>{message.text}</p>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <span className="label">
                  MicroLLM
                </span>

                <p>Generating...</p>
              </div>
            )}
          </div>

          <div className="input-area">
            <textarea
              value={prompt}
              onChange={(event) =>
                setPrompt(event.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="Ask MicroLLM something..."
              rows="1"
              disabled={loading}
            />

            <button
              onClick={handleSend}
              disabled={
                loading || !prompt.trim()
              }
            >
              {loading ? "..." : "Send"}
            </button>
          </div>

          {error && (
            <p
              style={{
                color: "#ff7b7b",
                padding: "0 18px 18px",
                margin: 0,
              }}
            >
              {error}
            </p>
          )}
        </section>

        <aside className="model-panel">
          <h3>Model</h3>

          <div className="stat">
            <span>Parameters</span>
            <strong>48.78M</strong>
          </div>

          <div className="stat">
            <span>Layers</span>
            <strong>27</strong>
          </div>

          <div className="stat">
            <span>Hidden Size</span>
            <strong>384</strong>
          </div>

          <div className="stat">
            <span>Attention</span>
            <strong>GQA</strong>
          </div>

          <div className="stat">
            <span>Vocabulary</span>
            <strong>16,384</strong>
          </div>

          <div className="stat">
            <span>Context</span>
            <strong>1,024</strong>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;