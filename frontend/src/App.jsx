function App() {
  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>MicroLLM</h1>
          <p>48.78M parameter language model</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Model Ready
        </div>
      </header>

      <main className="main">
        <section className="chat-panel">
          <div className="welcome">
            <h2>Talk to MicroLLM</h2>
            <p>
              A language model trained from scratch under a
              50M-parameter budget.
            </p>
          </div>

          <div className="messages">
            <div className="message user">
              <span className="label">You</span>
              <p>Explain what self-attention does.</p>
            </div>

            <div className="message assistant">
              <span className="label">MicroLLM</span>
              <p>
                Self-attention allows each token to consider
                information from other tokens in the sequence.
              </p>
            </div>
          </div>

          <div className="input-area">
            <textarea
              placeholder="Ask MicroLLM something..."
              rows="1"
            />

            <button>Send</button>
          </div>
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