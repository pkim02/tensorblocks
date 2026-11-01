import { Component } from "react";

// Keeps a demo alive: a crash shows a way back instead of a blank page.
export default class ErrorBoundary extends Component {
  state = { error: null };
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="app">
        <div className="empty">
          <h2>화면을 그리다가 문제가 생겼어요</h2>
          <p>{String(this.state.error.message || this.state.error)}</p>
          <button className="btn primary" onClick={() => this.setState({ error: null })}>
            다시 그리기
          </button>
        </div>
      </div>
    );
  }
}
