import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

export default function Layout() {
  const { user, logout } = useAuth();

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>WorkFlow AI</h1>
          <p className="muted">RAG · Resume match · Meeting notes</p>
        </div>
        <div className="header-right">
          <span className="muted">{user?.email}</span>
          <button type="button" className="btn ghost" onClick={logout}>
            Log out
          </button>
        </div>
      </header>

      <nav className="nav">
        <NavLink to="/pdf">Document Q&A</NavLink>
        <NavLink to="/match">Resume / JD</NavLink>
        <NavLink to="/meeting">Meeting</NavLink>
      </nav>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
