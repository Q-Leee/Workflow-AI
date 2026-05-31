import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth/AuthContext";
import Layout from "./components/Layout";
import LoginPage from "./pages/LoginPage";
import PdfQaPage from "./pages/PdfQaPage";
import ResumeMatchPage from "./pages/ResumeMatchPage";
import MeetingPage from "./pages/MeetingPage";

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <p className="muted">Loading…</p>;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }
      >
        <Route index element={<Navigate to="/pdf" replace />} />
        <Route path="pdf" element={<PdfQaPage />} />
        <Route path="match" element={<ResumeMatchPage />} />
        <Route path="meeting" element={<MeetingPage />} />
      </Route>
    </Routes>
  );
}
