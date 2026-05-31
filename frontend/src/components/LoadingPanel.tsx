type Props = {
  label: string;
  sublabel?: string;
};

export default function LoadingPanel({ label, sublabel }: Props) {
  return (
    <div className="loading-panel" role="status" aria-live="polite" aria-busy="true">
      <div className="spinner" aria-hidden="true" />
      <p className="loading-label running">{label}</p>
      {sublabel && <p className="loading-sublabel muted">{sublabel}</p>}
    </div>
  );
}
