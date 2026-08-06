/**
 * Inline SVG icons.
 *
 * An icon package would be 20-60KB for the handful of glyphs used here, all of
 * which are a few paths. `currentColor` throughout, so they inherit hover and
 * active states rather than needing their own rules.
 */

const base = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export function ArrowUp({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base} strokeWidth={2.25}>
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  );
}

export function Paperclip({ size = 17 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base}>
      <path d="M21 11.5l-8.6 8.6a5 5 0 0 1-7-7l8.5-8.6a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.6a1.7 1.7 0 0 1-2.4-2.4l7.9-7.9" />
    </svg>
  );
}

export function Layers({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base}>
      <path d="M12 3l9 5-9 5-9-5 9-5z" />
      <path d="M3 13l9 5 9-5M3 17l9 5 9-5" />
    </svg>
  );
}

export function Books({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base}>
      <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H19v18H6.5A2.5 2.5 0 0 0 4 22V4.5z" />
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H19" />
    </svg>
  );
}

export function Close({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base}>
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

export function Info({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      {...base}
      style={{ flexShrink: 0, marginTop: 2 }}
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" />
    </svg>
  );
}

export function UploadCloud({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base}>
      <path d="M17 17h1a4 4 0 0 0 .3-8A6 6 0 0 0 6.7 8.3 3.5 3.5 0 0 0 7 17h1" />
      <path d="M12 12v8M9 15l3-3 3 3" />
    </svg>
  );
}

export function Sparkle({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base}>
      <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" />
    </svg>
  );
}

export function Plus({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...base}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}
