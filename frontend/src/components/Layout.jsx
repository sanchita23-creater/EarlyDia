import { Link, NavLink, Outlet } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { MoonIcon, SunIcon, PulseIcon } from "./Icons";

export function Brand() {
  return (
    <Link to="/" className="brand" aria-label="EarlyDia home">
      <span className="brand-mark"><PulseIcon width={18} height={18} strokeWidth={2.4} /></span>
      <span className="brand-name">EarlyDia</span>
    </Link>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const dark = theme === "dark";
  return (
    <button type="button" className="icon-btn" onClick={toggle} aria-pressed={dark}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} title={dark ? "Light mode" : "Dark mode"}>
      {dark ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

export default function Layout() {
  return (
    <div className="app">
      <header className="header">
        <div className="container header-in">
          <Brand />
          <nav className="nav" aria-label="Main">
            <NavLink to="/about">How it works</NavLink>
            <NavLink to="/research">Research</NavLink>
            <ThemeToggle />
          </nav>
        </div>
      </header>
      <main className="main"><Outlet /></main>
      <footer className="footer">
        <div className="container">EarlyDia is a screening research prototype, not a diagnostic test.</div>
      </footer>
    </div>
  );
}
