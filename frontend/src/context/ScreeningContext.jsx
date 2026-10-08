import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { getConfig } from "../services/api";
import { emptyAnswers } from "../utils/answers";

const Ctx = createContext(null);
export const useScreening = () => useContext(Ctx);

const KEY = "earlydia-session";
const load = () => {
  try { return JSON.parse(sessionStorage.getItem(KEY)) || {}; } catch { return {}; }
};

export function ScreeningProvider({ children }) {
  const saved = load();
  const [answers, setAnswers] = useState({ ...emptyAnswers(), ...(saved.answers || {}) });
  const [result, setResult] = useState(saved.result || null);
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState(null);

  useEffect(() => {
    try { sessionStorage.setItem(KEY, JSON.stringify({ answers, result })); } catch { /* ignore */ }
  }, [answers, result]);

  const loadConfig = useCallback(() => {
    setConfigError(null);
    getConfig().then(setConfig).catch((e) => setConfigError(e.message));
  }, []);
  useEffect(() => { loadConfig(); }, [loadConfig]);

  const update = useCallback((patch) => setAnswers((a) => ({ ...a, ...patch })), []);
  const reset = useCallback(() => { setAnswers(emptyAnswers()); setResult(null); }, []);

  return (
    <Ctx.Provider value={{ answers, update, result, setResult, reset, config, configError, loadConfig }}>
      {children}
    </Ctx.Provider>
  );
}
