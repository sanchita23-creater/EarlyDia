import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ThemeProvider } from "./context/ThemeContext";
import { ScreeningProvider } from "./context/ScreeningContext";
import "./styles/global.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ThemeProvider>
      <ScreeningProvider>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </ScreeningProvider>
    </ThemeProvider>
  </React.StrictMode>
);
