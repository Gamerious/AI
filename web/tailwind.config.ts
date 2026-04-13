import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0B1120",
        paper: "#FAFBFC",
        primary: {
          DEFAULT: "#1E3A8A",
          foreground: "#FFFFFF",
          50: "#EFF2FB",
          100: "#D8E0F4",
          500: "#3B5FC7",
          600: "#1E3A8A",
          700: "#152B6B",
          900: "#0B1749",
        },
        accent: {
          citrus: "#F59E0B",
          leaf: "#10B981",
          crimson: "#DC2626",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      maxWidth: {
        content: "72ch",
      },
    },
  },
  plugins: [],
};

export default config;
