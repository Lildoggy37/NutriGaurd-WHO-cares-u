import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        solar: {
          50: "#fefce8",
          100: "#fef3c7",
          200: "#fde68a",
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b",
        },
        leaf: {
          50: "#f0fdf4",
          100: "#dcfce7",
          200: "#bbf7d0",
          300: "#86efac",
          400: "#4ade80",
          500: "#22c55e",
          600: "#16a34a",
          700: "#15803d",
          800: "#166534",
          900: "#14532d",
        },
        earth: {
          50: "#faf6f0",
          100: "#f5ecd9",
          200: "#ebd9b3",
          300: "#d9c08c",
          400: "#c4a565",
        },
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
        organic: "42% 58% 63% 37% / 45% 42% 58% 55%",
      },
      fontFamily: {
        body: ["var(--font-body)", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-soft": "pulseSoft 2s ease-in-out infinite",
        "leaf-sway": "leafSway 3s ease-in-out infinite",
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.4s ease-out",
      },
      keyframes: {
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
        leafSway: {
          "0%, 100%": { transform: "rotate(-2deg)" },
          "50%": { transform: "rotate(2deg)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
