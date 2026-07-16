import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import { defineConfig } from "eslint/config";

// Backend (Node/TypeScript) lint config. The Next.js dashboard has its own
// ESLint setup under dashboard/.
export default defineConfig([
  { ignores: ["dist/**", "dashboard/**", "node_modules/**"] },
  { files: ["**/*.{js,mjs,cjs,ts,mts,cts}"], plugins: { js }, extends: ["js/recommended"] },
  { files: ["**/*.{js,mjs,cjs,ts,mts,cts}"], languageOptions: { globals: { ...globals.node } } },
  tseslint.configs.recommended,
  {
    rules: {
      // Allow intentionally-unused identifiers when prefixed with `_`.
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_"
        }
      ]
    }
  },
  {
    // Test code legitimately uses `any` for mock/fixture data and augments
    // globals for the test harness.
    files: ["tests/**/*.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-namespace": "off",
      "no-var": "off"
    }
  }
]);
