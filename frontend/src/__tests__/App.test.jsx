import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { vi, test, expect, beforeEach, afterEach } from "vitest"; 
import App from "../App";

// --- MOCK SETUP ---
beforeEach(() => {
  // Fetch Mock
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      body: {
        getReader: () => ({
          read: () => Promise.resolve({ done: true, value: new Uint8Array() }), 
        }),
      },
    })
  );

  // ScrollIntoView Mock 
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  vi.clearAllMocks();
});


test("renders empty state with prompt", () => {
  render(<App />);
  expect(screen.getByText(/Ask finance anything/i)).toBeInTheDocument();
});

test("clicking chip pre-fills the composer textarea", () => {
  render(<App />);
  const chip = screen.getByRole("button", { name: /Summarize today's market movers/i });
  fireEvent.click(chip);
  const textarea = screen.getByPlaceholderText(/Ask about stocks/i);
  expect(textarea.value).toBe("Summarize today's market movers");
});

test("updates input value when typing", () => {
  render(<App />);
  const textarea = screen.getByPlaceholderText(/Ask about stocks/i);
  fireEvent.change(textarea, { target: { value: "Hello AI" } });
  expect(textarea.value).toBe("Hello AI");
});

test("toggles online research button", () => {
  render(<App />);
  const toggleBtn = screen.getByTitle(/Online research:/i);
  expect(toggleBtn).not.toHaveClass("active");
  fireEvent.click(toggleBtn);
  expect(toggleBtn).toHaveClass("active");
});

test("sends a message and clears input", async () => {
  render(<App />);
  const textarea = screen.getByPlaceholderText(/Ask about stocks/i);
  const sendButton = screen.getByRole("button", { name: /Send/i });

  fireEvent.change(textarea, { target: { value: "What is NVDA price?" } });
  fireEvent.click(sendButton);

  expect(global.fetch).toHaveBeenCalledTimes(1);
  
  await waitFor(() => {
    expect(screen.getByText("What is NVDA price?")).toBeInTheDocument();
  });

  expect(textarea.value).toBe("");
});