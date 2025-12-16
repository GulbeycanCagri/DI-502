import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import App from "../App.jsx";

test("renders empty state with prompt", () => {
  render(<App />);
  expect(screen.getByText(/Ask finance anything/i)).toBeInTheDocument();
});

test("clicking chip pre-fills the composer textarea", () => {
  render(<App />);
  const chip = screen.getByRole("button", { name: /Summarize today's market movers/i });
  fireEvent.click(chip);
  const textarea = screen.getByPlaceholderText(/Ask about stocks, ETFs, bonds, crypto, macro…/i);
  expect(textarea.value).toBe("Summarize today's market movers");
});