import { test, expect } from "bun:test";
import { DEFAULT_PROMPT, assistantText, createDescriber, parseModel, resolveSettings } from "../src/vision";

const noEnv = {};

test("parseModel splits on the first slash only", () => {
  expect(parseModel("google-vertex/gemini-3.8-flash")).toEqual({ providerID: "google-vertex", id: "gemini-3.8-flash" });
  expect(parseModel("openrouter/anthropic/claude-x")).toEqual({ providerID: "openrouter", id: "anthropic/claude-x" });
  for (const bad of ["nodash", "/x", "x/", "a b/c", "a/b c", ""]) expect(parseModel(bad)).toBeUndefined();
});

test("resolveSettings returns undefined when no model is configured", async () => {
  expect(await resolveSettings(undefined, noEnv)).toBeUndefined();
  expect(await resolveSettings({ model: "" }, { OPENCODE_VISION_MODEL: "" })).toBeUndefined();
});

test("resolveSettings reads the model from options, then env", async () => {
  expect((await resolveSettings({ model: "a/b" }, { OPENCODE_VISION_MODEL: "c/d" }))?.model).toEqual({ providerID: "a", id: "b" });
  expect((await resolveSettings(undefined, { OPENCODE_VISION_MODEL: "c/d" }))?.model).toEqual({ providerID: "c", id: "d" });
  expect((await resolveSettings({ model: "a/b" }, noEnv))?.prompt).toBe(DEFAULT_PROMPT);
});

test("resolveSettings rejects a malformed model", async () => {
  await expect(resolveSettings({ model: "nodash" }, noEnv)).rejects.toThrow("providerID/modelID");
});

test("resolveSettings rejects a relative promptFile without reading it", async () => {
  let reads = 0;
  const read = async () => (reads++, "x");
  await expect(resolveSettings({ model: "a/b", promptFile: "./prompt.md" }, noEnv, read)).rejects.toThrow("absolute");
  await expect(resolveSettings({ model: "a/b", promptFile: "prompt.md" }, noEnv, read)).rejects.toThrow("absolute");
  expect(reads).toBe(0);
});

test("resolveSettings reads an absolute promptFile; an inline prompt wins; an empty file falls back", async () => {
  const read = async (path: string) => `  from ${path}  \n`;
  expect((await resolveSettings({ model: "a/b", promptFile: "/etc/p.md" }, noEnv, read))?.prompt).toBe("from /etc/p.md");
  expect((await resolveSettings({ model: "a/b", promptFile: "/etc/p.md", prompt: "inline" }, noEnv, read))?.prompt).toBe("inline");
  expect((await resolveSettings({ model: "a/b", promptFile: "/etc/p.md" }, noEnv, async () => " "))?.prompt).toBe(DEFAULT_PROMPT);
});

test("assistantText returns the last assistant text and fails with short messages", () => {
  const msgs = [
    { type: "user", content: [{ type: "text", text: "q" }] },
    { type: "assistant", content: [{ type: "reasoning", text: "hmm" }, { type: "text", text: "old" }] },
    { type: "assistant", content: [{ type: "text", text: "new " }, { type: "text", text: "answer" }] },
  ];
  expect(assistantText(msgs)).toBe("new answer");
  expect(() => assistantText([msgs[0]])).toThrow("no answer");
  expect(() => assistantText([{ type: "assistant", content: [] }])).toThrow("no text");
  expect(() => assistantText([{ type: "assistant", content: [], error: { message: "secret body" } }])).toThrow(
    "vision model returned an error",
  );
  // partial text together with an error is a failure, not a result
  expect(() =>
    assistantText([{ type: "assistant", content: [{ type: "text", text: "half an answ" }], error: { message: "cut" } }]),
  ).toThrow("vision model returned an error");
});

test("a session created after the timeout is still removed", async () => {
  const removed: string[] = [];
  const internal = new Set<string>();
  const session = {
    create: async () => (await new Promise((r) => setTimeout(r, 40)), { id: "late" }),
    prompt: async () => {},
    wait: async () => {},
    context: async () => [],
    remove: async ({ sessionID }: { sessionID: string }) => void removed.push(sessionID),
  };
  const describe = createDescriber(session, { model: { providerID: "a", id: "b" }, prompt: "p" }, internal, () => {}, 10);
  await expect(describe({ base64: "AAAA", mime: "image/png" })).rejects.toThrow("timed out");
  await new Promise((r) => setTimeout(r, 80));
  expect(removed).toEqual(["late"]);
  expect(internal.size).toBe(0);
});

test("a remove() that never settles does not block the describer beyond the cleanup deadline", async () => {
  const internal = new Set<string>();
  const logs: string[] = [];
  const session = {
    create: async () => ({ id: "s1" }),
    prompt: async () => {},
    wait: async () => {},
    context: async () => [{ type: "assistant", content: [{ type: "text", text: "ok" }] }],
    remove: () => new Promise<void>(() => {}),
  };
  const describe = createDescriber(session, { model: { providerID: "a", id: "b" }, prompt: "p" }, internal, (m) => logs.push(m), 1000, false, 20);
  const started = Date.now();
  expect(await describe({ base64: "AAAA", mime: "image/png" })).toBe("ok");
  expect(Date.now() - started).toBeLessThan(500);
  expect(internal.has("s1")).toBe(false);
  expect(logs).toEqual(["could not remove vision session: remove timed out"]);
});
